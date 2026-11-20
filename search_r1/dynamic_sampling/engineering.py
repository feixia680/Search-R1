"""Transparent metrics for existing veRL token balancing."""
import math


def partition_summary(lengths, world_size):
    if world_size < 1 or len(lengths) % world_size:
        raise ValueError('Batch must split equally across ranks')
    width = len(lengths)//world_size
    tokens = [sum(lengths[i*width:(i+1)*width]) for i in range(world_size)]
    mean = sum(tokens)/world_size
    return dict(tokens=tokens, mean_tokens=mean,
                gap_percent=100*(max(tokens)-min(tokens))/max(1.,mean),
                cv_percent=100*math.sqrt(sum((x-mean)**2 for x in tokens)/world_size)/max(1.,mean))


def add_partition_metrics(metrics, before, after):
    for name, values in [('before',before),('after',after)]:
        for key in ('mean_tokens','gap_percent','cv_percent'):
            metrics[f'engineering/tokens_{name}/{key}'] = values[key]
        for rank,tokens in enumerate(values['tokens']):metrics[f'engineering/tokens_{name}/rank_{rank}']=tokens



def benchmark_updates(trainer, batch, initial_metrics):
    """Real FSDP updates with fixed inputs and LR=0; alternate measured variants."""
    import copy
    import hashlib
    import json
    import statistics
    import time
    import torch
    from verl.trainer.ppo.ray_trainer import reduce_metrics
    options = trainer.config.engineering
    if float(trainer.config.actor_rollout_ref.actor.optim.lr) != 0:
        raise ValueError('Benchmark requires lr=0')
    batch.save_to_disk(trainer.audit_dir/'engineering_workload.pkl')
    original = bool(options.sequence_balancing)
    def fingerprints(data):
        return sorted(hashlib.sha256(data.batch['input_ids'][i].cpu().numpy().tobytes()+
                      str(data.non_tensor_batch['uid'][i]).encode()).hexdigest() for i in range(len(data)))
    expected = fingerprints(batch)
    micros = sorted(set(options.get('profile_micro_batches', []) or
                       [int(trainer.config.actor_rollout_ref.actor.ppo_micro_batch_size)]))
    world = trainer.actor_rollout_wg.world_size
    for micro in micros:
        if micro < world or micro % world or int(trainer.config.actor_rollout_ref.actor.ppo_mini_batch_size)%micro:
            raise ValueError('Profile microbatch must divide global minibatch and world size')
    variants = {}
    for enabled, micro in [(False, micros[0])]+[(True,m) for m in micros]:
        data = copy.deepcopy(batch)
        options.sequence_balancing = enabled
        metrics = {}
        trainer._balance_batch(data, metrics)
        if fingerprints(data) != expected:
            raise ValueError('Balancing changed workload membership')
        data.meta_info['engineering_micro_batch_size'] = micro
        label = ('balanced' if enabled else 'unbalanced')+'_m'+str(micro)
        variants[label] = (data, metrics)
    records = []
    try:
        labels = list(variants)
        schedule = [('warmup', 0, label) for label in labels]
        for repeat in range(int(options.profile_repeats)):
            ordered = labels if repeat%2==0 else list(reversed(labels))
            schedule += [('measure', repeat, label) for label in ordered]
        for phase, repeat, label in schedule:
            data, metrics = variants[label]
            data = copy.deepcopy(data)
            data.meta_info['engineering_profile_phase'] = f'{phase}_{repeat}_{label}'
            data.meta_info['global_token_num'] = data.batch['attention_mask'].sum(-1).tolist()
            started = time.perf_counter()
            output = trainer.actor_rollout_wg.update_actor(data)
            elapsed = time.perf_counter()-started
            actor = reduce_metrics(output.meta_info['metrics'])
            if any(not math.isfinite(float(value)) for value in actor.values()):
                raise FloatingPointError('Nonfinite benchmark actor metric')
            row = dict(phase=phase, repeat=repeat, variant=label,
                       balanced=data.meta_info['engineering_balanced'],
                       micro_batch_size=data.meta_info['engineering_micro_batch_size'],
                       wall_seconds=elapsed, balance_metrics=metrics, actor_metrics=actor)
            records.append(row)
            trainer._append('engineering_measurements.jsonl', row)
        summary = dict(kind='fixed_workload_lr_zero', samples=len(batch),
            total_tokens=int(batch.batch['attention_mask'].sum()),
            workload_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
            initial_metrics=initial_metrics, variants={})
        for label in labels:
            rows = [r for r in records if r['phase']=='measure' and r['variant']==label]
            times = [r['wall_seconds'] for r in rows]
            summary['variants'][label] = dict(repeats=len(times),
                median_update_seconds=statistics.median(times), min_update_seconds=min(times),
                max_update_seconds=max(times), balance_metrics=variants[label][1],
                mean_rank_peak_allocated_gib=max(r['actor_metrics']['engineering/rank_mean_peak_allocated_gib'] for r in rows),
                grad_norms=[r['actor_metrics']['actor/grad_norm'] for r in rows])
        (trainer.audit_dir/'engineering_benchmark.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('ENGINEERING_BENCHMARK '+json.dumps(summary), flush=True)
    finally:
        options.sequence_balancing = original
