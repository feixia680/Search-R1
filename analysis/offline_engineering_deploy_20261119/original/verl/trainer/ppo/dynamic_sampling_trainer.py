"""Isolated evidence-aware bounded dynamic sampling. Legacy trainer is untouched."""
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, reduce_metrics, compute_data_metrics
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from search_r1.dynamic_sampling.selection import BoundedGroupSelector
from search_r1.dynamic_sampling.batching import merge_groups
from search_r1.dynamic_sampling.stratified import StratifiedQuestionSampler, load_manifest
from search_r1.dynamic_sampling.behavior import BehaviorMonitor, compact_trace, summarize_scopes, flatten


class DynamicSamplingTrainer(RayPPOTrainer):
    def __init__(self, *args, candidate_reward_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_reward_fn = candidate_reward_fn
        self.sampling = self.config.dynamic_sampling
        self.audit_dir = Path(self.sampling.audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._data_iterator = None
        self._candidate_epoch = 0
        self._stratified_sampler = None
        self._behavior_monitor = None
        self._behavior_roles = {}
        self._behavior_excluded_ids = set()
        self._pending_behavior_report = None
        behavior = self.config.get('behavior_sampling', {})
        monitor = self.config.get('behavior_monitor', {})
        if behavior.get('enabled', False) or monitor.get('enabled', False):
            files = self.config.data.train_files
            files = [files] if isinstance(files, str) else list(files)
            if len(files) != 1:
                raise ValueError('Offline behavior strata currently require one training parquet')
            path = behavior.get('manifest_path') or str(Path(files[0]).parent/'behavior_strata.json')
            sample_ids = [str(extra['index']) for extra in self.train_dataset.dataframe.extra_info]
            manifest = load_manifest(path, files[0], sample_ids)
            self._behavior_roles = manifest['roles']
            (self.audit_dir/'behavior_strata_snapshot.json').write_text(json.dumps(manifest, sort_keys=True)+'\n')
            if behavior.get('enabled', False):
                self._stratified_sampler = StratifiedQuestionSampler(
                    sample_ids, manifest['roles'], int(self.config.data.train_batch_size),
                    seed=int(self.config.trainer.get('seed', 42)),
                    uniform_fraction=float(behavior.get('uniform_fraction', .25)),
                    role_weights=dict(behavior.get('role_weights', {})))
            if monitor.get('enabled', False):
                self._behavior_monitor = BehaviorMonitor(self.audit_dir,
                    **{key: value for key, value in monitor.items() if key != 'enabled'})

    def _next_questions(self):
        sampler = getattr(self, '_stratified_sampler', None)
        if sampler is not None:
            from verl.utils.dataset.rl_dataset import collate_fn
            positions, self._last_strata_draw = sampler.sample(self._behavior_excluded_ids)
            items = [self.train_dataset[position] for position in positions]
            for item in items:
                item['behavior_stratum'] = self._behavior_roles[str(item['index'])]
            self._candidate_epoch = sampler.total_drawn // len(sampler.sample_ids)
            return collate_fn(items)
        if self._data_iterator is None:
            self._data_iterator = iter(self.train_dataloader)
        try:
            result = next(self._data_iterator)
        except StopIteration:
            self._candidate_epoch += 1
            self._data_iterator = iter(self.train_dataloader)
            result = next(self._data_iterator)
        if getattr(self, '_behavior_roles', None):
            result['behavior_stratum'] = np.array([self._behavior_roles[str(sid)] for sid in result['index']], dtype=object)
        return result

    def _append(self, name, record):
        with (self.audit_dir / name).open('a') as out:
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + '\n')

    def _candidate(self, manager, step, round_id):
        batch = DataProto.from_single_dict(self._next_questions())
        size = int(self.config.actor_rollout_ref.rollout.n_agent)
        batch = batch.repeat(repeat_times=size, interleave=True)
        gen = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
        first = gen.batch['input_ids'][:, -self.config.data.max_start_length:].clone().long()
        manager.timing_raw = {}
        started = time.perf_counter()
        output = manager.run_llm_loop(gen_batch=gen, initial_input_ids=first)
        elapsed = time.perf_counter() - started
        for key in output.batch.keys():
            output.batch[key] = output.batch[key].long()
        batch.non_tensor_batch['uid'] = np.array([
            f'{step}:{round_id}:{idx}' for idx in batch.non_tensor_batch['index']], dtype=object)
        batch.non_tensor_batch['sampling_round'] = np.array([round_id] * len(batch), dtype=object)
        batch = batch.union(output)
        batch.meta_info['global_step'] = step
        rewards = self.candidate_reward_fn(batch)
        if not torch.isfinite(rewards).all():
            raise FloatingPointError('Candidate token-level rewards are nonfinite')
        batch.batch['sampling_reward'] = rewards.sum(dim=-1)
        if getattr(self, '_behavior_monitor', None) is not None:
            records = getattr(self.candidate_reward_fn, 'last_records', None)
            if records is None or len(records) != len(batch):
                raise ValueError('Behavior diagnostics require aligned scored reward records')
            roles = batch.non_tensor_batch.get('behavior_stratum', ['unobserved'] * len(batch))
            compact = []
            for i, record in enumerate(records):
                if str(record.get('sample_id')) != str(batch.non_tensor_batch['index'][i]):
                    raise ValueError('Behavior scored record sample ID is misaligned')
                if not math.isclose(float(record['total_reward']), float(batch.batch['sampling_reward'][i]), abs_tol=1e-7, rel_tol=1e-7):
                    raise ValueError('Behavior scored record reward is misaligned')
                compact.append(compact_trace(record, batch.non_tensor_batch['uid'][i], roles[i]))
            batch.non_tensor_batch['behavior_record'] = np.array(compact, dtype=object)
        return batch, elapsed

    def _collect(self, manager, step):
        selector = BoundedGroupSelector(
            target_groups=int(self.config.data.train_batch_size),
            group_size=int(self.config.actor_rollout_ref.rollout.n_agent),
            max_batches=int(self.sampling.max_candidate_batches),
            epsilon=float(self.sampling.reward_range_epsilon),
            enabled=bool(self.sampling.enabled), seed=int(self.config.trainer.get('seed', 42)) + step)
        batches = []
        self._behavior_excluded_ids = set()
        draws = []
        generation_seconds = 0.
        generated_tokens = observation_tokens = response_tokens = retrieval_calls = failures = missing = 0
        for round_id in range(selector.max_batches):
            batch, elapsed = self._candidate(manager, step, round_id)
            generation_seconds += elapsed
            batches.append(batch)
            self._behavior_excluded_ids.update(map(str, batch.non_tensor_batch['index']))
            if getattr(self, '_stratified_sampler', None) is not None:
                draws.append(self._last_strata_draw)
            selector.offer(round_id, batch.non_tensor_batch['index'].tolist(),
                           batch.batch['sampling_reward'].tolist())
            width = batch.batch['prompts'].shape[1]
            response_tokens += int(batch.batch['attention_mask'][:, width:].sum())
            generated_tokens += int(batch.batch['info_mask'][:, width:].sum())
            for trace in batch.non_tensor_batch['trajectory']:
                retrieval_calls += trace['retrieval_attempts']
                failures += trace['retrieval_failures']
                missing += int(not str(trace.get('final_answer') or '').strip())
                observation_tokens += sum(e.get('observation_tokens_after_truncation', 0)
                                          for e in trace['events'] if e.get('attempted'))
            if selector.ready:
                break
        selected, stats = selector.finish()
        stats.update(candidate_generated_tokens=generated_tokens,
                     candidate_observation_tokens=observation_tokens,
                     candidate_response_tokens=response_tokens,
                     candidate_retrieval_calls=retrieval_calls,
                     candidate_retrieval_failures=failures,
                     candidate_missing_answer_fraction=missing / stats['candidate_trajectories'],
                     candidate_generation_seconds=generation_seconds,
                     candidate_epoch=self._candidate_epoch,
                     candidate_unique_questions=len(selector.seen))
        chosen = {(g.batch_id, g.sample_id) for g in selected}
        self._append('sampling_decisions.jsonl', dict(global_step=step, **stats, groups=[dict(
            round=g.batch_id, sample_id=g.sample_id, rewards=g.rewards,
            informative=g.reward_range > selector.epsilon,
            selected=(g.batch_id, g.sample_id) in chosen) for g in selector.offered]))
        merged = merge_groups(batches, selected, self.tokenizer.pad_token_id)
        counts = Counter(merged.non_tensor_batch['uid'])
        if len(counts) != selector.target or set(counts.values()) != {selector.group_size}:
            raise ValueError('GRPO group identity was lost during selection/merge')
        metrics = {'sampling/'+key: value for key, value in stats.items()}
        if getattr(self, '_behavior_monitor', None) is not None:
            def records(data):
                if 'behavior_record' not in data.non_tensor_batch:
                    raise ValueError('Missing scored behavior records before selection monitoring')
                return data.non_tensor_batch['behavior_record'].tolist()
            report = summarize_scopes([row for data in batches for row in records(data)], records(merged))
            report['stratified_draws'] = draws
            metrics.update(flatten(report))
            self._pending_behavior_report = report
        return merged, metrics

    def _fit_sampling(self):
        if self.config.trainer.get('val_only', False):
            self.logger.log(self._validate(), step=0)
            return
        if self.config.trainer.get('val_before_train', False):
            self.logger.log(self._validate(), step=0)
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns, max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url, topk=self.config.retriever.topk,
            search_timeout=self.config.retriever.get('timeout', 30.))
        manager = LLMGenerationManager(self.tokenizer, self.actor_rollout_wg, gen_config)
        # Preserve this repository's historical stopping convention: total=625
        # means updates 1..624, followed by validation labeled 625.
        for step in range(1, int(self.total_training_steps)):
            self.global_steps = step
            started = time.perf_counter()
            batch, metrics = self._collect(manager, step)
            self._balance_batch(batch, metrics=metrics)
            batch.meta_info['global_token_num'] = batch.batch['attention_mask'].sum(-1).tolist()
            tick = time.perf_counter()
            old = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old)
            metrics['timing_s/selected_old_log_prob'] = time.perf_counter()-tick
            if self.use_reference_policy:
                tick = time.perf_counter()
                batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))
                metrics['timing_s/ref'] = time.perf_counter()-tick
            batch.meta_info['global_step'] = step
            scores = self.reward_fn(batch)
            # Recompute only selected outcomes after padding; equality proves
            # padding and metadata reordering did not change selection rewards.
            if not torch.allclose(scores.sum(-1), batch.batch['sampling_reward'], atol=1e-7, rtol=1e-7):
                raise ValueError('Selected reward changed after merge or batch balancing')
            batch.batch['token_level_scores'] = scores
            batch.batch['token_level_rewards'] = scores
            metrics.update(self.reward_fn.last_metrics)
            batch = compute_advantage(batch, adv_estimator='grpo', gamma=self.config.algorithm.gamma,
                                      lam=self.config.algorithm.lam,
                                      num_repeat=self.config.actor_rollout_ref.rollout.n)
            for key in ('old_log_probs', 'token_level_scores', 'advantages'):
                if not torch.isfinite(batch.batch[key]).all():
                    raise FloatingPointError('Selected tensor is nonfinite: '+key)
            if 'ref_log_prob' in batch.batch and not torch.isfinite(batch.batch['ref_log_prob']).all():
                raise FloatingPointError('Reference log probabilities are nonfinite')
            batch, metrics = self._create_loss_mask(batch, metrics)
            tick = time.perf_counter()
            output = self.actor_rollout_wg.update_actor(batch)
            actor_metrics = reduce_metrics(output.meta_info['metrics'])
            if any(not math.isfinite(float(v)) for v in actor_metrics.values()):
                raise FloatingPointError('Nonfinite actor metric returned')
            metrics.update(actor_metrics)
            metrics['timing_s/update_actor'] = time.perf_counter()-tick
            behavior_alerts = []
            monitor = getattr(self, '_behavior_monitor', None)
            if monitor is not None and self._pending_behavior_report is not None:
                health, behavior_alerts = monitor.record(step, self._pending_behavior_report)
                metrics.update(health)
            if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and step % self.config.trainer.test_freq == 0:
                metrics.update(self._validate())
            if self.config.trainer.save_freq > 0 and step % self.config.trainer.save_freq == 0:
                self._save_checkpoint()
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
            metrics['timing_s/step'] = time.perf_counter()-started
            self.logger.log(metrics, step=step)
            self._append('completed_updates.jsonl', dict(global_step=step, metrics=metrics))
            if behavior_alerts and monitor.stop_on_alert:
                raise RuntimeError('Behavior monitor stopped after completed update: '+','.join(behavior_alerts))
        self.global_steps = int(self.total_training_steps)
        if self.val_reward_fn is not None and not self.config.trainer.get('skip_final_validation', False):
            self.logger.log(self._validate(), step=self.global_steps)

    def fit(self):
        try:
            self._fit_sampling()
        except Exception as error:
            # Detailed rank-local numerical failures are written by the new
            # worker. Driver report identifies the failed update and audit files.
            (self.audit_dir / 'failure.json').write_text(json.dumps(dict(
                global_step=getattr(self, 'global_steps', None), error=repr(error)), indent=2)+'\n')
            raise
