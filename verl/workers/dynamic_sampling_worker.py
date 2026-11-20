"""Numerical fail-stop guards scoped to the new trainer's actor instances."""
import json
import os
from pathlib import Path
import types
import time

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.single_controller.base.decorator import register, Dispatch


def collective_check(ok, phase, diagnostic_dir=None, details=None):
    distributed = dist.is_available() and dist.is_initialized()
    device = torch.device('cuda', torch.cuda.current_device()) if distributed and dist.get_backend() == 'nccl' else torch.device('cpu')
    flag = torch.tensor([0 if ok else 1], dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        rank = dist.get_rank() if distributed else 0
        if diagnostic_dir:
            path = Path(diagnostic_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'rank_{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(rank=rank, phase=phase, local_failure=not bool(ok),
                    details=details or {}), default=str)+'\n')
        raise FloatingPointError('Collective numerical guard stopped actor: '+phase)


def attach_numerical_guards(actor, diagnostic_dir=None):
    if getattr(actor, '_dynamic_guards_installed', False):
        return
    forward = actor._forward_micro_batch

    def guarded_forward(self, micro_batch, temperature):
        bad = [key for key in ('old_log_probs', 'ref_log_prob', 'advantages')
               if key in micro_batch and not torch.isfinite(micro_batch[key]).all()]
        group = int(getattr(self.config, 'engineering_loss_group_size', 0))
        if group and 'loss_mask' in micro_batch:
            mask = micro_batch['loss_mask']
            if mask.shape[0] % group or (mask.reshape(-1, group*mask.shape[-1]).sum(-1)<=0).any():
                bad.append('empty_loss_group')
        if 'loss_mask' in micro_batch and micro_batch['loss_mask'].sum().item() <= 0:
            bad.append('empty_loss_mask')
        details = dict(getattr(self, '_dynamic_update_context', {}), bad_fields=bad,
                       shapes={k: list(v.shape) for k, v in micro_batch.items() if torch.is_tensor(v)})
        collective_check(not bad, 'before_forward', diagnostic_dir, details)
        entropy, log_prob = forward(micro_batch=micro_batch, temperature=temperature)
        collective_check(bool(torch.isfinite(entropy).all() and torch.isfinite(log_prob).all()),
                         'forward_outputs', diagnostic_dir, details)
        return entropy, log_prob

    def guarded_step(self):
        if isinstance(self.actor_module, FSDP):
            norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        collective_check(bool(torch.isfinite(norm)), 'before_optimizer_step', diagnostic_dir,
                         getattr(self, '_dynamic_update_context', {}))
        self.actor_optimizer.step()
        return norm

    actor._forward_micro_batch = types.MethodType(guarded_forward, actor)
    actor._optimizer_step = types.MethodType(guarded_step, actor)
    actor._dynamic_guards_installed = True


class GuardedActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            from verl.workers.actor.dynamic_sampling_actor import GuardedDataParallelPPOActor
            self.actor = GuardedDataParallelPPOActor(config=self.config.actor,
                actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer)
            attach_numerical_guards(self.actor, os.environ.get('DYNAMIC_DIAGNOSTIC_DIR'))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_engineering_replay(self):
        # Real generation frees KV before actor updates; reproduce that state
        # when replay bypasses generation, so memory measurements are comparable.
        if self._is_actor and self.config.rollout.free_cache_engine:
            self.rollout.inference_engine.free_cache_engine()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data):
        requested_micro = data.meta_info.get('engineering_micro_batch_size')
        if requested_micro is not None:
            world = dist.get_world_size() if dist.is_initialized() else 1
            if requested_micro % world or requested_micro//world < 1:
                raise ValueError('Profile microbatch must divide across ranks')
            self.actor.config.ppo_micro_batch_size = requested_micro//world
        self.actor._dynamic_update_context = dict(
            global_step=data.meta_info.get('global_step'),
            sample_ids=[str(x) for x in data.non_tensor_batch.get('index', [])])
        if not self.config.actor.get('engineering_profile', False):
            return super().update_actor(data)
        tokens = int(data.batch['attention_mask'].sum().item())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        output = super().update_actor(data)
        torch.cuda.synchronize()
        record = dict(rank=dist.get_rank() if dist.is_initialized() else 0,
            global_step=data.meta_info.get('global_step'),
            phase=data.meta_info.get('engineering_profile_phase', 'train'),
            balanced=data.meta_info.get('engineering_balanced'), tokens=tokens,
            global_micro_batch_size=requested_micro,
            loss_group_size=self.actor.config.get('engineering_loss_group_size', 0),
            seconds=time.perf_counter()-started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        directory = os.environ.get('DYNAMIC_DIAGNOSTIC_DIR')
        if directory:
            path = Path(directory).parent/'engineering_profiles'
            path.mkdir(parents=True, exist_ok=True)
            with (path/f"rank_{record['rank']}.jsonl").open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        for key in ('seconds', 'peak_allocated_gib', 'peak_reserved_gib', 'tokens'):
            output.meta_info['metrics']['engineering/rank_mean_'+key] = [record[key]]
        return output
