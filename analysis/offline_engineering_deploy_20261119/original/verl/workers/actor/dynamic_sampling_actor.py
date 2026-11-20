"""Fixed-microbatch actor update with an explicit collective loss check.

The policy objective and accumulation match this checkout's dp_actor.py. This
class is instantiated only by the separate dynamic-sampling worker.
"""
import torch
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.dynamic_sampling_worker import collective_check
import os


class GuardedDataParallelPPOActor(DataParallelPPOActor):
    def update_policy(self, data):
        self.actor_module.train()
        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.state_masking:
            keys.append('loss_mask')
        if self.config.use_kl_loss:
            keys.append('ref_log_prob')
        batch = data.select(batch_keys=keys).batch
        metrics = {}
        for mini in batch.split(self.config.ppo_mini_batch_size):
            self.actor_optimizer.zero_grad()
            for micro in mini.split(self.config.ppo_micro_batch_size):
                micro = micro.cuda()
                width = micro['responses'].shape[1]
                mask = micro['loss_mask'] if self.config.state_masking else micro['attention_mask'][:, -width:]
                entropy, log_prob = self._forward_micro_batch(micro_batch=micro, temperature=temperature)
                pg_loss, clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=micro['old_log_probs'], log_prob=log_prob,
                    advantages=micro['advantages'], eos_mask=mask, cliprange=self.config.clip_ratio)
                entropy_loss = masked_mean(entropy, mask)
                loss = pg_loss - entropy_loss * self.config.entropy_coeff
                if self.config.use_kl_loss:
                    kl = core_algos.kl_penalty(logprob=log_prob, ref_logprob=micro['ref_log_prob'],
                                              kl_penalty=self.config.kl_loss_type)
                    kl_loss = masked_mean(kl, mask)
                    loss = loss + kl_loss * self.config.kl_loss_coef
                    # Preserve legacy last-microbatch KL reporting.
                    metrics['actor/kl_loss'] = kl_loss.detach().item()
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef
                loss = loss / self.gradient_accumulation
                collective_check(bool(torch.isfinite(loss)), 'before_backward',
                    os.environ.get('DYNAMIC_DIAGNOSTIC_DIR'),
                    getattr(self, '_dynamic_update_context', {}))
                loss.backward()
                append_to_dict(metrics, {'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(), 'actor/pg_clipfrac': clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item()})
            norm = self._optimizer_step()
            append_to_dict(metrics, {'actor/grad_norm': norm.detach().item()})
        self.actor_optimizer.zero_grad()
        return metrics
