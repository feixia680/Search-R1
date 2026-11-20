#!/usr/bin/env python3
"""Check token-loss gradients through the actor update and distributed scaling."""
import argparse
import json
import os
import sys

# Ray exposes one physical GPU as cuda:0 to each actor worker. Mirror that
# layout in the torchrun fixture, including TensorDict.cuda()'s device default.
if '--fsdp' in sys.argv:
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices[int(os.environ['LOCAL_RANK'])]

import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo import core_algos


def objective(log_prob, old, advantages, mask, reference, entropy, normalizer):
    ratio = (log_prob - old).exp()
    clipped = ratio.clamp(0.8, 1.2)
    policy = torch.maximum(-advantages * ratio, -advantages * clipped)
    kl = core_algos.kl_penalty(log_prob, reference, 'low_var_kl')
    return ((policy - 0.02 * entropy + 0.03 * kl) * mask).sum() * normalizer


def check_distributed():
    torch.distributed.init_process_group('gloo')
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    # Unequal token counts, including an empty microbatch on rank zero.
    mask = torch.tensor([[0., 0., 0.], [1., 0., 0.]]) if rank == 0 else torch.ones(2, 3)
    normalizer = core_algos.token_loss_normalizer(mask)
    parameter = torch.tensor(0.1, requires_grad=True)
    for micro_mask in mask.split(1):
        log_prob = parameter.expand_as(micro_mask)
        loss, _, _ = core_algos.compute_policy_loss(
            torch.zeros_like(log_prob), log_prob, torch.full_like(log_prob, rank + 1.),
            micro_mask, 0.2, loss_normalizer=normalizer)
        loss.backward()
    torch.distributed.all_reduce(parameter.grad)
    parameter.grad /= world
    expected = -(1. + 6. * 2.) / 7. * torch.exp(torch.tensor(0.1))
    torch.testing.assert_close(parameter.grad, expected)
    if rank == 0:
        print(json.dumps({'distributed_ranks': world, 'unequal_rank_tokens': 'passed',
                          'gradient': parameter.grad.item()}))
    torch.distributed.destroy_process_group()


def check_actor():
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    class TinyActor(DataParallelPPOActor):
        def _forward_micro_batch(self, micro_batch, temperature):
            rows = micro_batch['input_ids'][:, 0]
            log_prob = self.actor_module.weight[rows]
            return 1. + log_prob.square(), log_prob

    device = 'cuda'
    initial = torch.linspace(-0.35, 0.35, 24, device=device).reshape(4, 6)
    mask = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], device=device)
    advantages = torch.tensor([1., -1., 0.5, -0.5], device=device)[:, None].expand(4, 6)
    old = torch.zeros_like(initial)
    ref = torch.full_like(initial, 0.03)
    tensors = {
        'responses': torch.ones(4, 6, dtype=torch.long, device=device),
        'input_ids': torch.arange(4, device=device)[:, None].expand(4, 7),
        'position_ids': torch.arange(7, device=device)[None, :].expand(4, 7),
        'attention_mask': torch.ones(4, 7, dtype=torch.long, device=device),
        'loss_mask': mask, 'old_log_probs': old,
        'advantages': advantages, 'ref_log_prob': ref,
    }
    data = DataProto.from_dict(tensors, meta_info={'temperature': 1.})
    cases = []
    for mode in ['legacy-microbatch-mean', 'token-mean']:
        for micro_size in [1, 2, 4]:
            config = OmegaConf.create(dict(
                use_remove_padding=False, ulysses_sequence_parallel_size=1,
                loss_agg_mode=mode, ppo_mini_batch_size=4, ppo_micro_batch_size=micro_size,
                use_dynamic_bsz=False, state_masking=True, use_kl_loss=True,
                clip_ratio=0.2, entropy_coeff=0.02, kl_loss_coef=0.03,
                kl_loss_type='low_var_kl', grad_clip=100.))
            module = torch.nn.Linear(6, 4, bias=False, device=device)
            module.weight.data.copy_(initial)
            actor = TinyActor(config, module, torch.optim.SGD(module.parameters(), lr=0.1))
            expected_parameter = initial.clone().requires_grad_()
            if mode == 'token-mean':
                expected_loss = objective(expected_parameter, old, advantages, mask, ref,
                                          1. + expected_parameter.square(), 1. / mask.sum())
            else:
                expected_loss = 0.
                for start in range(0, 4, micro_size):
                    part = slice(start, start + micro_size)
                    expected_loss = expected_loss + objective(
                        expected_parameter[part], old[part], advantages[part], mask[part], ref[part],
                        1. + expected_parameter[part].square(), 1. / mask[part].sum()) / (4 // micro_size)
            expected_loss.backward()
            expected_weight = initial - 0.1 * expected_parameter.grad
            metrics = actor.update_policy(data)
            torch.testing.assert_close(module.weight, expected_weight, atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(module.weight[mask == 0], initial[mask == 0])
            if mode == 'token-mean':
                torch.testing.assert_close(
                    torch.tensor(metrics['actor/pg_loss'][0]),
                    core_algos.compute_policy_loss(old, initial, advantages, mask, 0.2)[0].cpu())
            cases.append({'mode': mode, 'microbatch': micro_size, 'passed': True})
    # Explicit zero-token microbatch and rejection of an empty optimizer batch.
    empty = torch.zeros(2, 6, device=device)
    values = initial[:2].clone().requires_grad_()
    loss, _, _ = core_algos.compute_policy_loss(old[:2], values, advantages[:2], empty, 0.2,
                                               loss_normalizer=1. / mask.sum())
    loss.backward()
    assert loss.item() == 0. and values.grad.abs().sum().item() == 0.
    try:
        core_algos.token_loss_normalizer(empty)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty optimizer minibatch was accepted')
    print(json.dumps({'actor_update_cases': cases, 'observation_mask': 'passed',
                      'empty_masks': 'passed'}))


def check_fsdp():
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl')
    assert torch.distributed.get_world_size() == 2
    device = torch.device('cuda', 0)

    class TinyFSDPActor(DataParallelPPOActor):
        def _forward_micro_batch(self, micro_batch, temperature):
            log_prob = self.actor_module(micro_batch['input_ids'][:, :6].float())
            return 1. + log_prob.square(), log_prob

    initial = torch.linspace(-0.3, 0.3, 36, device=device).reshape(6, 6)
    features = torch.eye(6, device=device)[:4]
    mask = torch.tensor([[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]], device=device)
    advantages = torch.tensor([1., -1., 0.5, -0.5], device=device)[:, None].expand(4, 6)
    old = torch.zeros(4, 6, device=device)
    reference = torch.full_like(old, 0.03)
    expected = initial.clone().requires_grad_()
    log_prob = features @ expected.T
    loss = objective(log_prob, old, advantages, mask, reference,
                     1. + log_prob.square(), 1. / mask.sum())
    loss.backward()
    expected_weight = initial - 0.1 * expected.grad

    model = torch.nn.Linear(6, 6, bias=False, device=device)
    model.weight.data.copy_(initial)
    model = FSDP(model, device_id=device)
    config = OmegaConf.create(dict(
        use_remove_padding=False, ulysses_sequence_parallel_size=1,
        loss_agg_mode='token-mean', ppo_mini_batch_size=2, ppo_micro_batch_size=1,
        use_dynamic_bsz=False, state_masking=True, use_kl_loss=True,
        clip_ratio=0.2, entropy_coeff=0.02, kl_loss_coef=0.03,
        kl_loss_type='low_var_kl', grad_clip=100.))
    actor = TinyFSDPActor(config, model, torch.optim.SGD(model.parameters(), lr=0.1))
    part = slice(rank * 2, rank * 2 + 2)
    tensors = {
        'responses': torch.ones(2, 6, dtype=torch.long, device=device),
        'input_ids': torch.cat([features[part], torch.zeros(2, 1, device=device)], dim=1),
        'attention_mask': torch.ones(2, 7, device=device),
        'position_ids': torch.arange(7, device=device)[None, :].expand(2, 7),
        'loss_mask': mask[part], 'old_log_probs': old[part],
        'advantages': advantages[part], 'ref_log_prob': reference[part],
    }
    actor.update_policy(DataProto.from_dict(tensors, meta_info={'temperature': 1.}))
    with FSDP.summon_full_params(model):
        torch.testing.assert_close(model.weight, expected_weight, atol=1e-7, rtol=1e-6)
        max_error = float((model.weight - expected_weight).abs().max())
    print(json.dumps({'rank': rank, 'fsdp_unequal_token_counts': 'passed',
                      'max_parameter_error': max_error}))
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--fsdp', action='store_true')
    args = parser.parse_args()
    if args.fsdp:
        check_fsdp()
    elif args.distributed:
        check_distributed()
    else:
        check_actor()
