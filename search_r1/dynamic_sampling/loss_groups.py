"""Preserve legacy loss groups while batching their model forwards together."""
import torch


def grouped_masked_mean(values, mask, group_size):
    if group_size < 1 or values.shape[0] % group_size:
        raise ValueError('Loss groups must partition the microbatch')
    numerator = (values*mask).reshape(-1, group_size*values.shape[-1]).sum(-1)
    denominator = mask.reshape(-1, group_size*mask.shape[-1]).sum(-1)
    return (numerator/denominator).mean()


def grouped_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, group_size):
    difference = log_prob-old_log_prob
    ratio = difference.exp()
    raw = -advantages*ratio
    clipped = -advantages*ratio.clamp(1.-cliprange, 1.+cliprange)
    return (grouped_masked_mean(torch.maximum(raw, clipped),eos_mask,group_size),
            grouped_masked_mean((clipped>raw).float(),eos_mask,group_size),
            grouped_masked_mean(-difference,eos_mask,group_size))
