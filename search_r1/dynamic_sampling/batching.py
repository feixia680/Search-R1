"""Join groups generated with different prompt/response widths without re-tokenizing."""
import numpy as np
import torch
import torch.nn.functional as F
from verl import DataProto


ROW_META = ('turns_stats', 'active_mask', 'valid_action_stats', 'valid_search_stats')


def merge_groups(batches, groups, pad_token_id):
    if not groups:
        raise ValueError('Cannot merge an empty selection')
    pmax = max(batches[g.batch_id].batch['prompts'].shape[1] for g in groups)
    rmax = max(batches[g.batch_id].batch['responses'].shape[1] for g in groups)
    keys = set(batches[groups[0].batch_id].batch.keys())
    nonkeys = set(batches[groups[0].batch_id].non_tensor_batch)
    tensors, metadata = {k: [] for k in keys}, {k: [] for k in nonkeys}
    meta = dict(batches[groups[0].batch_id].meta_info)
    for key in ROW_META:
        if key in meta:
            meta[key] = []
    for group in groups:
        batch = batches[group.batch_id]
        if set(batch.batch.keys()) != keys or set(batch.non_tensor_batch) != nonkeys:
            raise ValueError('Candidate tensor/metadata schemas disagree')
        indices = list(group.row_indices)
        p = batch.batch['prompts'].shape[1]
        r = batch.batch['responses'].shape[1]
        for key, tensor in batch.batch.items():
            value = tensor[indices]
            if key == 'prompts':
                value = F.pad(value, (pmax-p, 0), value=pad_token_id)
            elif key in ('responses', 'responses_with_info_mask'):
                value = F.pad(value, (0, rmax-r), value=pad_token_id)
            elif key == 'input_ids':
                value = torch.cat((F.pad(value[:, :p], (pmax-p, 0), value=pad_token_id),
                                   F.pad(value[:, p:], (0, rmax-r), value=pad_token_id)), dim=1)
            elif key in ('attention_mask', 'info_mask', 'position_ids'):
                if value.shape[1] != p+r:
                    raise ValueError('Combined input field lost prompt/response alignment')
                value = torch.cat((F.pad(value[:, :p], (pmax-p, 0), value=0),
                                   F.pad(value[:, p:], (0, rmax-r), value=0)), dim=1)
            elif value.ndim != 1:
                raise ValueError('Unexpected pre-update tensor field: '+key)
            tensors[key].append(value)
        for key, values in batch.non_tensor_batch.items():
            metadata[key].append(values[indices])
        for key in ROW_META:
            if key in meta:
                source = batch.meta_info[key]
                if len(source) != len(batch):
                    raise ValueError('Per-row generation metadata is misaligned')
                meta[key].extend(source[i] for i in indices)
    result = DataProto.from_dict({k: torch.cat(v, dim=0) for k, v in tensors.items()},
        non_tensors={k: np.concatenate(v, axis=0) for k, v in metadata.items()}, meta_info=meta)
    assert result.batch['input_ids'].shape[1] == pmax+rmax
    return result
