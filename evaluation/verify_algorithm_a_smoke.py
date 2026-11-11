"""Validate recorded reward accounting and saved parameter changes after smoke."""
import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, AutoConfig


def load_slice(directory, name):
    for file in Path(directory).glob('*.safetensors'):
        with safe_open(str(file), framework='pt', device='cpu') as reader:
            if name in reader.keys():
                return reader.get_slice(name)[:8, :128].float()
    raise ValueError(f'{name} not found in {directory}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--beta', type=float, default=.02)
    parser.add_argument('--gamma', type=float, default=.05)
    args = parser.parse_args()
    run = Path(args.run_dir)
    train = [json.loads(line) for line in (run / 'traces/train.jsonl').read_text().splitlines()]
    val = [json.loads(line) for line in (run / 'traces/validation.jsonl').read_text().splitlines()]
    assert len(train) == 4 and len(val) == 1, (len(train), len(val))
    assert {row['global_step'] for row in train} == {1}
    assert len({row['trajectory_id'] for row in train}) == 4
    for row in train:
        expected = row['em'] - args.beta * row['retrieval_attempts'] - args.gamma * row['duplicate_queries']
        assert math.isclose(row['total_reward'], expected, abs_tol=1e-6)
        assert row['retrieval_attempts'] == row['retrieval_successes'] + row['retrieval_failures']
        assert row['retrieval_attempts'] == sum(bool(e.get('attempted')) for e in row['events'])
    assert val[0]['total_reward'] == val[0]['em']  # Held-out scoring stays pure EM.
    checkpoint = run / 'checkpoints/actor/global_step_1'
    AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    name = 'model.layers.0.self_attn.q_proj.weight'
    base = load_slice(args.base_model, name)
    updated = load_slice(checkpoint, name)
    assert torch.isfinite(updated).all()
    delta = (updated - base).abs()
    assert delta.max().item() > 0, 'No saved parameter change detected'
    summary = dict(train_trajectories=len(train), validation_trajectories=len(val),
                   search_actions=sum(r['search_actions'] for r in train),
                   retrieval_attempts=sum(r['retrieval_attempts'] for r in train),
                   retrieval_successes=sum(r['retrieval_successes'] for r in train),
                   duplicate_queries=sum(r['duplicate_queries'] for r in train),
                   mean_em=sum(r['em'] for r in train)/len(train),
                   mean_total_reward=sum(r['total_reward'] for r in train)/len(train),
                   parameter_probe=name, max_parameter_delta=delta.max().item(),
                   changed_probe_elements=int((delta > 0).sum()), checkpoint=str(checkpoint))
    (run / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
