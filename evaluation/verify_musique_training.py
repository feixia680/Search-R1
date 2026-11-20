"""Check a saved GRPO actor against its own initialization and training reward traces."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def weight_slice(folder,name):
    index=folder/"model.safetensors.index.json"
    shard=json.loads(index.read_text())["weight_map"][name] if index.exists() else "model.safetensors"
    with safe_open(folder/shard,framework="pt",device="cpu") as fp:
        return fp.get_slice(name)[:8].float()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("initial",type=Path)
    p.add_argument("step",type=int)
    args=p.parse_args()
    ckpt=args.run/"checkpoints/actor"/f"global_step_{args.step}"
    names=["model.layers.0.self_attn.q_proj.weight","model.layers.12.mlp.down_proj.weight","model.embed_tokens.weight"]
    changes={}
    for name in names:
        delta=(weight_slice(ckpt,name)-weight_slice(args.initial,name)).abs()
        if not torch.isfinite(delta).all():
            raise ValueError(f"Nonfinite checkpoint weight delta in {name}")
        changes[name]=float(delta.max())
    groups=defaultdict(list)
    with (args.run/"traces/train.jsonl").open() as fp:
        for line in fp:
            row=json.loads(line)
            assert row["split"]=="train" and math.isfinite(row["total_reward"])
            groups[(int(row["global_step"]),row["sample_id"])].append(row["total_reward"])
    stats={}
    for (step,_),values in groups.items():
        st=stats.setdefault(step,dict(groups=0,rollouts=0,groups_with_reward_spread=0))
        st["groups"]+=1
        st["rollouts"]+=len(values)
        st["groups_with_reward_spread"]+=int(max(values)-min(values)>1e-8)
    assert max(changes.values())>0 and args.step in stats
    result=dict(max_abs_weight_changes=changes,updates=stats)
    (args.run/"training_verification.json").write_text(json.dumps(result,indent=2))
    print("Checkpoint parameter change verified; reward spread by step:",stats,flush=True)


if __name__=="__main__":
    main()
