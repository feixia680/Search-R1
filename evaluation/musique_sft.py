"""One-GPU action SFT on verified train-only decisions. Observations are prompt tokens."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def step_loss(model,turn,device):
    prompt,target=turn["prompt"],turn["target"]
    if not prompt or not target or len(prompt)+len(target)>5888:
        raise ValueError("Invalid prompt/target lengths")
    ids=torch.tensor([prompt+target],device=device)
    labels=ids.clone()
    labels[:,:len(prompt)]=-100
    return model(input_ids=ids,labels=labels,use_cache=False).loss


def main():
    p=argparse.ArgumentParser()
    for key in ("model","collection","data","out"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--max-updates",type=int)
    args=p.parse_args()
    summary=json.loads((args.collection/"collection.json").read_text())
    assert summary["split"]=="train" and Path(summary["model"]).resolve()==args.model.resolve()
    train=args.data/"prepared/policy/train.parquet"
    assert summary["dataset_sha256"]==hashlib.sha256(train.read_bytes()).hexdigest()
    allowed={json.loads(line)["id"] for line in (args.data/"prepared/references/train.jsonl").open()}
    turns=[]
    seen=set()
    for line in (args.collection/"demonstrations.jsonl").open():
        r=json.loads(line)
        assert r["id"] in allowed and r["id"] not in seen
        seen.add(r["id"])
        turns.extend(r["turns"])
    assert turns and len(seen)==summary["stop"]+summary["continue"]
    random.Random(42).shuffle(turns)
    if args.max_updates is not None:
        assert args.max_updates>0
        turns=turns[:8*args.max_updates]
    torch.manual_seed(42)
    device=torch.device("cuda:0")
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.float32,
                                               attn_implementation="sdpa").to(device)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    optim=torch.optim.AdamW(model.parameters(),lr=2e-6,foreach=False)
    logs=[]
    probe=model.get_output_embeddings().weight
    probe_before=probe.detach()[:256].float().clone()
    for start in range(0,len(turns),8):
        group=turns[start:start+8]
        optim.zero_grad(set_to_none=True)
        group_loss=0.0
        for turn in group:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=step_loss(model,turn,device)/len(group)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite SFT loss")
            group_loss+=loss.detach().item()
            loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optim.step()
        logs.append(dict(step=len(logs)+1,loss=group_loss,grad_norm=float(norm)))
        if start%80==0: print("SFT turns",min(start+len(group),len(turns)),"/",len(turns),flush=True)
    temp=args.out.with_name(args.out.name+".partial")
    temp.mkdir(exist_ok=False)
    delta=(probe.detach()[:256]-probe_before).abs().max().item()
    if not any(x["grad_norm"]>0 for x in logs) or not delta>0:
        raise RuntimeError("SFT produced no measured parameter update")
    model.save_pretrained(temp)
    tokenizer.save_pretrained(temp)
    (temp/"musique_sft.json").write_text(json.dumps(dict(training_questions=len(seen),turns=len(turns),steps=logs,probe_max_abs_change=delta,start_model=str(args.model))))
    os.rename(temp,args.out)


if __name__=="__main__":
    main()
