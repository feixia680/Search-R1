"""Autonomous budget evaluation and matched search/answer branch collection for MuSiQue."""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from search_r1.llm_agent.evidence import score_evidence
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.reward_score.qa_em import em_check


class Worker:
    """Greedy evaluation, with optional shared first action and second-turn intervention for train data."""
    def __init__(self, model, tokenizer, batch_size):
        from vllm import LLM, SamplingParams
        self.tokenizer = tokenizer
        self.params = SamplingParams
        self.llm = LLM(model=str(model), tokenizer=str(model), dtype="bfloat16", tensor_parallel_size=1,
                       max_model_len=5888, max_num_seqs=batch_size, enforce_eager=True,
                       gpu_memory_utilization=0.75, seed=42)
        self.reset()

    def reset(self, action=None, first=None, seed=42):
        self.action, self.first, self.seed = action, first, seed
        self.history = defaultdict(list)
        self.initial_order = []
        self.turn = 0

    def generate_sequences(self, batch):
        assert not batch.non_tensor_batch
        prompts = [r[m.bool()].long().tolist() for r, m in zip(batch.batch["input_ids"], batch.batch["attention_mask"])]
        prefix = []
        at_cap = self.turn==getattr(self,"answer_at_turn",None)
        if at_cap:
            prefix=self.tokenizer.encode("<answer>",add_special_tokens=False)
        elif self.turn == 1 and self.action:
            prefix = self.tokenizer.encode("<" + self.action + ">", add_special_tokens=False)
        if self.action:
            if self.turn == 0:
                self.initial_order = [tuple(p) for p in prompts]
                assert len(set(self.initial_order)) == len(prompts), "Duplicate prompts in collection batch"
            keys = [next(k for k in self.initial_order if tuple(p[:len(k)])==k) for p in prompts]
        if self.turn == 0 and self.first is not None:
            outputs = [self.first[key] for key in keys]
        else:
            sample = self.turn == 1 and self.action == "search"
            params = self.params(temperature=0.8 if sample else 0, top_p=0.95 if sample else 1.0,
                                 seed=self.seed, max_tokens=256-len(prefix), detokenize=at_cap,
                                 **(dict(stop=["</answer>","</search>"],include_stop_str_in_output=True) if at_cap else {}))
            result = self.llm.generate(prompt_token_ids=[p+prefix for p in prompts], sampling_params=params, use_tqdm=False)
            outputs = [prefix + list(x.outputs[0].token_ids) for x in result]
        if self.action:
            for key,prompt,output in zip(keys,prompts,outputs):
                self.history[key].append((prompt,output))
        self.turn += 1
        ids = torch.full((len(outputs), max(1,max(map(len,outputs)))), self.tokenizer.pad_token_id, dtype=torch.long)
        for i, seq in enumerate(outputs):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        return DataProto.from_dict({"responses":ids})


def rollout(worker, tokenizer, batch, budget, url):
    gen = batch.pop(batch_keys=["input_ids","attention_mask","position_ids"])
    assert not gen.non_tensor_batch
    gen.meta_info.update(do_sample=False, validate=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    cfg = GenerationConfig(max_turns=budget, max_start_length=512, max_prompt_length=5632, max_response_length=256,
                           max_obs_length=1024, num_gpus=1, topk=3, search_url=url)
    result = LLMGenerationManager(tokenizer, worker, cfg, is_validation=True).run_llm_loop(gen, gen.batch["input_ids"])
    records = []
    for i, trace in enumerate(result.non_tensor_batch["trajectory"]):
        if trace["retrieval_failures"] or trace["retrieval_attempts"] > budget:
            raise RuntimeError("Retrieval failure or budget violation invalidates rollout")
        ref = batch.non_tensor_batch["extra_info"][i]["evidence"]
        answers = batch.non_tensor_batch["reward_model"][i]["ground_truth"]["target"]
        evidence = score_evidence(ref,trace)
        gains = evidence["evidence_events"]
        actions = [{k:e[k] for k in ("action","generated_text","query","document_ids","attempted","observation_truncated") if k in e}
                   for e in trace["events"]]
        for event in gains:
            actions[event["turn"]]["new_visible_support"] = event["new_fact_ids"]
        records.append(dict(id=ref["raw_id"], answer=trace["final_answer"], em=int(bool(trace["final_answer"]) and em_check(trace["final_answer"],answers)),
                            coverage=evidence["evidence_coverage"], first_coverage=gains[0]["cumulative_coverage"] if gains else 0,
                            calls=trace["retrieval_attempts"], termination=trace["termination_reason"], actions=actions))
    return records


def summarize(rows):
    n = len(rows)
    return dict(n=n, em=sum(r["em"] for r in rows)/n, mean_calls=sum(r["calls"] for r in rows)/n,
                evidence_complete=sum(r["coverage"]==1 for r in rows)/n, mean_coverage=sum(r["coverage"] for r in rows)/n,
                no_answer=sum(not r["answer"] for r in rows)/n,
                exhausted=sum(r["termination"]=="search_budget_exhausted" for r in rows)/n)


def evaluate(args, ds, tok, worker):
    prior = {}
    summary = {}
    for budget in args.budgets:
        worker.answer_at_turn=budget if args.answer_at_cap else None
        started=time.monotonic()
        rows = []
        dest = args.out / f"budget_{budget}.jsonl"
        with dest.with_suffix(".partial").open("w") as fp:
            for raw in DataLoader(ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate_fn):
                worker.reset()
                records = rollout(worker,tok,DataProto.from_single_dict(raw),budget,args.url)
                for r in records:
                    fp.write(json.dumps(r,ensure_ascii=False)+"\n")
                rows.extend(records)
                fp.flush()
                print(f"budget={budget} completed={len(rows)}/{len(ds)} elapsed={time.monotonic()-started:.1f}s",flush=True)
        assert len(rows)==len(ds) and len({r["id"] for r in rows})==len(rows)
        dest.with_suffix(".partial").replace(dest)
        stats = summarize(rows)
        stats["by_hops"] = {h:summarize(group) for h in ("2","3","4") if (group:=[r for r in rows if r["id"].startswith(h+"hop")])}
        if prior:
            stats["newly_correct_vs_previous_budget"] = sum(r["em"] and not prior[r["id"]] for r in rows)
            stats["newly_incorrect_vs_previous_budget"] = sum(not r["em"] and prior[r["id"]] for r in rows)
        prior = {r["id"]:r["em"] for r in rows}
        summary[budget] = stats
        print(budget,stats,flush=True)
    (args.out/"summary.json").write_text(json.dumps(dict(model=str(args.model),split=args.split,budgets=summary,answer_at_cap=args.answer_at_cap),indent=2))


def train_indices(ds, root, count):
    groups = defaultdict(list)
    with (root/"diagnostics/train_first.csv").open() as fp:
        for i, row in enumerate(csv.DictReader(fp)):
            extra = ds.dataframe.iloc[i]["extra_info"]
            assert extra["split"]=="train" and row["id"]==extra["evidence"]["raw_id"]
            hits,total=int(row["retrieved_support_count"]),int(row["support_count"])
            state = "full" if hits==total else "partial" if hits else "empty"
            groups[(state,int(row["hops"]))].append(i)
    assert sum(map(len,groups.values()))==len(ds)
    rng = random.Random(42)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result)<min(len(ds),count):
        for k in sorted(groups):
            if groups[k] and len(result)<count:
                result.append(groups[k].pop())
    return result


def accepted(branch, immediate=None):
    if not branch["em"] or any(e["action"]=="invalid" for e in branch["actions"]):
        return False
    if len(branch["actions"])<2 or branch["actions"][0]["action"]!="search":
        return False
    if immediate is None:
        return branch["actions"][1]["action"]=="answer" and branch["first_coverage"]==1
    return (branch["actions"][1]["action"]=="search" and branch["calls"]>=2 and
            not immediate["em"] and branch["coverage"]>branch["first_coverage"])


def collect(args, ds, tok, worker):
    path = args.out/"demonstrations.jsonl"
    picked = train_indices(ds,args.data,args.count)
    stats = defaultdict(int)

    def run(indices, action, first=None, seed=42):
        worker.reset(action, first, seed)
        batch=DataProto.from_single_dict(collate_fn([ds[i] for i in indices]))
        records=rollout(worker,tok,batch,4,args.url)
        return {i:(r,worker.history[key]) for i,r,key in zip(indices,records,worker.initial_order)}

    with path.with_suffix(".partial").open("w") as demos, (args.out/"branch_audit.jsonl").open("w") as audit:
        for start in range(0,len(picked),args.batch_size):
            indices=picked[start:start+args.batch_size]
            direct=run(indices,"answer")
            trials={i:[] for i in indices}
            winners={i:(*direct[i],"stop") for i in indices if accepted(direct[i][0])}
            pending=[i for i in indices if not direct[i][0]["em"] and
                     [e["action"] for e in direct[i][0]["actions"]]==["search","answer"]]
            for attempt in range(2):
                if not pending:
                    break
                first={tuple(direct[i][1][0][0]):direct[i][1][0][1] for i in pending}
                result=run(pending,"search",first,seed=42+2*start+attempt)
                for i,(branch,history) in result.items():
                    assert history[1][0]==direct[i][1][1][0], "Branches must share exact decision-state tokens"
                    trials[i].append(branch)
                    if accepted(branch,direct[i][0]):
                        winners[i]=(branch,history,"continue")
                pending=[i for i in pending if i not in winners]
            for i in indices:
                kind=None
                if i in winners:
                    branch,history,kind=winners[i]
                    assert len(history)==len(branch["actions"])
                    examples=[dict(prompt=prompt,target=tok.encode(event["generated_text"],add_special_tokens=False))
                              for (prompt,_),event in zip(history,branch["actions"])]
                    assert all(t["target"] for t in examples)
                    demos.write(json.dumps(dict(id=branch["id"],kind=kind,turns=examples))+"\n")
                    stats[kind]+=1
                audit.write(json.dumps(dict(id=direct[i][0]["id"],direct=direct[i][0],
                                            continuations=trials[i],selected=kind),ensure_ascii=False)+"\n")
                stats["attempted"]+=1
            audit.flush()
            demos.flush()
            print(dict(stats),flush=True)
        if stats["continue"] and stats["stop"]<max(8,stats["continue"]//2):
            used={ds.dataframe.iloc[i]["extra_info"]["evidence"]["raw_id"] for i in picked}
            add_stop_examples(args,ds,tok,worker,stats,used,demos,audit)
    if not (stats["stop"] and stats["continue"]):
        raise RuntimeError(f"No verified stop/continue pair: {dict(stats)}. Inspect branch_audit.jsonl.")
    path.with_suffix(".partial").replace(path)
    (args.out/"collection.json").write_text(json.dumps(dict(stats,model=str(args.model),split="train",
        dataset_sha256=hashlib.sha256((args.data/"prepared/policy/train.parquet").read_bytes()).hexdigest()),indent=2))


def add_stop_examples(args, ds, tok, worker, stats, used_ids, demos, audit):
    """Mine train questions whose reference first query can return all support.
    Prefer the policy's query. If scarce, query with the user question verbatim
    and supervise only the subsequent model answer, never the fixed search."""
    with (args.data/"diagnostics/train_first.csv").open() as fp:
        pool=[i for i,r in enumerate(csv.DictReader(fp)) if r["id"] not in used_ids and
              r["support_count"]==r["retrieved_support_count"]]
    random.Random(7361).shuffle(pool)
    desired=max(8,stats["continue"]//2)
    for s in range(0,min(len(pool),192),args.batch_size):
        if stats["stop"]>=desired:
            break
        indices=pool[s:s+args.batch_size]
        items=[ds[i] for i in indices]
        todo=list(range(len(items)))
        for question_first in (False,True):
            if not todo:
                break
            selected=[items[j] for j in todo]
            first=None
            if question_first:
                first={}
                for j in todo:
                    r=items[j]
                    ids=r["input_ids"][r["attention_mask"].bool()].tolist()
                    q=ds.dataframe.iloc[indices[j]]["prompt"][0]["content"].split("Question: ",1)[1].strip()
                    first[tuple(ids)]=tok.encode("<search>"+q+"</search>",add_special_tokens=False)
            worker.reset("answer",first)
            batch=DataProto.from_single_dict(collate_fn(selected))
            records=rollout(worker,tok,batch,4,args.url)
            failed=[]
            for j,rec,key in zip(todo,records,worker.initial_order):
                ok=accepted(rec)
                source="user_question_first" if question_first else "policy_first"
                audit.write(json.dumps(dict(id=rec["id"],direct=rec,continuations=[],source=source,
                                            selected="stop" if ok else None),ensure_ascii=False)+"\n")
                if not ok:
                    failed.append(j)
                    continue
                hist=worker.history[key]
                assert len(hist)==len(rec["actions"])==2
                begin=1 if question_first else 0
                turns=[dict(prompt=hist[k][0],target=tok.encode(rec["actions"][k]["generated_text"],add_special_tokens=False))
                       for k in range(begin,len(hist))]
                demos.write(json.dumps(dict(id=rec["id"],kind="stop",source=source,turns=turns))+"\n")
                stats["stop"]+=1
                stats["supplement_"+source]+=1
            todo=failed
        stats["attempted"]+=len(items)
        audit.flush()
        demos.flush()
        print(dict(stats),flush=True)


def supplement(args,ds,tok,worker):
    import shutil
    if args.source is None:
        raise ValueError("Supplement requires an existing train-branch directory")
    source=args.source
    path=source/"demonstrations.jsonl"
    if not path.exists():
        path=source/"demonstrations.partial"
    audit_path=source/"branch_audit.jsonl"
    train_ids={e["evidence"]["raw_id"] for e in ds.dataframe.extra_info}
    base=[json.loads(l) for l in path.open()]
    used={json.loads(l)["id"] for l in audit_path.open()}
    assert used<=train_ids and {r["id"] for r in base}<=used
    assert len({r["id"] for r in base})==len(base)
    stats=defaultdict(int,attempted=len(used),stop=sum(r["kind"]=="stop" for r in base),
                      **{"continue":sum(r["kind"]=="continue" for r in base)})
    tmp=args.out/"demonstrations.partial"
    shutil.copyfile(path,tmp)
    shutil.copyfile(audit_path,args.out/"branch_audit.jsonl")
    with tmp.open("a") as demos,(args.out/"branch_audit.jsonl").open("a") as audit:
        add_stop_examples(args,ds,tok,worker,stats,used,demos,audit)
    assert stats["stop"] and stats["continue"]
    tmp.replace(args.out/"demonstrations.jsonl")
    (args.out/"collection.json").write_text(json.dumps(dict(stats,model=str(args.model),split="train",
        dataset_sha256=hashlib.sha256((args.data/"prepared/policy/train.parquet").read_bytes()).hexdigest()),indent=2))



def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["evaluate","collect","supplement"])
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--url",required=True)
    p.add_argument("--source",type=Path)
    p.add_argument("--split",choices=["train","dev200","dev_holdout"],required=True)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--count",type=int,default=512)
    p.add_argument("--budgets", type=int, nargs="+", choices=(1,2,3,4), default=[1,2,4])
    p.add_argument("--limit", type=int)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--answer-at-cap",action="store_true",help="Separate control: prefix the last available generation with <answer>")
    p.add_argument("--warmup-four", action="store_true")
    args=p.parse_args()
    if (args.mode!="evaluate") != (args.split=="train") or min(args.batch_size,args.count)<1:
        p.error("Collection requires train and evaluation requires a dev split; sizes must be positive")
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    ds=RLHFDataset(str(args.data/"prepared/policy"/(args.split+".parquet")),tok,max_prompt_length=512,truncation="error")
    if args.mode=="evaluate" and args.limit is not None:
        if not 0 <= args.offset < args.limit <= len(ds):
            p.error("limit out of range")
        groups=defaultdict(list)
        for i, extra in enumerate(ds.dataframe.extra_info):
            qid=extra["evidence"]["raw_id"]
            groups[qid.split("hop")[0]].append((hashlib.sha256(("eval42"+qid).encode()).hexdigest(),i))
        ordered=[]
        from itertools import zip_longest
        for stripe in zip_longest(*(sorted(groups[h]) for h in sorted(groups))):
            ordered.extend(x[1] for x in stripe if x is not None)
        ds.dataframe=ds.dataframe.iloc[ordered[args.offset:args.limit]].reset_index(drop=True)
    args.out.mkdir(parents=True,exist_ok=False)
    worker=Worker(args.model,tok,args.batch_size)
    if args.warmup_four and args.mode=="evaluate":
        batch=collate_fn([ds[i] for i in range(min(4,len(ds)))])
        t=time.monotonic()
        warm=rollout(worker,tok,DataProto.from_single_dict(batch),4,args.url)
        (args.out/"warmup_four.json").write_text(json.dumps(dict(seconds=time.monotonic()-t,rows=warm),ensure_ascii=False))
    {"collect":collect,"supplement":supplement,"evaluate":evaluate}[args.mode](args,ds,tok,worker)


if __name__=="__main__":
    main()
