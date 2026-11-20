"""Replay annotated queries through the real rollout parser, HTTP service and visibility scorer on CPU."""
import argparse
import json
from collections import Counter
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from search_r1.llm_agent.evidence import score_evidence
from search_r1.llm_agent.generation import GenerationConfig, LLMGenerationManager


class ScriptedQueries:
    def __init__(self, tokenizer, queries):
        self.tokenizer = tokenizer
        self.actions = iter([f"<search>{q}</search>" for q in queries] + ["<answer>cpu fixture</answer>"])

    def generate_sequences(self, batch):
        assert len(batch) == 1 and not batch.non_tensor_batch
        ids = self.tokenizer(next(self.actions), add_special_tokens=False, return_tensors="pt").input_ids
        return DataProto.from_dict({"responses": ids}, meta_info={})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18081/retrieve")
    parser.add_argument("--tokenizer", default="models/Qwen2.5-3B")
    args = parser.parse_args()
    diag = args.root / "diagnostics"
    trace_rows = [json.loads(s) for s in (diag / "dev200_trace.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    dataset = RLHFDataset(str(args.root / "prepared/policy/dev200.parquet"), tok, max_prompt_length=512)
    assert len(trace_rows) == len(dataset) == 200
    # Both return formats, including the unscored public API, must return identical documents.
    queries = [r["question"] for r in trace_rows[:3]]
    expected = [[h["id"] for h in r["trace"][0]["hits"]] for r in trace_rows[:3]]
    for scored in (False, True):
        resp = requests.post(args.url, json=dict(queries=queries, topk=3, return_scores=scored), timeout=30)
        resp.raise_for_status()
        groups = resp.json()["result"]
        assert [[(h["document"] if scored else h)["id"] for h in hits] for hits in groups] == expected
    stats = Counter()
    out = diag / "dev200_cpu_visible.jsonl"
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as fp:
        for i, row in enumerate(trace_rows):
            item = dataset[i]
            ref = item["extra_info"]["evidence"]
            assert ref["raw_id"] == row["id"]
            prompt = dataset.dataframe.iloc[i]["prompt"][0]["content"]
            assert prompt.split("Question: ", 1)[1] == row["question"] + "\n"
            checks = []
            budgets = (1, 2, 4) if len(row["trace"]) == 4 and not stats["budget_paths_checked"] else (4,)
            for budget in budgets:
                planned = row["trace"][:budget]
                worker = ScriptedQueries(tok, [t["query"] for t in planned])
                cfg = GenerationConfig(max_turns=budget, max_start_length=512, max_prompt_length=5632,
                                       max_response_length=256, max_obs_length=1024, num_gpus=1, topk=3, search_url=args.url)
                manager = LLMGenerationManager(tok, worker, cfg, is_validation=True)
                batch = DataProto.from_single_dict(collate_fn([item]))
                gen = batch.pop(batch_keys=["input_ids","attention_mask","position_ids"])
                assert not gen.non_tensor_batch
                gen.meta_info.update(do_sample=False, validate=True, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
                result = manager.run_llm_loop(gen, gen.batch["input_ids"][:, -512:].clone())
                trace = result.non_tensor_batch["trajectory"][0]
                assert trace["final_answer"] == "cpu fixture" and trace["retrieval_failures"] == 0
                searches = [e for e in trace["events"] if e.get("attempted")]
                assert len(searches) == len(planned)
                for e, t in zip(searches, planned):
                    assert e["document_ids"] == [h["id"] for h in t["hits"]]
                scored = score_evidence(ref, trace)
                matched = {doc for e in scored["evidence_events"] for doc in e["matched_fact_ids"]}
                assert matched <= set(planned[-1]["coverage"])
                if budget == 4:
                    stats.update(questions=1, http_queries=len(searches),
                                 observation_truncated=sum(e["observation_truncated"] for e in searches),
                                 visible_complete=int(scored["evidence_coverage"]==1.0), visible_links=len(matched))
                    checks = [dict(query=e["query"], doc_ids=e["document_ids"], truncated=e["observation_truncated"],
                                   visible_support=ev["matched_fact_ids"]) for e,ev in zip(searches,scored["evidence_events"])]
            if len(budgets)==3:
                stats["budget_paths_checked"] += 1
            fp.write(json.dumps(dict(id=row["id"], actions="scripted_reference", events=checks),ensure_ascii=False)+"\n")
    assert stats["budget_paths_checked"] and stats["questions"]==200
    tmp.replace(out)
    (diag / "cpu_visible_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
