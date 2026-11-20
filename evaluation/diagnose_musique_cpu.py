"""Audit BM25 and annotated queries. No policy generation or model outcomes."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher
from search_r1.llm_agent.evidence import contains_sentence


def read_rows(path):
    with path.open() as fp:
        for line in fp:
            yield json.loads(line)


def resolve_query(step, steps):
    def replace(match):
        pos = int(match.group(1))
        return steps[pos - 1]["answer"] if pos in step["depends"] else match.group(0)
    return " ".join(re.sub(r"#(\d+)", replace, step["template"]).replace(">>", " ").split())


def diagnose(row, lookup, contents):
    support = set(row["support"])
    first = lookup(row["question"])
    first_docs = {x["id"] for x in first}
    steps = row["steps"]
    references = []
    for step in steps:
        query = resolve_query(step, steps)
        hits = lookup(query)
        references.append(dict(query=query, hits=hits, target=step["doc"], target_hit=any(h["id"] == step["doc"] for h in hits)))
    retrieved = set(first_docs)
    attempted = set()
    trace = [dict(query=row["question"], hits=first, coverage=sorted(support & retrieved))]
    while len(trace) < 4 and not support <= retrieved:
        eligible = []
        for j, s in enumerate(steps):
            if j in attempted or s["doc"] in retrieved:
                continue
            # Gold decomposition chooses the subgoal. Bridge answers must appear
            # in their earlier retrieved gold passage before substitution.
            if all(steps[k-1]["doc"] in retrieved and contains_sentence(contents[steps[k-1]["doc"]], steps[k-1]["answer"])
                   for k in s["depends"]):
                eligible.append(j)
        if not eligible:
            break
        j = eligible[0]
        attempted.add(j)
        ref = references[j]
        prev = set(retrieved)
        retrieved.update(h["id"] for h in ref["hits"])
        trace.append(dict(step=j+1, query=ref["query"], hits=ref["hits"], coverage=sorted(support & retrieved),
                          new_support=sorted(support & (retrieved-prev))))
    union = {h["id"] for r in references for h in r["hits"]}
    return dict(id=row["id"], question=row["question"], hops=row["hops"], support=row["support"], trace=trace,
                annotated_queries=references, annotated_union=sorted(union & support),
                first_found=len(first_docs & support), final_found=len(retrieved & support))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    prep = args.root / "prepared"
    out = args.root / "diagnostics"
    out.mkdir(exist_ok=False)
    searcher = LuceneSearcher(str(args.root / "index"))
    content = {r["id"]:r["contents"] for r in read_rows(prep / "collection/passages.jsonl")}
    assert searcher.num_docs == len(content)
    # Check actual Lucene stored contents on every document including distractors.
    for key, expected in content.items():
        doc = searcher.doc(key)
        assert doc is not None and json.loads(doc.raw()) == {"id": key, "contents": expected}
    @lru_cache(maxsize=4096)
    def lookup(query):
        return [dict(id=h.docid, score=float(h.score)) for h in searcher.search(query, 3)]
    all_stats = {}
    for name in ("dev200", "train"):
        grouped = {}
        with (out / (name + "_first.csv")).open("w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(["id", "hops", "support_count", "retrieved_support_count", "retrieved_ids"])
            for r in read_rows(prep / "references" / (name+".jsonl")):
                hits = [h["id"] for h in lookup(r["question"])]
                count = len(set(r["support"]) & set(hits))
                w.writerow([r["id"], r["hops"], len(r["support"]), count, " ".join(hits)])
                for key in ("all", str(r["hops"])):
                    st = grouped.setdefault(key,Counter())
                    st.update(rows=1, first_complete=int(count==len(r["support"])), first_zero=int(count==0),
                              first_hit_links=count, support_links=len(r["support"]))
        all_stats[name] = grouped
        print(name, grouped["all"], flush=True)
    with (out / "dev200_trace.jsonl").open("w") as traces, (out / "dev200_diagnosis.csv").open("w", newline="") as table:
        w = csv.writer(table)
        w.writerow(["id","question","hops","required_passages","first_found","after_2_found","after_3_found",
                    "after_4_found","reference_calls_incl_first","all_annotated_queries_union_found","annotated_queries_hitting_target"])
        for r in read_rows(prep / "references/dev200.jsonl"):
            d = diagnose(r, lookup, content)
            traces.write(json.dumps(d,ensure_ascii=False)+"\n")
            t = d["trace"]
            curves = [len(t[min(b,len(t))-1]["coverage"]) for b in (1,2,3,4)]
            n = len(r["support"])
            refs = d["annotated_queries"]
            w.writerow([r["id"],r["question"],r["hops"],n,*curves,len(t),len(d["annotated_union"]),sum(x["target_hit"] for x in refs)])
            for key in ("all",str(r["hops"])):
                st = all_stats["dev200"][key]
                st.update({f"budget_{b}_complete":int(x==n) for b,x in enumerate(curves,1)})
                st.update({f"budget_{b}_hit_links":x for b,x in enumerate(curves,1)})
                st.update(annotated_union_complete=int(len(d["annotated_union"])==n),
                          annotated_target_hits=sum(x["target_hit"] for x in refs), annotated_query_count=len(refs))
    report = dict(index_documents_verified=len(content), topk=3, results=all_stats)
    (out / "summary.json").write_text(json.dumps(report,indent=2))
    searcher.close()


if __name__ == "__main__":
    main()
