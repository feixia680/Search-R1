"""Is an extra search needed, or could the actor already answer at budget two?"""
import argparse
import csv
import json
from pathlib import Path


def load(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    result={r["id"]:r for r in rows}
    assert len(rows)==len(result)
    return result


def previous_actions(row):
    return [(e["action"],e["generated_text"],e.get("document_ids")) for e in row["actions"][:2]]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("ordinary",type=Path)
    p.add_argument("control",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    a=load(args.ordinary/"budget_2.jsonl")
    b=load(args.ordinary/"budget_4.jsonl")
    c=load(args.control/"budget_2.jsonl")
    assert a.keys()==b.keys()==c.keys()
    matched=[i for i in a if previous_actions(a[i])==previous_actions(b[i])==previous_actions(c[i])]
    blocked=[i for i in matched if a[i]["termination"]=="search_budget_exhausted"]
    gained=[i for i in matched if not a[i]["em"] and b[i]["em"]]
    later_only=[i for i in gained if not c[i]["em"]]
    extra_support=[i for i in later_only if any(e.get("new_visible_support") for e in
                  [e for e in b[i]["actions"] if e.get("attempted")][2:])]
    result=dict(questions=len(a),exact_prefix_matched=len(matched),
                autonomous_2_correct=sum(r["em"] for r in a.values()),
                autonomous_4_correct=sum(r["em"] for r in b.values()),
                forced_at_cap_2_correct=sum(r["em"] for r in c.values()),
                matched_blocked_at_2=len(blocked),matched_blocked_now_answer=sum(bool(c[i]["answer"]) for i in blocked),
                matched_blocked_now_correct=sum(c[i]["em"] for i in blocked),
                matched_gains_of_4=gained,also_correct_if_answer_at_2=[i for i in gained if c[i]["em"]],
                correct_only_at_4_among_these_runs=later_only,also_retrieved_new_support_beyond_2=extra_support)
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"answer_control.json").write_text(json.dumps(result,indent=2))
    with (args.out/"answer_control.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(["id","matched_prefix","autonomous_2_em","answer_at_2_em","autonomous_4_em","termination_at_2","calls_at_4"])
        for i in a:
            w.writerow([i,i in matched,a[i]["em"],c[i]["em"],b[i]["em"],a[i]["termination"],b[i]["calls"]])
    print({k:len(v) if isinstance(v,list) else v for k,v in result.items()})


if __name__=="__main__":
    main()
