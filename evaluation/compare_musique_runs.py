"""Paired EM, evidence, failure categories and matched-cost curves on identical question IDs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    indexed = {r["id"]:r for r in rows}
    assert indexed and len(rows)==len(indexed)
    return indexed


def reason(row):
    if row["em"]:
        return "correct"
    if row["coverage"]==1:
        return "wrong_despite_full_visible_support"
    if row["termination"]=="search_budget_exhausted":
        return "attempted_search_beyond_budget"
    if row["calls"]<=1 and row["termination"]=="answer":
        return "answered_early_with_incomplete_support"
    if row["calls"]>1 and row["coverage"]==row["first_coverage"]:
        return "later_search_added_no_support"
    if row["calls"]>1 and row["coverage"]>row["first_coverage"]:
        return "added_support_but_chain_incomplete"
    return "other_unresolved"


def cost_point(points, cost):
    # Randomly mix adjacent budget policies; this is an estimate, not a new run.
    points=sorted(points)
    for x,y in points:
        if abs(x-cost)<1e-10:
            return y
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        if x1<cost<x2:
            return y1+(cost-x1)*(y2-y1)/(x2-x1)
    return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run",nargs=2,action="append",required=True,metavar=("LABEL","DIRECTORY"))
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    groups={}
    for label,folder in args.run:
        groups[label]={int(f.stem.split("_")[1]):read(f) for f in sorted(Path(folder).glob("budget_*.jsonl"))}
        if not groups[label]:
            raise ValueError(f"No completed budgets for {label}")
    baseline=next(iter(groups))
    ref=groups[baseline][max(groups[baseline])]
    insufficient={k for k,r in ref.items() if r["calls"] and r["first_coverage"]<1}
    sufficient={k for k,r in ref.items() if r["calls"] and r["first_coverage"]==1}
    report={"n":len(ref),"fixed_insufficient_n":len(insufficient),"fixed_sufficient_n":len(sufficient),"models":{}}
    curves={}
    cases={}
    for label,runs in groups.items():
        summary={}
        curves[label]=[]
        for budget,rows in sorted(runs.items()):
            assert rows.keys()==ref.keys(),"Paired comparison requires identical IDs"
            n=len(rows)
            em=sum(r["em"] for r in rows.values())/n
            cost=sum(r["calls"] for r in rows.values())/n
            curves[label].append((cost,em))
            summary[budget]=dict(em=em,mean_calls=cost,correct=sum(r["em"] for r in rows.values()),
                complete=sum(r["coverage"]==1 for r in rows.values()),mean_coverage=sum(r["coverage"] for r in rows.values())/n,
                error_categories=dict(Counter(reason(r) for r in rows.values())),
                fixed_insufficient_correct=sum(rows[k]["em"] for k in insufficient),
                fixed_sufficient_calls=sum(rows[k]["calls"] for k in sufficient),
                second_search=sum(r["calls"]>=2 for r in rows.values()))
            cases[f"{label}_{budget}"]={k:reason(r) for k,r in rows.items()}
        if 2 in runs and 4 in runs:
            a,b=runs[2],runs[4]
            gains=[k for k in a if b[k]["em"]>a[k]["em"]]
            losses=[k for k in a if b[k]["em"]<a[k]["em"]]
            rescue=[k for k in gains if any(e.get("new_visible_support") for e in
                [x for x in b[k]["actions"] if x.get("attempted")][2:])]
            summary["paired_2_to_4"]={"gains":gains,"losses":losses,"gains_with_new_support_after_second":rescue}
        report["models"][label]=summary
    low=max(min(x for x,_ in pts) for pts in curves.values())
    high=min(max(x for x,_ in pts) for pts in curves.values())
    report["matched_cost_mixture_estimate"]=None
    if low<=high:
        cost=(low+high)/2
        report["matched_cost_mixture_estimate"]={"mean_calls":cost,"em":{k:cost_point(v,cost) for k,v in curves.items()}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"comparison.json").write_text(json.dumps(report,indent=2))
    with (args.out/"metrics.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(["model","budget","n","correct","em","mean_calls","support_complete","mean_coverage",
                    "fixed_insufficient_n","fixed_insufficient_correct","fixed_sufficient_n","fixed_sufficient_mean_calls"])
        for label,stats in report["models"].items():
            for budget,item in stats.items():
                if not isinstance(budget,int):
                    continue
                w.writerow([label,budget,len(ref),item["correct"],item["em"],item["mean_calls"],item["complete"],
                    item["mean_coverage"],len(insufficient),item["fixed_insufficient_correct"],len(sufficient),
                    item["fixed_sufficient_calls"]/len(sufficient) if sufficient else ""])
    with (args.out/"paired_questions.csv").open("w",newline="") as fp:
        w=csv.writer(fp)
        w.writerow(["model","budget","id","hops","em","answer","calls","first_visible_coverage","final_visible_coverage",
                    "termination","observed_outcome","later_queries_with_new_visible_support"])
        for label,runs in groups.items():
            for budget,rows in sorted(runs.items()):
                for qid,r in rows.items():
                    later=[e.get("query") for e in [e for e in r["actions"] if e.get("attempted")][1:]
                           if e.get("new_visible_support")]
                    w.writerow([label,budget,qid,qid.split("hop")[0],r["em"],r["answer"],r["calls"],
                                r["first_coverage"],r["coverage"],r["termination"],reason(r),json.dumps(later,ensure_ascii=False)])
    (args.out/"failure_labels.json").write_text(json.dumps(cases,indent=2))
    print("Paired results:",args.out/"comparison.json",flush=True)


if __name__=="__main__":
    main()
