"""Measure where each gold subquestion's target passage ranks in the frozen index."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    diag = args.root / "diagnostics"
    searcher = LuceneSearcher(str(args.root / "index"))
    counts = {}
    with (diag / "dev200_trace.jsonl").open() as rows, (diag / "dev200_reference_ranks.csv").open("w", newline="") as fp:
        table = csv.writer(fp)
        table.writerow(["id","hops","step","annotated_query","target_doc","rank_in_top20"])
        for line in rows:
            row = json.loads(line)
            for n, ref in enumerate(row["annotated_queries"], 1):
                hits = [h.docid for h in searcher.search(ref["query"],20)]
                rank = hits.index(ref["target"])+1 if ref["target"] in hits else None
                table.writerow([row["id"],row["hops"],n,ref["query"],ref["target"],rank])
                for group in ("all", str(row["hops"])):
                    st = counts.setdefault(group, Counter())
                    st.update(queries=1)
                    for k in (3,10,20):
                        st[f"target_at_{k}"] += int(rank is not None and rank<=k)
    searcher.close()
    (diag / "reference_ranks.json").write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
