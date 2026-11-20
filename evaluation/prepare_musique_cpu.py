"""Prepare official MuSiQue-Ans train/dev and a fixed shared paragraph retrieval corpus."""
import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def question_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).rstrip(" ?")


def dump_line(fp, value):
    fp.write(json.dumps(value, ensure_ascii=False) + "\n")


def dependencies(template, step_index):
    # A literal rank such as "#44" is not a decomposition variable.
    return sorted({int(n) for n in re.findall(r"#(\d+)", template) if 1 <= int(n) <= step_index})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("baseline", type=Path, help="Existing evidence parquet for schema and prompt prefix")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in ("collection", "references", "policy"):
        (out / name).mkdir()
    template = pq.ParquetFile(args.baseline)
    schema = template.schema_arrow.remove_metadata()
    prefix = next(template.iter_batches(batch_size=1)).to_pylist()[0]["prompt"][0]["content"].split("Question: ", 1)[0] + "Question: "
    if "<search>" not in prefix:
        raise ValueError("Unexpected baseline prompt")
    collection = {}
    samples = {}
    keys = {}
    step_ids = {}
    source_hashes = {}
    for split in ("train", "dev"):
        path = args.raw / f"musique_ans_v1.0_{split}.jsonl.gz"
        source_hash = hashlib.sha256()
        samples[split] = []
        keys[split] = set()
        step_ids[split] = set()
        ids = set()
        with gzip.open(path, "rb") as fp:
            for line in fp:
                source_hash.update(line)
                r = json.loads(line)
                assert r["answerable"] and r["id"] not in ids and r["answer"].strip()
                ids.add(r["id"])
                keys[split].add(question_key(r["question"]))
                docs = {}
                marked = set()
                for p in r["paragraphs"]:
                    assert p["idx"] not in docs and p["title"].strip() and p["paragraph_text"].strip()
                    content = '"' + p["title"] + '"\n' + p["paragraph_text"]
                    doc_id = digest(content)
                    assert collection.setdefault(doc_id, content) == content
                    docs[p["idx"]] = doc_id
                    if p["is_supporting"]:
                        marked.add(doc_id)
                steps = []
                for j, s in enumerate(r["question_decomposition"]):
                    step_ids[split].add(s["id"])
                    steps.append(dict(template=s["question"], answer=s["answer"], doc=docs[s["paragraph_support_idx"]],
                                      depends=dependencies(s["question"], j)))
                assert marked == {s["doc"] for s in steps}
                assert len(steps) in (2, 3, 4)
                samples[split].append(dict(id=r["id"], question=r["question"], answers=list(dict.fromkeys(
                    [r["answer"], *r["answer_aliases"]])), hops=len(steps), steps=steps, support=sorted(marked)))
        source_hashes[split] = source_hash.hexdigest()
    overlap = {k: len(v["train"] & v["dev"]) for k, v in (("question", keys), ("subquestion_id", step_ids))}
    assert not any(overlap.values()) and not ({r["id"] for r in samples["train"]} & {r["id"] for r in samples["dev"]})
    with (out / "collection/passages.jsonl").open("w") as fp:
        for doc_id, content in sorted(collection.items()):
            dump_line(fp, {"id": doc_id, "contents": content})
    groups = defaultdict(list)
    for r in samples["dev"]:
        groups[r["hops"]].append(r)
    selected = set()
    selected_questions = set()
    for hops, target in ((2, 67), (3, 67), (4, 66)):
        group = sorted(groups[hops], key=lambda r: digest("musique-cpu-v1-42:" + r["id"]))
        for r in group:
            key = question_key(r["question"])
            if key not in selected_questions:
                selected.add(r["id"])
                selected_questions.add(key)
                target -= 1
                if not target:
                    break
        assert target == 0
    holdout = [r for r in samples["dev"] if question_key(r["question"]) not in selected_questions]
    partitions = {"train": samples["train"], "dev200": [r for r in samples["dev"] if r["id"] in selected],
                  "dev_holdout": holdout}
    audit = {}
    for name, rows in partitions.items():
        output = out / "policy" / (name + ".parquet")
        with (out / "references" / (name + ".jsonl")).open("w") as refs, pq.ParquetWriter(output, schema) as writer:
            batch = []
            for i, row in enumerate(rows):
                dump_line(refs, row)
                facts = []
                for doc in row["support"]:
                    title, body = collection[doc].split("\n", 1)
                    facts.append(dict(fact_id=doc, title=title[1:-1], sentence_id=-1, sentence=body))
                batch.append(dict(prompt=[dict(role="user", content=prefix + row["question"] + "\n")], data_source="musique",
                                  ability="fact-reasoning", reward_model=dict(style="rule", ground_truth=dict(target=row["answers"])),
                                  extra_info=dict(index=i, split=name, evidence=dict(schema_version=1, raw_id=row["id"],
                                      status="valid", errors=[], supporting_facts=facts))))
                if len(batch) >= 256 or i + 1 == len(rows):
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    batch.clear()
        audit[name] = dict(rows=len(rows), hop_counts=dict(Counter(r["hops"] for r in rows)),
                           support_links=sum(len(r["support"]) for r in rows))
    report = dict(source="StonyBrookNLP/musique official MuSiQue-Ans v1.0 (CC BY 4.0)",
                  source_url="https://github.com/StonyBrookNLP/musique", raw_jsonl_sha256=source_hashes,
                  source_rows={k:len(v) for k,v in samples.items()}, corpus_paragraphs=len(collection),
                  cross_split_overlap=overlap, partitions=audit,
                  dev_duplicates_excluded=len(samples["dev"])-200-len(holdout))
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
