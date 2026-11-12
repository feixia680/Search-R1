"""Attach trusted HotpotQA references to extra_info; original prompts stay intact."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from search_r1.llm_agent.evidence import reference_from_raw, question_key


def attach_references(processed, raw):
    records, excluded = [], []
    for position, row in enumerate(processed.to_dict('records')):
        row = dict(row)
        extra = dict(row.get('extra_info', {}))
        index = int(extra.get('index', -1))
        question = row['prompt'][0]['content'].split('Question:', 1)[-1]
        if (index < 0 or index >= len(raw)
                or question_key(question) != question_key(raw.iloc[index]['question'])):
            evidence = dict(schema_version=1, raw_id='', status='mapping_error',
                            supporting_facts=[], errors=['question_index_mismatch'])
        else:
            evidence = reference_from_raw(raw.iloc[index])
        extra['evidence'] = evidence
        row['extra_info'] = extra
        records.append(row)
        if evidence['status'] != 'valid':
            excluded.append(dict(position=position, sample_index=index,
                                 raw_id=evidence['raw_id'], status=evidence['status'],
                                 errors=evidence['errors']))
    annotated = pd.DataFrame(records, columns=processed.columns)
    eligible = annotated[annotated.extra_info.map(lambda x: x['evidence']['status'] == 'valid')]
    return annotated, eligible, excluded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-root', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    root, out = Path(args.base_root), Path(args.out_dir)
    if out.resolve() == (root / 'data/processed/hotpotqa').resolve():
        raise ValueError('Output directory must not overwrite the baseline dataset')
    out.mkdir(parents=True, exist_ok=True)
    # Generated datasets and question text remain local.
    (out / '.gitignore').write_text('*\n')
    report = dict(schema_version=1, policy_prompts_changed=False, splits={})
    for split, raw_split in [('train', 'train'), ('test', 'validation')]:
        original = root / f'data/processed/hotpotqa/{split}.parquet'
        raw_path = root / f'data/raw/hotpotqa_{raw_split}.parquet'
        processed, raw = pd.read_parquet(original), pd.read_parquet(raw_path)
        annotated, eligible, errors = attach_references(processed, raw)
        annotated.to_parquet(out / f'{split}_all_audited.parquet', index=False)
        annotated.to_parquet(out / f'{split}.parquet', index=False)
        eligible.to_parquet(out / f'{split}_evidence_eligible.parquet', index=False)
        loaded = pd.read_parquet(out / f'{split}_all_audited.parquet')
        assert [r[0]['content'] for r in loaded.prompt] == [r[0]['content'] for r in processed.prompt]
        report['splits'][split] = dict(original_count=len(processed), eligible_count=len(eligible),
            evidence_ineligible_count=len(errors), evidence_ineligible=errors,
            primary_output_count=len(annotated),
            primary_output_preserves_original_sample_order=True,
            supporting_fact_count=sum(len(x['evidence']['supporting_facts']) for x in eligible.extra_info),
            processed_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
            raw_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256((out / f'{split}.parquet').read_bytes()).hexdigest())
    (out / 'annotation_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
