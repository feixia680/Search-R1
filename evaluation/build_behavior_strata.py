"""Build a frozen offline training-strata manifest from training traces only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from search_r1.dynamic_sampling.stratified import file_sha256


def build(dataset, traces, output):
    import pandas as pd
    data = pd.read_parquet(dataset)
    refs = {}
    for extra in data.extra_info:
        sid = str(extra['index'])
        if sid in refs: raise ValueError('Duplicate training sample ID')
        if extra.get('split') not in ('train', 'training'): raise ValueError('Strata require training data only')
        refs[sid] = extra.get('evidence', {}).get('raw_id')
    stats = defaultdict(lambda: dict(rows=0, observed=0, coverage_sum=0., zero_first=0, duplicates=0, second_new=0))
    count = 0
    with Path(traces).open() as stream:
        for line in stream:
            r = json.loads(line)
            if r.get('split') != 'train': raise ValueError('Reference traces must contain training split only')
            sid = str(r['sample_id'])
            if sid not in refs or r.get('evidence_reference_id') != refs[sid]:
                raise ValueError('Reference trace IDs do not match training annotations')
            count += 1; st = stats[sid]; st['rows'] += 1
            st['duplicates'] += int(r.get('duplicate_queries', 0) > 0)
            calls = [e for e in r.get('events', []) if e.get('attempted')]
            if not r.get('evidence_eligible') or not calls: continue
            first = next((e for e in r.get('evidence_events', []) if e.get('turn') == calls[0]['turn']), None)
            if first is None: continue
            coverage = float(first.get('cumulative_coverage', 0))
            if not 0 <= coverage <= 1: raise ValueError('Invalid first-search coverage')
            st['observed'] += 1; st['coverage_sum'] += coverage
            st['zero_first'] += int(coverage == 0)
            st['second_new'] += int(any(e.get('evidence_gain', 0) > 0 for e in r.get('evidence_events', [])
                                         if e.get('turn', -1) > calls[0]['turn']))
    if not count: raise ValueError('No reference training traces')
    roles = {}
    for sid in refs:
        st = stats[sid]
        if not st['observed']: role = 'unobserved'
        elif st['coverage_sum']/st['observed'] >= 1. - 1e-8: role = 'first_sufficient'
        elif st['second_new']: role = 'first_insufficient'
        elif st['zero_first']/st['observed'] >= .5 or st['duplicates']/st['rows'] >= .5: role = 'query_rewrite'
        else: role = 'first_insufficient'
        roles[sid] = role
    result = dict(schema_version=1, dataset_sha256=file_sha256(dataset),
                  reference_traces_sha256=file_sha256(traces), reference_traces=str(Path(traces).resolve()),
                  reference_split='train', training_samples=len(refs), reference_rows=count,
                  role_counts=dict(Counter(roles.values())), roles=roles,
                  methodology='Historical first-query coverage is a training-selection proxy, not proof of necessary search depth; no labels enter prompts/rewards.')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise FileExistsError('Manifest already exists; choose a new output rather than overwrite frozen labels')
    temp = output.with_suffix(output.suffix+'.tmp'); temp.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n'); temp.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True); parser.add_argument('--traces', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = build(args.dataset, args.traces, args.output)
    print(json.dumps({k:result[k] for k in ('training_samples','reference_rows','role_counts','dataset_sha256')}, indent=2))


if __name__ == '__main__': main()
