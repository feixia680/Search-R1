#!/usr/bin/env python3
"""Preflight shared by the evidence launchers; does not start training."""
import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-root', required=True)
parser.add_argument('--evidence-dir', required=True)
parser.add_argument('--retriever-url', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--gpu-ids', required=True)
parser.add_argument('--gpus', required=True)
parser.add_argument('--alpha', required=True)
parser.add_argument('--skip-gpu-checks', action='store_true', help='Validate inputs and BM25 only; never launches training')
args = parser.parse_args()

import math, subprocess, sys
ids, count, alpha = args.gpu_ids, args.gpus, args.alpha
assert len(ids.split(',')) == int(count), f'Expected {count} GPU IDs, got {ids}'
assert len(set(ids.split(','))) == int(count), 'GPU IDs must be distinct'
assert math.isfinite(float(alpha)) and float(alpha) >= 0, 'Invalid evidence alpha'
if not args.skip_gpu_checks:
    result = subprocess.check_output(['nvidia-smi', '-i', ids, '--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'], text=True)
    rows = [line for line in result.splitlines() if line.strip()]
    assert len(rows) == int(count), f'Expected {count} GPUs, found {len(rows)}'
    for row in rows:
        gpu, used = row.split(',')
        if int(used.strip()) >= 500:
            raise SystemExit(f'GPU {gpu.strip()} busy ({used.strip()} MiB); refusing to start')

if not (Path(args.model) / 'config.json').is_file():
    raise SystemExit(f'Base model missing: {args.model}')
if not all((Path(args.evidence_dir) / f'{split}.parquet').is_file()
           for split in ['train', 'test']):
    subprocess.run([sys.executable,
                    str(Path(__file__).with_name('prepare_algorithm_a_evidence.py')),
                    '--base-root', args.base_root, '--out-dir', args.evidence_dir], check=True)

import hashlib, json, sys
from pathlib import Path
import pandas as pd
import requests
base, evidence, url = Path(args.base_root), Path(args.evidence_dir), args.retriever_url
audit = json.loads((evidence / 'annotation_audit.json').read_text())
for split in ['train', 'test']:
    source = base / f'data/processed/hotpotqa/{split}.parquet'
    target = evidence / f'{split}.parquet'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit['splits'][split]['processed_sha256'], f'{split}: baseline data changed'
    assert hashlib.sha256(target.read_bytes()).hexdigest() == audit['splits'][split]['output_sha256'], f'{split}: annotated data changed'
    original = pd.read_parquet(source)
    attached = pd.read_parquet(evidence / f'{split}.parquet')
    assert len(original) == len(attached), f'{split}: sample count differs from baseline'
    assert [r[0]['content'] for r in original.prompt] == [r[0]['content'] for r in attached.prompt], f'{split}: prompts/order differ'
    assert [int(x['index']) for x in original.extra_info] == [int(x['index']) for x in attached.extra_info], f'{split}: IDs differ'
    for extra in attached.extra_info:
        ref = extra.get('evidence', {})
        assert ref.get('schema_version') == 1 and ref.get('status') in ['valid', 'invalid_annotation'], f'{split}: invalid/missing reference'
session = requests.Session()
session.trust_env = False
try:
    response = session.post(url, json={'queries': ['evidence reward readiness check'], 'topk': 3, 'return_scores': True}, timeout=30)
    response.raise_for_status()
    assert isinstance(response.json()['result'], list) and len(response.json()['result']) == 1
except (requests.RequestException, KeyError, ValueError, AssertionError) as exc:
    raise SystemExit(f'BM25 not ready at {url}: {exc}\nStart it first: bash start_bm25_local.sh')
