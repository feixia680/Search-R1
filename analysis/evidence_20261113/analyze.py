import collections
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from search_r1.llm_agent.evidence import score_evidence

root = Path.cwd()
run = root / 'algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015'
out = root / 'analysis/evidence_20261113'
out.mkdir(parents=True, exist_ok=True)

def metrics(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', path.read_text(errors='replace'))
    result = []
    for line in text.splitlines():
        if 'step:' not in line:
            continue
        items = dict(re.findall(r'([A-Za-z0-9_/]+):(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)', line))
        if 'step' in items:
            result.append({k: float(v) for k, v in items.items()})
    return result

logs = {'baseline': metrics(root / 'hotpotqa-qwen25-3b-grpo-2gpu.log'),
        'evidence': metrics(run / 'train.log')}
val_scores = {label: {int(x['step']): x['val/test_score/hotpotqa'] for x in xs
                     if 'val/test_score/hotpotqa' in x} for label, xs in logs.items()}
windows = []
for label, xs in logs.items():
    for lo, hi in [(1,100),(101,200),(201,300),(301,400),(401,500),(525,624)]:
        selected = [x for x in xs if lo <= x['step'] <= hi and 'actor/pg_loss' in x]
        row = {'model': label, 'start': lo, 'end': hi, 'updates': len(selected)}
        for k in ['critic/score/mean','env/number_of_valid_search','env/finish_ratio',
                  'response_length/mean','actor/entropy_loss','actor/grad_norm',
                  'algorithm_a/evidence_coverage','algorithm_a/em','timing_s/gen']:
            values = [x[k] for x in selected if k in x]
            if values:
                row[k] = statistics.mean(values)
        row['all_zero_advantage_update_fraction'] = statistics.mean(
            x.get('critic/advantages/min') == 0 and x.get('critic/advantages/max') == 0
            for x in selected)
        windows.append(row)

def summarize(xs):
    n = len(xs)
    avg = lambda k: statistics.mean(float(x.get(k,0) or 0) for x in xs)
    second = [x for x in xs if x['retrieval_attempts'] >= 2]
    added = [x for x in second if len(x.get('evidence_events',[])) >= 2
             and x['evidence_events'][1]['evidence_gain'] > 0]
    full_before = [x for x in second if x['evidence_events'][0]['cumulative_coverage'] >= 1]
    full = [x for x in xs if x.get('evidence_coverage',0) >= 1]
    missing = [x for x in xs if x.get('final_answer') is None or not str(x.get('final_answer','')).strip()]
    first = [sum(e['evidence_gain'] for e in x.get('evidence_events',[])[:1]) for x in xs]
    second_gain = [sum(e['evidence_gain'] for e in x.get('evidence_events',[])[1:2]) for x in xs]
    return dict(n=n, em=avg('em'), correct=sum(x['em'] for x in xs),
                final_answer_em=avg('final_answer_em'), coverage=avg('evidence_coverage'),
                mean_attempts=avg('retrieval_attempts'), attempts_hist=dict(collections.Counter(x['retrieval_attempts'] for x in xs)),
                second_calls=len(second), second_adds_evidence=len(added), second_no_new_evidence=len(second)-len(added),
                second_after_full_evidence=len(full_before), first_hop_coverage=statistics.mean(first),
                second_hop_coverage_gain=statistics.mean(second_gain), full_evidence=len(full),
                full_evidence_em_mismatch=sum(not x['em'] for x in full), missing_final=len(missing),
                mean_duplicates=avg('duplicate_queries'), mean_invalid=avg('invalid_actions'),
                retrieval_failures=sum(x['retrieval_failures'] for x in xs))

refs = {}
for split, filename in [('train','train'),('validation','test')]:
    df = pd.read_parquet(root / f'algorithm_a_data/evidence/{filename}.parquet')
    refs[split] = {str(x['index']): x['evidence'] for x in df.extra_info}
train = collections.defaultdict(list)
validation = collections.defaultdict(list)
audit = collections.Counter()
pairs = collections.defaultdict(list)
for split, target in [('train',train),('validation',validation)]:
    for line in (run / f'traces/{split}.jsonl').open():
        x = json.loads(line)
        target[x['global_step']].append(x)
        ref = refs[split][x['sample_id']]
        if ref['status'] == 'valid':
            recomputed = score_evidence(ref, x)
            assert abs(recomputed['evidence_gain']-x['evidence_gain']) < 1e-10
            assert recomputed['evidence_hit_count'] == x['evidence_hit_count']
            assert abs(sum(e['evidence_gain'] for e in x['evidence_events'])-x['evidence_gain']) < 1e-10
            audit[f'{split}_valid_recomputed'] += 1
        else:
            assert not x['evidence_eligible']
            audit[f'{split}_invalid_annotation_skipped'] += 1
        bonus = .1 * x.get('evidence_gain',0) if split == 'train' else 0
        assert abs(x['evidence_reward']-bonus) < 1e-10
        assert abs(x['total_reward']-x['em']-bonus) < 1e-10
        assert x['retrieval_attempts'] <= 2
        audit[f'{split}_reward_arithmetic_passed'] += 1
        if split == 'train':
            pairs[(x['global_step'],x['sample_id'])].append(x)

val = {s:summarize(xs) for s,xs in validation.items()}
assert all(len(xs)==200 for xs in validation.values())
assert all({x['sample_id'] for x in xs} == {x['sample_id'] for x in validation[500]}
           for xs in validation.values())
pair_summary = collections.Counter()
for xs in pairs.values():
    assert len(xs)==2
    pair_summary['groups'] += 1
    if xs[0]['em']==xs[1]['em']:
        pair_summary['same_em'] += 1
        if abs(xs[0].get('evidence_gain',0)-xs[1].get('evidence_gain',0)) > 1e-10:
            pair_summary['same_em_different_evidence'] += 1
    else:
        pair_summary['different_em'] += 1

cfg = OmegaConf.to_container(OmegaConf.load(run/'hydra/.hydra/config.yaml'),resolve=True)
baseline_cfg = OmegaConf.to_container(OmegaConf.load(root/'outputs/2026-11-11/11-37-52/.hydra/config.yaml'),resolve=True)
def flatten(x, prefix=''):
    result = {}
    for k,v in x.items():
        if isinstance(v,dict): result.update(flatten(v,prefix+k+'.'))
        else: result[prefix+k] = v
    return result
a,b=flatten(baseline_cfg),flatten(cfg)
diff={k:{'baseline':a.get(k),'evidence':b.get(k)} for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k)}
checks=[]
for split in ['train','test']:
    original=pd.read_parquet(root/f'data/processed/hotpotqa/{split}.parquet')
    attached=pd.read_parquet(root/f'algorithm_a_data/evidence/{split}.parquet')
    assert len(original)==len(attached)==5000
    assert [x[0]['content'] for x in original.prompt]==[x[0]['content'] for x in attached.prompt]
    assert [str(x['index']) for x in original.extra_info]==[str(x['index']) for x in attached.extra_info]
    checks.append(split+': 5000 prompts and IDs identical, same row order')
summary=dict(run=str(run),validation_scores=val_scores,validation_traces=val,
             windows=windows,trace_audit=dict(audit),pair_summary=dict(pair_summary),
             config_differences=diff,data_checks=checks,
             actual_updates={k:sum('actor/pg_loss' in x for x in xs) for k,xs in logs.items()},
             saved_checkpoints=sorted(p.name for p in (run/'checkpoints/actor').iterdir() if p.is_dir()),
             caveats=['Single seed; historical baseline shuffle seed not recorded.',
                      'Training validation vs offline reevaluation scores differ; do not mix protocols.',
                      'Final validation labeled step625 follows update624; final weights were not saved.',
                      'Observation traces count visible gold supporting sentences; coverage is not answer correctness.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
for name,rows in [('validation', [dict(step=s,baseline_em=val_scores['baseline'][s],**x) for s,x in val.items()]),('training_windows',windows)]:
    keys=list(dict.fromkeys(k for x in rows for k in x))
    with (out/f'{name}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ['config_differences','windows']},ensure_ascii=False,indent=2))
print('WINDOWS',json.dumps(windows,ensure_ascii=False))
