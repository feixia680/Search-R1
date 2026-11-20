import collections
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path('/g22538014wej/11x22f/Search-R1')
OUT=ROOT/'analysis/unified_eval_20261113'
POLICIES=['fixed_one','fixed_two','autonomous']
COHORTS=['original200','heldout500']
MODELS=['baseline','evidence']

def stats(xs):
    avg=lambda k:float(np.mean([float(x.get(k,0) or 0) for x in xs]))
    second=[x for x in xs if x['retrieval_attempts']>=2]
    gains=[x['evidence_events'][1]['evidence_gain'] for x in second]
    return dict(n=len(xs),correct=int(sum(x['em'] for x in xs)),em=avg('em'),
        final_answer_em=avg('final_answer_em'),mean_calls=avg('retrieval_attempts'),
        calls_hist=dict(collections.Counter(x['retrieval_attempts'] for x in xs)),
        coverage=avg('evidence_coverage'),full_evidence=sum(x['evidence_coverage']==1 for x in xs),
        second_calls=len(second),second_new_evidence=sum(g>0 for g in gains),
        second_zero_evidence=sum(g==0 for g in gains),
        second_after_full=sum(x['evidence_events'][0]['cumulative_coverage']==1 for x in second),
        mean_second_gain=float(sum(gains)/len(xs)),
        mean_observation_tokens=float(np.mean([sum(e.get('observation_tokens_after_truncation',0) for e in x['events'] if e.get('attempted')) for x in xs])),
        missing_final=sum(not str(x.get('final_answer') or '').strip() for x in xs),
        invalid_actions=sum(x['invalid_actions'] for x in xs),
        duplicate_queries=sum(x['duplicate_queries'] for x in xs),
        retrieval_failures=sum(x['retrieval_failures'] for x in xs))

def compare(a,b):
    assert set(a)==set(b)
    ids=sorted(a,key=int)
    dif=np.array([b[i]['em']-a[i]['em'] for i in ids],dtype=float)
    good=sum(a[i]['em']==0 and b[i]['em']==1 for i in ids)
    bad=sum(a[i]['em']==1 and b[i]['em']==0 for i in ids)
    rng=np.random.default_rng(42)
    ci=np.quantile(dif[rng.integers(0,len(dif),size=(10000,len(dif)))].mean(axis=1),[.025,.975])
    discord=good+bad
    p=min(1.,2*sum(math.comb(discord,j) for j in range(min(good,bad)+1))/2**discord) if discord else 1.
    baseline=np.mean([a[i]['em'] for i in ids])
    return dict(n=len(ids),em_delta=float(dif.mean()),relative_em_improvement=float(dif.mean()/baseline) if baseline else None,
        paired_bootstrap_95ci=list(map(float,ci)),wrong_to_right=good,right_to_wrong=bad,
        exact_mcnemar_p=p,mean_calls_delta=float(np.mean([b[i]['retrieval_attempts']-a[i]['retrieval_attempts'] for i in ids])),
        mean_coverage_delta=float(np.mean([b[i]['evidence_coverage']-a[i]['evidence_coverage'] for i in ids])))

data={}
rows=[]
for model in MODELS:
    assert (OUT/f'{model}_COMPLETE.json').exists()
    for policy in POLICIES:
        xs=[json.loads(s) for s in (OUT/f'{model}_{policy}.jsonl').open()]
        assert len(xs)==700
        assert len({x['sample_id'] for x in xs})==700
        for cohort in COHORTS:
            selected=[x for x in xs if x['cohort']==cohort]
            data[model,policy,cohort]={x['sample_id']:x for x in selected}
            row=dict(model=model,policy=policy,cohort=cohort,**stats(selected))
            rows.append(row)
for model in MODELS:
    for cohort in COHORTS:
        a=data[model,'fixed_one',cohort];b=data[model,'fixed_two',cohort]
        assert set(a)==set(b)
        for sid in a:
            ea=[e for e in a[sid]['events'] if e.get('attempted')][0]
            eb=[e for e in b[sid]['events'] if e.get('attempted')][0]
            assert ea['query']==eb['query'] and ea['observation_text']==eb['observation_text'],(model,sid,'first hop differs')

between={}
within={}
for cohort in COHORTS:
    for policy in POLICIES:
        between[f'{cohort}/{policy}']=compare(data['baseline',policy,cohort],data['evidence',policy,cohort])
    for model in MODELS:
        pair=compare(data[model,'fixed_one',cohort],data[model,'fixed_two',cohort])
        b=data[model,'fixed_two',cohort]
        pair['second_adds_evidence_corrected_answer']=sum(
            not data[model,'fixed_one',cohort][i]['em'] and x['em'] and x['evidence_events'][1]['evidence_gain']>0
            for i,x in b.items())
        within[f'{cohort}/{model}/fixed_two_minus_one']=pair
summary=dict(rows=rows,between_models=between,within_model=within,
    fixed_first_query_and_observation_identical=True,
    primary_comparison='heldout500/autonomous',
    protocol=json.loads((OUT/'protocol.json').read_text()),
    caveats=['One training seed per method; old baseline shuffle seed not recorded.',
        'Forced policies enforce action prefixes and closing tags. Autonomous uses native actions; protocol differs.',
        'Forced fixed policies share first-hop query tokens. No wall-clock latency claim.',
        'Full gold support coverage is not proof that the answer is correct.',
        'Heldout500 excludes selection200, normalized training-question matches, duplicate questions and invalid annotations; no exhaustive benchmark contamination audit.',
        'Paired question bootstrap measures evaluation-sample uncertainty, not training-seed variability. Secondary p-values are uncorrected for multiple comparisons.'])
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
keys=list(dict.fromkeys(k for r in rows for k in r))
with (OUT/'results.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

lines=['# 两模型统一评测与搜索预算对照','','模型均为 step-500 checkpoint。原 200 题复测，另冻结 500 道未用于选 checkpoint 的题目。没有训练。',
       '', '主比较预先指定为 heldout500/autonomous。所有策略共用同一执行器、检索服务、tokenizer、greedy 解码及上下文限制。',
       '', '## 结果', '', '| 题目集 | 策略 | 模型 | 正确/题数 | EM | 实际检索/题 | 支持事实覆盖率 | 检索 token/题 |',
       '|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['cohort']} | {r['policy']} | {r['model']} | {r['correct']}/{r['n']} | {r['em']:.1%} | {r['mean_calls']:.3f} | {r['coverage']:.2%} | {r['mean_observation_tokens']:.1f} |")
lines+=['','## 配对比较','','下表为 evidence 减 baseline；CI 为按问题配对 bootstrap 10,000 次。','',
        '| 题目集/策略 | EM差值 | 95% CI | 相对EM变化 | 错→对 | 对→错 |',
        '|---|---:|---:|---:|---:|---:|']
for name,r in between.items():
    lo,hi=r['paired_bootstrap_95ci']
    rel=f"{r['relative_em_improvement']:+.1%}" if r['relative_em_improvement'] is not None else 'NA'
    lines.append(f"| {name} | {r['em_delta']*100:+.1f} pp | [{lo*100:+.1f}, {hi*100:+.1f}] pp | {rel} | {r['wrong_to_right']} | {r['right_to_wrong']} |")
lines+=['','## 第二次检索的答题收益','','同模型的固定两次减固定一次，共享第一跳查询和返回内容；第二跳后重新作答。','',
        '| 题目集/模型 | EM差值 | 95% CI | 错→对 | 对→错 | 新证据且纠正答案 |',
        '|---|---:|---:|---:|---:|---:|']
for name,r in within.items():
    lo,hi=r['paired_bootstrap_95ci']
    lines.append(f"| {name} | {r['em_delta']*100:+.1f} pp | [{lo*100:+.1f}, {hi*100:+.1f}] pp | {r['wrong_to_right']} | {r['right_to_wrong']} | {r['second_adds_evidence_corrected_answer']} |")
lines+=['','## 自主策略的第二跳','','| 题目集 | 模型 | 搜两次题数 | 第二跳新增证据 | 无新增gold证据 | 第一跳已全覆盖仍搜 | 缺答案 |',
        '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    if r['policy']=='autonomous':
        lines.append(f"| {r['cohort']} | {r['model']} | {r['second_calls']} | {r['second_new_evidence']} | {r['second_zero_evidence']} | {r['second_after_full']} | {r['missing_final']} |")
lines+=['','## 解释范围','',
        '固定一次/两次使用外部动作前缀约束，两个模型的协议完全相同；自主策略直接使用模型原生动作。模型在相同固定预算下的差异，可以检验收益是否超出调用数变化；自主与固定策略的差异也包含前缀约束的影响，不能全部归因于预算。',
        '', '固定两次与固定一次的第一跳查询及返回内容逐题核验一致。支持事实严格按原有 token 可见性规则评分，模型自行生成的文本不计为检索证据。覆盖率为逐题平均。检索 token 统计每次 observation 截断后 token 数，不含生成 token，也不等于整段累计上下文长度。',
        '', '原200题已用于验证和选 checkpoint，只作复测；heldout500 是本次新增评估，不在看到结果后重新选择模型或预算。它仍来自同一数据集，因此还不能代表跨数据集泛化。',
        '', *['- '+c for c in summary['caveats']],
        '', '完整逐题轨迹、冻结 manifest、协议、CSV、配对统计保存在同目录。analysis/ 不推送。']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
