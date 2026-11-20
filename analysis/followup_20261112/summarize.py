import json
from pathlib import Path
import numpy as np
ROOT=Path('/g22538014wej/11x22f/Search-R1/analysis/followup_20261112')
policy=json.loads((ROOT/'policies/step500_summary.json').read_text())
summaries={k:json.loads((ROOT/f'training_eval/{k}_summary.json').read_text()) for k in ['em','evidence','evidence_cost']}
records={k:[json.loads(s) for s in (ROOT/f'training_eval/{k}.jsonl').read_text().splitlines()] for k in summaries}
comparisons={}
rng=np.random.default_rng(42)
for a,b in [('em','evidence'),('em','evidence_cost'),('evidence','evidence_cost')]:
    assert [r['sample_id'] for r in records[a]]==[r['sample_id'] for r in records[b]]
    d=np.array([y['action_em']-x['action_em'] for x,y in zip(records[a],records[b])])
    boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(axis=1)
    comparisons[f'{a}_vs_{b}']={'delta_em':float(d.mean()),'paired_95ci':np.quantile(boot,[.025,.975]).tolist(),'wrong_to_right':int((d==1).sum()),'right_to_wrong':int((d==-1).sum())}
result=dict(policy=policy,training=summaries,comparisons=comparisons)
(ROOT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# 搜索策略与证据奖励对照结果','','## 搜索策略（step500，同题配对）','','|策略|EM|平均调用数|支持事实覆盖|新增覆盖|','|---|---:|---:|---:|---:|']
for k,s in policy['policies'].items(): lines.append(f"|{k}|{s['em']:.3f}|{s['mean_calls']:.3f}|{s['coverage']:.3f}|{s['mean_added_coverage']:.3f}|")
lines+=['','固定两次与按需策略复用同一首跳；按需条件为首跳答案 token 几何平均概率低于 0.8 或答案为空，不使用 gold。查询和答案使用统一强制前缀，因此属于受控干预评估，不能直接和此前自然动作评估混合比较。第二跳由模型改写 query。保存了完整逐题记录。','','## 奖励训练对照','','每组从同一 Qwen2.5-3B Base 开始，1008题候选子集，同一随机种子、顺序和125次实际更新，batch8、每题2条轨迹。alpha=0.5，beta=0.05。证据奖励为各轮新增可见支持事实覆盖率之和，去重；成本为每次实际检索0.05。验证仅按EM计分。单种子、短程对照，不能据此断言完整训练收敛或稳定优势。','','|组|原始EM|实际answer动作EM|平均检索数|','|---|---:|---:|---:|']
for k,s in summaries.items(): lines.append(f"|{k}|{s['em']:.3f}|{s['action_correct']/s['n']:.3f}|{s['mean_search']:.3f}|")
lines+=['','配对95% bootstrap区间（实际answer动作EM）：',json.dumps(comparisons,ensure_ascii=False,indent=2),'','评估沿用已有200题验证集；这些题曾参与历史检查点选择，本次结果属于验证对照，不是独立测试结果。训练组固定更新数选checkpoint，不按本次评估分数筛选。']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(result,ensure_ascii=False),flush=True)
