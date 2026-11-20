"""CPU replay of saved rewards; does not infer counterfactual EM or stopping."""
import argparse
import collections
import json
from pathlib import Path
import statistics
import pandas as pd
from search_r1.llm_agent.evidence import score_search_state
from search_r1.llm_agent.trajectory import reward_components

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out-dir', type=Path, default=Path('analysis/cost_design_20261113'))
    a=parser.parse_args();root=a.root.resolve();out=a.out_dir
    out.mkdir(parents=True,exist_ok=True)
    refs={}
    sizes={}
    for split in ('train','test'):
        frame=pd.read_parquet(root/'algorithm_a_data/evidence'/f'{split}.parquet')
        refs[split]={str(x['index']):x['evidence'] for x in frame.extra_info}
        sizes[split]=dict(collections.Counter(len(x['supporting_facts']) for x in refs[split].values() if x['status']=='valid'))
    sources=[
        ('train_all',root/'algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/traces/train.jsonl','train'),
        ('original200_autonomous',root/'analysis/unified_eval_20261113/evidence_autonomous.jsonl','test')]
    groups={}
    for name,path,split in sources:
        rows=[]
        for line in path.open():
            x=json.loads(line)
            if split=='test' and x.get('cohort')!='original200': continue
            ref=refs[split][str(x['sample_id'])]
            state=score_search_state(ref,x) if ref['status']=='valid' else dict(after_full_search_calls=0)
            gains=[e['evidence_gain'] for e in x.get('evidence_events',[])]
            rows.append(dict(trace=x,state=state,gains=gains))
        groups[name]=rows
        if split=='train':
            groups['train_steps375plus']=[r for r in rows if (r['trace'].get('global_step') or 0)>=375]
    results={}
    for name,rows in groups.items():
        n=len(rows); pos=[g for r in rows for g in r['gains'] if g>0]
        miss=[r['trace'] for r in rows if not str(r['trace'].get('final_answer') or '').strip()]
        result=dict(n=n,mean_calls=statistics.mean(r['trace']['retrieval_attempts'] for r in rows),
            positive_gains=len(pos),median_positive_gain=statistics.median(pos) if pos else None,
            missing_answer=len(miss),
            missing_termination=dict(collections.Counter(x.get('termination_reason') for x in miss)),
            missing_last_action=dict(collections.Counter(x['events'][-1]['action'] for x in miss)),
            after_full_calls=sum(r['state']['after_full_search_calls'] for r in rows),candidates=[])
        for beta,lam in [(0,0),(.002,0),(.005,0),(.01,0),(.02,0),(.05,0),(.005,.01)]:
            rewards=[]
            for r in rows:
                x=r['trace']
                rewards.append(reward_components(x['em'],x,alpha=.1,beta=beta,
                    evidence_gain=x.get('evidence_gain',x.get('evidence_coverage',0)),
                    after_full_lambda=lam,after_full_search_calls=r['state']['after_full_search_calls'],
                    require_final_answer=True,max_search_calls=2)['total_reward'])
            result['candidates'].append(dict(beta=beta,after_full_lambda=lam,
                mean_resource_cost=beta*result['mean_calls']+lam*result['after_full_calls']/n,
                mean_guarded_reward=statistics.mean(rewards),
                positive_gain_calls_with_bonus_above_base_cost=sum(.1*g>beta for g in pos),
                positive_gain_calls=len(pos)))
        results[name]=result
    output=dict(support_fact_count_distribution=sizes,groups=results,
        warning='Reward replay only. All candidate arms use the same final-answer guard. No causal prediction of new EM, call counts, or best beta.')
    (out/'calibration.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# 搜索成本尺度检查','','仅重放已有轨迹；不能预测新模型 EM 或停止行为。全部候选采用相同缺答案保护。',
           '', '|轨迹|数量|平均调用|充分后调用|缺答案|正增益中位数|','|---|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"|{name}|{r['n']}|{r['mean_calls']:.3f}|{r['after_full_calls']}|{r['missing_answer']}|{r['median_positive_gain']}|")
    for name,r in results.items():
        lines+=['',f'## {name}','','|beta|充分后额外成本|平均资源扣分|平均保护后奖励|正增益奖励超过基础成本的调用|',
                '|---:|---:|---:|---:|---:|']
        for c in r['candidates']:
            lines.append(f"|{c['beta']}|{c['after_full_lambda']}|{c['mean_resource_cost']:.5f}|{c['mean_guarded_reward']:.5f}|{c['positive_gain_calls_with_bonus_above_base_cost']}/{c['positive_gain_calls']}|")
        lines+=['',f"缺答案终止原因：{r['missing_termination']}；最后动作：{r['missing_last_action']}。"]
    lines+=['','建议初始候选：beta=0.005，充分后额外成本=0.01；这只是保持证据奖励尺度的起点，不是验证过的最优参数。',
            'gamma/delta=0；不额外惩罚零gold新增。参考事实只用于训练奖励，不进入策略输入。']
    (out/'CALIBRATION.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:{key:v for key,v in r.items() if key!='candidates'} for k,r in results.items()},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
