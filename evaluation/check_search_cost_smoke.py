"""Audit a completed isolated state-cost GPU smoke run without launching inference."""
import argparse, collections, contextlib, json, math, re
from pathlib import Path
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from search_r1.llm_agent.evidence import score_evidence, score_search_state
from search_r1.llm_agent.trajectory import reward_components

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--smoke-dir',type=Path)
    a=ap.parse_args()
    root=Path.cwd()
    smoke=a.smoke_dir or Path((root/'analysis/cost_smoke_latest.txt').read_text().strip())
    assert (smoke/'exit_code').read_text().strip()=='0','Training did not exit successfully'
    scope=json.loads((smoke/'scope.json').read_text())
    refs={}
    for split in ('train','test'):
        df=pd.read_parquet(root/f'algorithm_a_data/evidence/{split}.parquet')
        refs[split]={str(x['index']):x['evidence'] for x in df.extra_info}
    train=[json.loads(s) for s in (smoke/'run/traces/train.jsonl').open()]
    val=[json.loads(s) for s in (smoke/'run/traces/validation.jsonl').open()]
    assert len(train)==32 and collections.Counter(x['global_step'] for x in train)=={1:16,2:16}
    assert len(val)==8
    assert [x['sample_id'] for x in val]==scope['val_sample_ids']
    for x in train:
        ref=refs['train'][x['sample_id']]
        ev=score_evidence(ref,x) if ref['status']=='valid' else {}
        state=score_search_state(ref,x) if ref['status']=='valid' else {'after_full_search_calls':0}
        expected=reward_components(x['em'],x,alpha=.1,beta=.005,
            evidence_gain=ev.get('evidence_gain',0),after_full_lambda=.01,
            after_full_search_calls=state['after_full_search_calls'],
            require_final_answer=True,max_search_calls=2)
        for key,value in expected.items():
            assert math.isfinite(x[key]) and math.isclose(x[key],value,rel_tol=1e-7,abs_tol=1e-7),(key,x[key],value)
        assert math.isclose(x['final_answer_em'],x['em'])
    for x in val:
        assert x['total_reward']==x['em']
        for key in ('search_cost','after_full_cost','failure_guard_cost','evidence_reward'):
            assert x[key]==0,(key,x[key])
    total_calls=sum(x['retrieval_attempts'] for x in train)
    assert total_calls>0 and sum(x['retrieval_failures'] for x in train+val)==0
    log=re.sub(r'\x1b\[[0-9;]*m','',(smoke/'runner.log').read_text())
    logs=[]
    for line in log.splitlines():
        m=re.search(r'step:(\d+) - ',line)
        if not m: continue
        metrics={}
        for field in line[m.start():].split(' - '):
            k,v=field.rsplit(':',1)
            metrics[k]=float(v)
        assert all(math.isfinite(v) for v in metrics.values())
        if 'actor/grad_norm' in metrics:
            assert metrics['actor/grad_norm']>0
            logs.append(metrics)
    assert [int(x['step']) for x in logs]==[1,2]
    grouped=collections.defaultdict(list)
    for x in train:grouped[(x['global_step'],x['sample_id'])].append(x['total_reward'])
    nonconstant=sum(max(v)>min(v) for v in grouped.values())
    assert nonconstant>0
    checkpoint=smoke/'run/checkpoints/actor/global_step_2'
    base=root/'models/Qwen2.5-3B'
    maps=[json.loads((p/'model.safetensors.index.json').read_text())['weight_map'] for p in (base,checkpoint)]
    AutoConfig.from_pretrained(checkpoint,local_files_only=True)
    tok=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
    assert len(tok.encode('smoke checkpoint'))>0
    delta_max=0.;changed=0;numel=0
    with contextlib.ExitStack() as stack:
        handles=[]
        for path,mapping in zip((base,checkpoint),maps):
            opened={}
            for name in sorted(set(mapping.values())):
                opened[name]=stack.enter_context(safe_open(path/name,framework='pt',device='cpu'))
            handles.append(opened)
        for key,shard in maps[1].items():
            assert key in handles[1][shard].keys()
            s=handles[1][shard].get_slice(key)
            shape=s.get_shape()
            index=tuple(slice(0,min(n,64)) for n in shape)
            actual=s[index].float()
            assert torch.isfinite(actual).all()
            if key not in maps[0]:continue
            original=handles[0][maps[0][key]].get_slice(key)[index].float()
            assert actual.shape==original.shape
            delta=(actual-original).abs()
            changed+=int((delta>0).sum())
            numel+=actual.numel()
            delta_max=max(delta_max,float(delta.max()))
    assert changed>0,'Saved policy samples equal base; optimizer update not demonstrated'
    result=dict(status='PASS',smoke_dir=str(smoke),training_steps=[1,2],train_trajectories=len(train),
        validation_trajectories=len(val),validation_original200_ids=scope['val_sample_ids'],
        train_retrieval_calls=total_calls,
        train_after_full_calls=sum(x['after_full_search_calls'] for x in train),
        train_positive_evidence_trajectories=sum(x['evidence_gain']>0 for x in train),
        train_missing_answers=sum(not str(x.get('final_answer') or '').strip() for x in train),
        train_guarded_trajectories=sum(x['failure_guard_cost']>0 for x in train),
        train_reward_min=min(x['total_reward'] for x in train),train_reward_max=max(x['total_reward'] for x in train),
        nonconstant_grpo_groups=nonconstant,
        retrieval_failures=0,all_trace_reward_components_verified=True,
        actor_grad_norm=[x['actor/grad_norm'] for x in logs],
        checkpoint=str(checkpoint),checkpoint_tensor_keys=len(maps[1]),
        sampled_parameter_values=numel,sampled_changed_parameter_values=changed,max_sampled_parameter_delta=delta_max,
        scope='1-GPU two-update execution test; not an accuracy or long-run stability result.')
    (smoke/'SMOKE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 1GPU 搜索成本 smoke test','','状态：PASS','',f'目录：{smoke}','',
        f"- 完成两次真实 GRPO 更新；训练轨迹 {len(train)} 条，验证原 200 题中的 8 题。",
        f"- 检索调用 {total_calls} 次；检索失败 0；充分后搜索 {result['train_after_full_calls']} 次。",
        f"- 正证据奖励轨迹 {result['train_positive_evidence_trajectories']} 条；缺答案保护触发 {result['train_guarded_trajectories']} 条。",
        f"- 所有训练轨迹的成本和总奖励重算一致；验证保持纯 EM。",
        f"- 非零奖励方差采样组 {nonconstant} 个；两步梯度范数 {result['actor_grad_norm']}。",
        f"- step-2 checkpoint 已保存、权重分片可读，tokenizer/config 可加载；抽查 {numel} 个参数值，其中 {changed} 个较 base 改变。",
        '', '本次关闭学习率 warmup 以验证实际参数更新；正式训练默认 warmup 保持原配置。',
        '仅验证执行链路，不证明训练收益、长时间稳定性或两 GPU 可运行性。']
    if not result['train_after_full_calls']:
        lines+=['充分后额外成本分支未被本次随机生成触发；其非零数值已在独立 CPU 集成测试验证。']
    (smoke/'SMOKE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
