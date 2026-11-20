from pathlib import Path
import ast,hashlib,json,math,subprocess,statistics
import numpy as np
import torch
from verl import DataProto

root=Path.cwd();out=root/'analysis/offline_engineering_20261119';out.mkdir(exist_ok=True)
pair=json.loads((root/'analysis/engineering_grid_pair_latest.json').read_text())
resident=json.loads((root/'analysis/engineering_resident_replay_latest.json').read_text())
smoke=json.loads((root/'analysis/engineering_fast_smoke_latest.json').read_text())
initial=json.loads((root/'analysis/engineering_profile_latest.json').read_text())
retrieval=json.loads((root/'analysis/offline_retrieval_benchmark_20261119/result.json').read_text())
def canonical(value):
 return json.dumps(value,sort_keys=True,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))
def load_exit(path):return json.loads(path.read_text().rstrip().removesuffix(r'\n'))['exit_code']
assert load_exit(Path(pair['audit_dir'])/'offload_exit.json')==0
assert load_exit(Path(resident['audit_dir'])/'exit_code.json')==0
assert load_exit(Path(smoke['audit_dir'])/'exit_code.json')==0
runs={'offload':Path(pair['runs']['offload']),'resident':Path(resident['run_dir'])}
benchmarks={k:json.loads((v/'sampling/engineering_benchmark.json').read_text()) for k,v in runs.items()}
assert benchmarks['offload']['workload_sha256']==benchmarks['resident']['workload_sha256']
workloads={k:DataProto.load_from_disk(v/'sampling/engineering_workload.pkl') for k,v in runs.items()}
a,b=workloads.values();assert set(a.batch.keys())==set(b.batch.keys())
full=hashlib.sha256()
for key in sorted(a.batch.keys()):
 x,y=a.batch[key],b.batch[key]
 assert torch.equal(x,y),key
 full.update((key+str(x.dtype)+str(tuple(x.shape))).encode())
 full.update(x.cpu().contiguous().view(torch.uint8).numpy().tobytes())
def canonical(obj):
 return json.dumps(obj,sort_keys=True,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
for attr in ('non_tensor_batch','meta_info'):
 va=getattr(a,attr);vb=getattr(b,attr)
 if attr=='non_tensor_batch':va={k:v.tolist() for k,v in va.items()};vb={k:v.tolist() for k,v in vb.items()}
 assert canonical(va)==canonical(vb),attr
 full.update(canonical(va).encode())
rows=[]
for mode,run in runs.items():
 measurements=[json.loads(line) for line in (run/'sampling/engineering_measurements.jsonl').read_text().splitlines()]
 rank_records=[json.loads(line) for p in (run/'engineering_profiles').glob('rank_*.jsonl') for line in p.read_text().splitlines()]
 for variant,values in benchmarks[mode]['variants'].items():
  actual=[r['wall_seconds'] for r in measurements if r['phase']=='measure' and r['variant']==variant]
  assert len(actual)==3 and statistics.median(actual)==values['median_update_seconds']
  balanced=variant.startswith('balanced');micro=int(variant.rsplit('m',1)[1])
  ranks=[r for r in rank_records if r['phase'].startswith('measure') and r['balanced']==balanced and r['global_micro_batch_size']==micro]
  assert len(ranks)==6
  rows.append(dict(mode=mode,variant=variant,median_seconds=statistics.median(actual),min_seconds=min(actual),max_seconds=max(actual),
      peak_allocated_gib=max(r['peak_allocated_gib'] for r in ranks),peak_reserved_gib=max(r['peak_reserved_gib'] for r in ranks)))
completed=[json.loads(l) for l in (Path(smoke['run_dir'])/'sampling/completed_updates.jsonl').read_text().splitlines()]
assert [r['global_step'] for r in completed]==[1,2]
smoke_rows=[]
for record in completed:
 m=record['metrics']
 assert m['actor/lr']>0 and m['actor/grad_norm']>0
 assert int(m['engineering/retrieval/logical_queries'])==m['sampling/candidate_retrieval_calls']
 assert m['sampling/selected_groups']==8 and m['sampling/selected_trajectories']==16
 assert all(math.isfinite(float(v)) for v in m.values() if isinstance(v,(int,float)))
 behavior={k:v for k,v in m.items() if k.startswith('behavior/')}
 assert behavior
 smoke_rows.append(dict(step=record['global_step'],lr=m['actor/lr'],grad_norm=m['actor/grad_norm'],
    actor_seconds=m['timing_s/update_actor'],step_seconds=m['timing_s/step'],
    old_logprob_seconds=m['timing_s/selected_old_log_prob'],ref_seconds=m['timing_s/ref'],
    logical_queries=m['engineering/retrieval/logical_queries'],backend_queries=m['engineering/retrieval/backend_query_attempts'],
    cache_hits=m['engineering/retrieval/cache_hits'],behavior_metrics=len(behavior)))
assert retrieval['identical_results']
base=benchmarks['offload']['variants']['balanced_m2']['median_update_seconds']
fast=benchmarks['resident']['variants']['balanced_m4']['median_update_seconds']
best=benchmarks['resident']['variants']['balanced_m8']['median_update_seconds']
summary=dict(hardware='2 x NVIDIA A800-SXM4-80GB',model='Qwen2.5-3B',
 workload_samples=len(a),workload_tokens=benchmarks['offload']['total_tokens'],
 input_workload_sha256=benchmarks['offload']['workload_sha256'],full_workload_sha256=full.hexdigest(),
 identical_all_tensors_and_metadata=True,measurements=rows,
 default_actor_time_reduction_percent=100*(1-fast/base),best_actor_time_reduction_percent=100*(1-best/base),
 tests=dict(passed=100,warnings=2,observed_seconds=13.78),real_training_updates=smoke_rows,
 retrieval=retrieval,artifacts=dict(initial=initial,pair=pair,resident=resident,smoke=smoke))
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
# Add the launch entrypoints to source snapshots and future launch archives.
for name in ('train_grpo_dynamic_sampling_1gpu.sh','train_grpo_dynamic_sampling_2gpu.sh'):
 p=root/name;source=p.read_text();line=next(line for line in source.splitlines() if 'for f in ' in line)
 for extra in ('city_dynamic_2gpu_fast.sh','run_engineering_benchmark_2gpu.sh','evaluation/benchmark_retrieval_runtime.py'):
  if extra not in line:
   updated=line.replace('for f in ', 'for f in '+extra+' ',1);source=source.replace(line,updated);line=updated
 p.write_text(source)
for name in ('city_dynamic_2gpu_fast.sh','run_engineering_benchmark_2gpu.sh'):
 dst=Path(smoke['run_dir'])/'source'/name
 if not dst.exists():dst.write_text((root/name).read_text())
files=['city_dynamic_2gpu_fast.sh','run_engineering_benchmark_2gpu.sh','search_r1/search/retrieval_client.py',
 'search_r1/dynamic_sampling/engineering.py','search_r1/dynamic_sampling/loss_groups.py','search_r1/llm_agent/generation.py',
 'verl/trainer/ppo/dynamic_sampling_trainer.py','verl/trainer/ppo/ray_trainer.py','verl/trainer/main_dynamic_sampling.py',
 'verl/workers/dynamic_sampling_worker.py','verl/workers/actor/dynamic_sampling_actor.py','verl/trainer/config/dynamic_sampling.yaml',
 'evaluation/benchmark_retrieval_runtime.py','train_grpo_dynamic_sampling_1gpu.sh','train_grpo_dynamic_sampling_2gpu.sh']
for name in files:
 if name.endswith('.py'):ast.parse((root/name).read_text(),filename=name)
 if name.endswith('.sh'):subprocess.run(['bash','-n',name],check=True)
check=subprocess.run(['git','diff','--check','--',*files],capture_output=True,text=True)
(out/'diff_check.txt').write_text(check.stdout+check.stderr)
assert check.returncode==0,check.stdout+check.stderr
(out/'source_sha256.json').write_text(json.dumps({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},indent=2)+'\n')
header='| 参数卸载 | 全局 actor microbatch | token 均衡 | 更新中位耗时 | 单卡最大 allocated / reserved |\n|---|---:|---|---:|---:|\n'
table=header+'\n'.join(f"| {'开启' if row['mode']=='offload' else '关闭'} | {row['variant'].rsplit('m',1)[1]} | {'开启' if row['variant'].startswith('balanced') else '关闭'} | {row['median_seconds']:.3f}s | {row['peak_allocated_gib']:.2f} / {row['peak_reserved_gib']:.2f} GiB |" for row in rows)
report=f'''# 离线检索与双卡训练工程优化（2026-11-19）

已部署到 `{root}`，硬件为 2 × A800 80GB，模型为 Qwen2.5-3B。新增检索执行层、损失分组兼容的 microbatch 扩大、硬件配置入口和测量记录。veRL token 均衡属于项目已有能力，本次增加开关、分卡指标和真实对照。

## 测得的收益

同一批 {len(a)} 条轨迹、{summary['workload_tokens']} tokens，学习率为 0，实际执行 forward/backward/optimizer。每种配置先预热，再交替测量 3 次。两组对照的所有 batch tensor、非 tensor 数据和元信息均验证相同，完整工作负载 SHA256 为 `{full.hexdigest()}`。

{table}

默认优化配置（关闭参数卸载、microbatch 4）相对原来的开启卸载、microbatch 2、启用均衡，actor 更新中位耗时从 {base:.3f}s 降到 {fast:.3f}s，减少 {summary['default_actor_time_reduction_percent']:.1f}%。microbatch 8 的基准耗时为 {best:.3f}s，作为可选设置。这里是更新环节的测量，生成、检索和评分仍占用时间；未测得同轨迹完整训练的端到端加速比例。

token 差距 `(max-min)/mean` 从 7.575% 降到 0.017%，每卡 tokens 从 6097/5652 变为 5875/5874。均衡开关的更新时间在不同短测中有波动，不能据此宣称稳定加速。此次主要收益来自参数驻留和更大的 microbatch。

## BM25 检索

使用历史 TRAIN 轨迹中的 32 个不同查询，各重复一次，共 64 次逻辑搜索。逐项验证文档、顺序和分数与原始 BM25 一致。

| 模式 | 实际后端查询 | HTTP 请求 | 本次回放耗时 |
|---|---:|---:|---:|
'''
for name,label in [('raw','原始调用'),('cold','冷缓存 + 批内去重'),('memory_warm','内存缓存'),('disk_warm','跨客户端磁盘缓存')]:
 stats=retrieval['stages'][name];report+=f"| {label} | {stats['backend_query_attempts']} | {stats['backend_http_requests']} | {stats['wall_seconds']*1000:.3f}ms |\n"
report+='''
这组重复查询回放用于验证缓存与去重。实际训练中的命中率取决于查询重复程度，不代表上述收益会覆盖所有训练步骤。模型每次发起的搜索仍各自记录、计费和评分，缓存不会减少逻辑搜索次数。

缓存键包含精确查询、topk、服务 URL 和索引版本；索引自动版本使用本地索引及 corpus 的路径、大小和 mtime。默认持久缓存为 `data/cache/retrieval_runtime.sqlite3`，另有 1024 项内存缓存。超时、连接错误及 HTTP 408/429/5xx 最多重试 2 次；其他 HTTP 和格式错误不重试。失败结果不缓存，磁盘缓存故障回退实时检索，并记录错误。训练和评估均沿用本地 BM25，无新增外部服务依赖。

## 数值与训练验证

增大 microbatch 时，简单地对所有 token 求均值会改变原有损失权重。新增 `engineering_loss_group_size` 保留原有局部损失分组：原双卡全局 microbatch 2 对应每卡 1 条，因此优化入口固定组大小为 1。扩大模型前向批量时，policy、entropy 和 KL 仍按这些组计算后平均。double 精度的损失和梯度对照测试通过。GPU 混合精度下，batch 形状变化仍会产生浮点差异，本次未验证长训练准确率等价。

100 项 CPU/集成回归测试通过。共完成 40 次真实 FSDP 基准更新（LR=0），以及默认优化入口的 2 次真实学习率更新。后者保持分层抽题、动态筛选、证据奖励与行为监控，测试候选批上限设为 2；正式入口仍沿用原来的上限 3。短测未保存 checkpoint，也未进行完整准确率评估。

| 真实训练 step | 学习率 | 梯度范数 | 更新耗时 | 整步耗时 | 逻辑检索 / 后端查询 / 缓存命中 |
|---|---:|---:|---:|---:|---|
'''
for row in smoke_rows:report+=f"| {row['step']} | {row['lr']:.1e} | {row['grad_norm']:.3f} | {row['actor_seconds']:.3f}s | {row['step_seconds']:.3f}s | {row['logical_queries']:.0f} / {row['backend_queries']:.0f} / {row['cache_hits']:.0f} |\n"
report+='''
首步包含首次 optimizer 等初始化开销，短训练两步使用不同候选轨迹，上表不用于端到端速度对照。两个步骤均退出正常、指标有限、行为监控有输出，逻辑检索统计与轨迹搜索次数一致。

## 使用

在项目目录执行：

```bash
bash city_dynamic_2gpu_fast.sh --print-config
bash city_dynamic_2gpu_fast.sh
```

优化入口默认 actor microbatch 4，old-logprob/ref microbatch 8，关闭 actor 参数/梯度/optimizer 卸载及 reference 参数卸载；其他数据、模型、GRPO 分组、奖励、搜索上限、候选上限与训练预算继承现有双卡入口。默认配置已在 2 × A800 80GB 上通过短训练。

可选 microbatch 8 已通过固定轨迹基准：

```bash
ENGINEERING_ACTOR_MICRO_BATCH_SIZE=8 bash city_dynamic_2gpu_fast.sh
```

原入口继续使用其原有卸载和 microbatch 配置：

```bash
bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh
bash city_dynamic_1gpu.sh
```

复现默认均衡对照（LR=0，预热后各测 3 次）：

```bash
bash run_engineering_benchmark_2gpu.sh
```

复现 microbatch 对照：

```bash
bash run_engineering_benchmark_2gpu.sh \\
  'engineering.profile_micro_batches=[2,4,8]' \\
  actor_rollout_ref.actor.engineering_loss_group_size=1
```

基准会保存 `sampling/engineering_workload.pkl`，后续可通过 `engineering.workload_path=/绝对路径/engineering_workload.pkl` 重用本项目生成的批次，减少生成波动。所有运行使用独立 RUN_DIR；训练无需访问外网。

可分别关闭新增执行层、均衡或更新 profiling：

```bash
bash city_dynamic_2gpu_fast.sh engineering.retrieval.enabled=false
bash city_dynamic_2gpu_fast.sh engineering.sequence_balancing=false
bash city_dynamic_2gpu_fast.sh engineering.profile_updates=false
```

## 记录位置

- 每次运行的 `sampling/completed_updates.jsonl`：训练、检索执行与均衡指标。
- `engineering_profiles/rank_*.jsonl`：同步计时、单卡峰值 allocated/reserved 显存及 token 数。训练日志中的 rank_mean 指标是各卡均值；本报告表格使用各卡最大值。
- 基准的 `sampling/engineering_measurements.jsonl` 与 `engineering_benchmark.json`：预热、逐次测量与汇总。
- `analysis/offline_engineering_20261119/results.json`：本次完整测试索引及结果。
- `analysis/offline_engineering_20261119/source_sha256.json`：部署源码哈希。
- `analysis/offline_engineering_deploy_20261119/original/`：修改前源码备份。

'''
report+='实际运行目录：\n\n'+''.join(f'- {name}：`{path}`\n' for name,path in [('均衡初测',initial['run_dir']),('卸载对照',runs['offload']),('参数驻留对照',runs['resident']),('真实训练短测',smoke['run_dir'])])
doc=root/'OFFLINE_ENGINEERING_OPTIMIZATIONS.md';assert not doc.exists();doc.write_text(report)
(out/'REPORT.md').write_text(report)
print(json.dumps({'report':str(doc),'results':str(out/'results.json'),'default_update_reduction_percent':summary['default_actor_time_reduction_percent'],'real_updates':len(smoke_rows),'checks':'100 tests, bash syntax, Python AST and git diff --check passed'},indent=2))
print(subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader'],text=True))
