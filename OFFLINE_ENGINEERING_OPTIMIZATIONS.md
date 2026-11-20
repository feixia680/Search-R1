# 离线检索与双卡训练工程优化（2026-11-19）

已部署到 `/g22538014wej/cityins1122/cityins`，硬件为 2 × A800 80GB，模型为 Qwen2.5-3B。新增检索执行层、损失分组兼容的 microbatch 扩大、硬件配置入口和测量记录。veRL token 均衡属于项目已有能力，本次增加开关、分卡指标和真实对照。

## 测得的收益

同一批 16 条轨迹、11749 tokens，学习率为 0，实际执行 forward/backward/optimizer。每种配置先预热，再交替测量 3 次。两组对照的所有 batch tensor、非 tensor 数据和元信息均验证相同，完整工作负载 SHA256 为 `d11a592725029535a7a0de0d8f6cf2b8c82b975043da02022ecd04eef2ed6111`。

| 参数卸载 | 全局 actor microbatch | token 均衡 | 更新中位耗时 | 单卡最大 allocated / reserved |
|---|---:|---|---:|---:|
| 开启 | 2 | 关闭 | 5.309s | 29.52 / 36.36 GiB |
| 开启 | 2 | 开启 | 5.728s | 29.52 / 35.96 GiB |
| 开启 | 4 | 开启 | 4.895s | 29.52 / 37.16 GiB |
| 开启 | 8 | 开启 | 4.059s | 29.52 / 37.52 GiB |
| 关闭 | 2 | 关闭 | 3.221s | 31.82 / 37.91 GiB |
| 关闭 | 2 | 开启 | 2.902s | 31.81 / 37.02 GiB |
| 关闭 | 4 | 开启 | 2.127s | 31.81 / 37.54 GiB |
| 关闭 | 8 | 开启 | 1.635s | 31.81 / 37.14 GiB |

默认优化配置（关闭参数卸载、microbatch 4）相对原来的开启卸载、microbatch 2、启用均衡，actor 更新中位耗时从 5.728s 降到 2.127s，减少 62.9%。microbatch 8 的基准耗时为 1.635s，作为可选设置。这里是更新环节的测量，生成、检索和评分仍占用时间；未测得同轨迹完整训练的端到端加速比例。

token 差距 `(max-min)/mean` 从 7.575% 降到 0.017%，每卡 tokens 从 6097/5652 变为 5875/5874。均衡开关的更新时间在不同短测中有波动，不能据此宣称稳定加速。此次主要收益来自参数驻留和更大的 microbatch。

## BM25 检索

使用历史 TRAIN 轨迹中的 32 个不同查询，各重复一次，共 64 次逻辑搜索。逐项验证文档、顺序和分数与原始 BM25 一致。

| 模式 | 实际后端查询 | HTTP 请求 | 本次回放耗时 |
|---|---:|---:|---:|
| 原始调用 | 64 | 4 | 223.431ms |
| 冷缓存 + 批内去重 | 32 | 4 | 111.278ms |
| 内存缓存 | 0 | 0 | 0.356ms |
| 跨客户端磁盘缓存 | 0 | 0 | 3.604ms |

这组重复查询回放用于验证缓存与去重。实际训练中的命中率取决于查询重复程度，不代表上述收益会覆盖所有训练步骤。模型每次发起的搜索仍各自记录、计费和评分，缓存不会减少逻辑搜索次数。

缓存键包含精确查询、topk、服务 URL 和索引版本；索引自动版本使用本地索引及 corpus 的路径、大小和 mtime。默认持久缓存为 `data/cache/retrieval_runtime.sqlite3`，另有 1024 项内存缓存。超时、连接错误及 HTTP 408/429/5xx 最多重试 2 次；其他 HTTP 和格式错误不重试。失败结果不缓存，磁盘缓存故障回退实时检索，并记录错误。训练和评估均沿用本地 BM25，无新增外部服务依赖。

## 数值与训练验证

增大 microbatch 时，简单地对所有 token 求均值会改变原有损失权重。新增 `engineering_loss_group_size` 保留原有局部损失分组：原双卡全局 microbatch 2 对应每卡 1 条，因此优化入口固定组大小为 1。扩大模型前向批量时，policy、entropy 和 KL 仍按这些组计算后平均。double 精度的损失和梯度对照测试通过。GPU 混合精度下，batch 形状变化仍会产生浮点差异，本次未验证长训练准确率等价。

100 项 CPU/集成回归测试通过。共完成 40 次真实 FSDP 基准更新（LR=0），以及默认优化入口的 2 次真实学习率更新。后者保持分层抽题、动态筛选、证据奖励与行为监控，测试候选批上限设为 2；正式入口仍沿用原来的上限 3。短测未保存 checkpoint，也未进行完整准确率评估。

| 真实训练 step | 学习率 | 梯度范数 | 更新耗时 | 整步耗时 | 逻辑检索 / 后端查询 / 缓存命中 |
|---|---:|---:|---:|---:|---|
| 1 | 1.0e-06 | 8.900 | 5.261s | 56.235s | 16 / 14 / 2 |
| 2 | 1.0e-06 | 6.661 | 2.511s | 50.963s | 19 / 18 / 1 |

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
bash run_engineering_benchmark_2gpu.sh \
  'engineering.profile_micro_batches=[2,4,8]' \
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

实际运行目录：

- 均衡初测：`/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_profile_2gpu_20261119_091328`
- 卸载对照：`/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_offload_2gpu_20261119_092100`
- 参数驻留对照：`/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_resident_replay_2gpu_20261119_092829`
- 真实训练短测：`/g22538014wej/cityins1122/cityins/algorithm_a_runs/engineering_fast_smoke_2gpu_20261119_093454`
