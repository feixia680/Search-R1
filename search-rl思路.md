# SearchAgent-RL：Search-R1 项目改造思路

> **项目定位**：基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 与 veRL，构建面向多跳知识问答的高效自适应搜索强化学习 Agent，重点体现 **两项算法能力（A+B）+ 两项工程能力（A+B）**。目标是让 Agent **搜索更聪明、训练更稳定、Rollout 更高效、训练系统更可靠**。
>
> **当前状态**：设计规划，以下方案和指标均为待实现/待实验验证的目标，不代表已完成。

## 一、算法能力

### 算法 A：Budget-aware Adaptive Search（自适应搜索策略）

**问题**：Agent 不知道何时搜索、如何改写查询、何时停止，容易产生重复或无价值检索。

**改造方案**：
- **多目标奖励**：以答案正确性为核心，加入新增有效证据奖励、搜索调用成本、重复查询惩罚、无效动作惩罚。
- **Evidence Gain**：依据多跳问答数据中的支持事实标注，衡量每轮搜索新增的证据覆盖率，奖励真正有贡献的搜索。
- **Adaptive Search / Stop**：保留 Search-R1 的 `<search>` / `<answer>` 动作空间，训练模型自主决定继续检索或结束回答，并学习 Query Reformulation。
- **Budget-aware Optimization**：优化准确率与工具调用数、检索 Token 成本的权衡，避免单纯少搜导致正确率下降。

**奖励函数（初版）**：

```text
R = R_answer + α·R_evidence_gain − β·R_search_cost
             − γ·R_repetition − δ·R_invalid
```

优先将多项奖励汇总为**轨迹级 Reward**，复用现有 GRPO 流程；如扩展为 **step-wise credit assignment**，需额外记录每个搜索动作的 Token 边界并实现动作级 return/advantage，不应仅将轨迹奖励称为逐步奖励。

### 算法 B：Stable GRPO Training（训练稳定性优化）

**问题**：最终答案奖励稀疏；同一问题的采样组可能全对/全错，优势信号弱；长轨迹带来训练不稳定。

**改造方案**：
- **Adaptive Sampling**：按题目难度、组内 Reward 方差、成功率选择更有训练价值的样本；与固定采样对比。
- **Group Advantage Optimization**：诊断全对/全错组、零方差组，研究采样分配与优势估计策略，避免盲目修改 GRPO 核心公式。
- **KL / Clip 调优**：跟踪 KL、clip fraction、策略熵、梯度范数、奖励分布，控制策略漂移。
- **对照实验**：Vanilla GRPO、改进 GRPO、REINFORCE；验证收敛速度、稳定性和最终问答准确率。

## 二、工程能力

### 工程 A：High-throughput Agent Rollout Engine（高吞吐轨迹生成）

**问题**：多轮 `Generate → Search → Observe` 存在同步检索等待、重复请求、变长轨迹低效等开销。

**改造方案**：
- **Async Retrieval**：支持异步 HTTP、并发限流、超时/重试与异常回退，确保不同轨迹之间可以并发调度。
- **Query Cache + Dedup**：批内及跨批的重复查询去重；缓存键包含 query、语料版本和检索配置，避免不同环境误命中。
- **Dynamic Batching**：按活跃轨迹与长度调度生成，降低 Padding 及已结束轨迹的无效计算。
- **Trajectory Tracing**：记录动作序列、检索结果、Token、搜索耗时、Rollout 耗时、Reward 和终止原因。

**注意**：同一轨迹的后续生成仍需依赖本轮检索结果；异步改造的价值必须用端到端 samples/s、GPU 利用率和检索等待占比实测，不能只用接口并发数代替。

### 工程 B：Reliable RL Training Infrastructure（可靠训练系统）

**问题**：多轮长轨迹易 OOM、训练中断后难以恢复，且缺乏完整诊断手段。

**改造方案**：
- **Checkpoint Resume**：正确恢复模型、优化器、学习率调度器、global step 和必要的训练/采样状态。
- **OOM Recovery**：对过长轨迹进行长度分桶和受控截断，合理调整 micro-batch / KV cache，记录恢复次数与性能代价。
- **Training Diagnostics**：W&B/日志追踪 Reward mean/std、零方差组比例、KL、梯度范数、有效动作率、搜索次数、Token、耗时。
- **Reproducible Training**：固定数据切分、随机种子、检索语料与索引版本、训练配置；保存 Git commit 与实验配置。

## 三、整体架构

```text
NQ / HotpotQA / 2Wiki / MuSiQue
                 │
                 ▼
      Search Policy（Qwen 3B）
       Think / Search / Answer
                 │
                 ▼
       High-throughput Rollout
  Async Retrieval + Cache + Batching
                 │
                 ▼
      Multi-objective Reward
  Answer + Evidence Gain - Search Cost
                 │
                 ▼
         Stable RL Training
 veRL + GRPO + Adaptive Sampling + KL
                 │
                 ▼
        Evaluation & Diagnostics
 EM/F1 + Evidence Recall + Calls + Throughput
```

## 四、基于现有仓库的改造位置

| 模块 / 文件 | 主要改动 |
| --- | --- |
| `search_r1/llm_agent/generation.py` | 搜索预算、异步检索调度、轨迹与动作级 Trace |
| `verl/trainer/main_ppo.py` | Multi-objective RewardManager |
| `verl/trainer/ppo/ray_trainer.py` | 轨迹元数据、采样与训练统计、监控与恢复接线 |
| `verl/trainer/ppo/core_algos.py` | 优势估计与算法对照（按实验需要修改） |
| `search_r1/search/retrieval_server.py` | 缓存、去重、超时、服务监控 |
| 新增 `evaluation/` | 数据处理、Baseline、消融及性能评测脚本 |
| 新增 `serving/`（可选） | FastAPI + vLLM 推理接口、Trace 查询、Docker 部署 |

> 原仓库已经支持 GRPO、PPO、多个检索器和多轮工具调用，这些不能单独算项目创新。真正的个人贡献应体现在可解释的策略改动、训练系统实现、性能分析及消融数据。

## 五、评测与消融实验

**数据**：NQ 用于基础复现；HotpotQA 作为多跳问答主实验（可利用支持事实标注）；2WikiMultiHopQA / MuSiQue 用于进一步泛化验证。统一语料、索引、数据切分与搜索预算，避免数据泄漏。

| 实验组 | 变化 | 主要验证 |
| --- | --- | --- |
| E0 | 原始 Search-R1 + GRPO | Baseline |
| E1 | + Search Cost | 搜索次数与答案准确率权衡 |
| E2 | + Evidence Gain | 支持事实覆盖与答案质量 |
| E3 | + 重复搜索惩罚 | 冗余查询比例 |
| E4 | E1+E2+E3 完整方法 | 算法整体收益 |
| E5 | E4 + Async/Cache/Batching | 端到端 Rollout 吞吐收益 |
| E6 | REINFORCE / 其他采样对照 | 训练稳定性与样本效率 |

**核心指标**：
- **算法 A**：EM、F1、Evidence Recall、平均搜索次数、答案正确率—搜索预算曲线。
- **算法 B**：Reward 方差、零优势组比例、KL、训练失败率、收敛步数。
- **工程 A**：Rollout samples/s、检索 P50/P95、等待占比、缓存命中率、GPU 利用率。
- **工程 B**：断点恢复正确性、OOM 频率、可复现性、训练有效吞吐与异常恢复耗时。

所有“提升 X%”需要在跑完对照和消融后填写，**不预先编造结果**。

## 六、实施优先级

1. **Baseline**：跑通 Qwen2.5-3B + GRPO + NQ/HotpotQA，记录完整 Trace 和指标。
2. **算法 A**：先做搜索成本、证据增益、重复检索惩罚，并完成单项消融。
3. **算法 B**：实现稳定性诊断与自适应采样，比较 GRPO/REINFORCE。
4. **工程 A**：改造异步检索 + 去重缓存 + 活跃轨迹调度，以真实吞吐数据验证。
5. **工程 B**：断点续训、OOM 处理、可观测性与复现脚本；补齐全量评测。

## 七、最终简历表达目标

**SearchAgent-RL：面向多跳问答的高效自适应搜索强化学习系统**

- **算法**：设计基于证据增益与搜索成本的多目标奖励，优化多轮 Search/Answer 策略；引入自适应采样与 GRPO 稳定性诊断，完成算法消融。
- **工程**：改造 Search-R1 的多轮 Tool-Calling Rollout，引入并发检索、查询缓存、动态调度；建设断点恢复、OOM 处理与训练可观测性能力。
- **评测**：在多跳问答数据集上统计 EM/F1、证据召回、调用成本与吞吐；用真实实验结果量化算法和工程收益。

**核心技术栈**：Python、PyTorch、veRL、GRPO、REINFORCE、Ray、vLLM、FSDP、FAISS、FastAPI、W&B。
