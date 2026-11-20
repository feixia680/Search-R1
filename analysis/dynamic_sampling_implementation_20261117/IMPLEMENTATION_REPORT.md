# 动态采样实现与核验

服务器：10.130.10.166:30382。项目：/g22538014wej/11x22f/Search-R1。分支：main。

## 实现

新增20个项目文件，包括独立采样内核、变长轨迹合并、训练入口、数值保护、配置、测试和1GPU/2GPU启动脚本。
默认每批8题、每题2条尝试，最多3批候选；优先保留总奖励不同的整组，不足时随机补入本轮零差异组。
双答错但证据不同的组保留。不新增缺答案过滤，仍记录候选和入选轨迹的缺答案率。
奖励保持 EM + 0.1*新增可见支持事实覆盖率，成本项全部为0，缺答案奖励保护保持false。
补题期间不更新模型，不跨策略版本复用候选。只有入选轨迹计算old/ref log-prob并反向传播。
新增loss/梯度数值检查、多rank异常共识、候选与入选轨迹分开保存和所有候选生成消耗统计。

## 核验

- 21项CPU测试通过，包括整组筛选、预算回退、题目去重、变长padding与元数据对齐。
- 真实tokenizer的可见证据奖励在合并前后保持一致。
- 模拟worker完成两轮训练回路、优势计算、验证与checkpoint调用核验。
- 非有限loss在backward前被阻断，非有限梯度在optimizer.step前被阻断。
- 双进程CPU Gloo验证：一个rank异常，两边均停止。
- 1GPU/2GPU × 动态采样/均匀对照共4份配置与4个shell脚本检查通过。
- 280个原文件哈希均未变化；本次实现全部位于新文件。

## 入口

采样：search_r1/dynamic_sampling/selection.py
训练器：verl/trainer/ppo/dynamic_sampling_trainer.py
入口：verl/trainer/main_dynamic_sampling.py
数值保护：verl/workers/dynamic_sampling_worker.py、verl/workers/actor/dynamic_sampling_actor.py
详细说明：ALGORITHM_B_DYNAMIC_SAMPLING.md

有GPU且完成GPU smoke后，可运行：

    bash run_grpo_dynamic_sampling_1gpu_with_bm25.sh
    bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh

均匀对照：

    SAMPLING_ENABLED=false bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh

配置预览不启动训练：

    bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh --print-config

## 验证边界

本服务器当前未检测到GPU，没有启动真实vLLM/FSDP训练或评测。GPU smoke仍待执行。
CPU训练回路使用模拟worker，checkpoint只验证调用；Gloo测试不等于NCCL/FSDP验证。
有上限回退的筛选不是完整DAPO，也不是历史权重优先抽题。
数值保护是fail-stop，不回滚此前mini-batch更新，不支持完整训练恢复或OOM恢复。
沿用历史停止约定：total_training_steps=625对应624次更新，最后验证标签625。
不能据这些代码核验声明问答准确率或训练吞吐提高。
