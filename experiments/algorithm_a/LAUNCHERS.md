# 证据奖励训练脚本

项目目录：`/g22538014wej/11x22f/Search-R1`。

- `train_grpo_evidence_1gpu.sh`：默认 GPU 0。
- `train_grpo_evidence_2gpu.sh`：默认 GPU 0、1。

两份脚本保留相同的全局 batch=8、mini-batch=8、micro-batch=2、每题2条轨迹、
最多2次搜索、256 response tokens、1024 observation tokens，以及原625步学习率计划。
单卡版本主要用于单卡运行；正式历史baseline对照计划采用双卡。

奖励为 `EM + alpha * 新增可见支持事实覆盖率`。`alpha` 默认0.1，是待验证的起始值；
搜索成本、重复、无效动作惩罚均为0。终止防护保持关闭，与旧baseline设置一致。
训练/验证仍保留原5000题，已知异常标注仅跳过证据加分，验证奖励保持纯EM。

## 只查看配置

```bash
bash train_grpo_evidence_1gpu.sh --print-config
bash train_grpo_evidence_2gpu.sh --print-config
```

该模式由 Hydra 打印配置后退出，不检查GPU占用、不准备数据、不访问检索器、不启动训练。

## 后续手动运行

先在一个终端启动现有 BM25 服务：

```bash
bash start_bm25_local.sh
```

服务就绪后，在另一个终端选择一个训练脚本：

```bash
bash train_grpo_evidence_1gpu.sh
```

或者使用双卡：

```bash
bash train_grpo_evidence_2gpu.sh
```

可显式选择GPU及alpha，例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 EVIDENCE_ALPHA=0.1 bash train_grpo_evidence_2gpu.sh
```

其他可配置环境变量：`BASE_ROOT`、`BASE_MODEL`、`CONDA_ENV`、`EVIDENCE_DATA_DIR`、
`RETRIEVER_URL`、`TRAIN_SEED`、`EXPERIMENT_NAME`、`RUN_DIR`。
默认检索地址为 `http://127.0.0.1:8000/retrieve`，每次返回3个文档。

脚本会检查所选GPU是否空闲、数据题目/顺序及文件哈希是否一致、BM25是否可用。
支持事实数据缺失时会先在CPU上生成。每次运行保存独立的日志、Hydra配置、轨迹和检查点，
拒绝覆盖已有运行目录，使用独立Ray实例，不清理其他作业。

输出目录：`algorithm_a_runs/evidence_1gpu_*` 或 `algorithm_a_runs/evidence_2gpu_*`。
模型位于对应目录的 `checkpoints/actor/global_step_500` 等检查点中。

## 对照口径

保留 `trainer.total_training_steps=625`，以保持178步warmup；当前循环实际执行624次更新。
与已保存baseline主要比较双方的step500。修改为501会改变warmup计划，不能视作只缩短运行。

旧baseline未记录driver shuffle随机状态，新脚本固定seed42，历史训练顺序无法保证完全一致。
这些脚本还未进行新的GPU训练验证；Bash语法与Hydra配置预览检查不等于训练成功或效果提升。

## Launcher structure update (2026-11-13)

Both evidence launchers now use the same direct, multiline `python3 -m verl.trainer.main_ppo` command layout and parameter order as their baseline launchers. The previous inline GPU, dataset, and retriever checks are preserved in `evaluation/check_evidence_training_inputs.py`. Keep that file with the launchers.

Both launchers retain the existing evidence experiment settings (global batch 8, observation 1024, schedule 625), including the 1-GPU variant; this structure change does not switch it to the smaller 1-GPU baseline settings. `--print-config`, environment overrides, Hydra overrides, BM25 wrappers, output isolation, and trace logging are retained.

Validation: Bash syntax, Python syntax, before/after Hydra configuration equality (excluding timestamped output paths), and both BM25 wrapper configuration previews passed. Preview option ordering was also checked with a Hydra override. No training was started for this update. The earlier interrupted smoke remains incomplete.
