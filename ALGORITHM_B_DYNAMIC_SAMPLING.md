# Evidence-aware bounded dynamic sampling

## Files and scope

This implementation is separate from the historical training entrypoint. All
files introduced here are new; existing main_ppo.py, ray_trainer.py, dp_actor.py,
reward code, configs and evidence/cost launchers are not modified.

- search_r1/dynamic_sampling/selection.py: dependency-free group selector.
- search_r1/dynamic_sampling/batching.py: variable-width rollout merge.
- verl/trainer/ppo/dynamic_sampling_trainer.py: candidate refill and GRPO update.
- verl/trainer/main_dynamic_sampling.py: isolated FSDP training entrypoint.
- verl/workers/dynamic_sampling_worker.py: scoped actor and collective guards.
- verl/workers/actor/dynamic_sampling_actor.py: loss check before backward.
- verl/trainer/config/dynamic_sampling.yaml: new Hydra config extending ppo_trainer.
- train_grpo_dynamic_sampling_{1,2}gpu.sh and corresponding BM25 wrappers.

No GPU training is launched by installation, CPU tests or --print-config.
No training effect or runtime speedup is established by these tests.

## Selection protocol

For the default global batch of 8 questions and two rollouts per question:

1. Draw the next question batch from the existing shuffled training stream.
2. Generate complete search/answer trajectories with the unchanged policy.
3. Score every candidate using EM + alpha * unique visible support coverage.
   Alpha defaults to 0.1. All search penalties are zero; the final-answer reward
   guard remains false, matching the historical evidence objective.
4. Keep groups whose float32 total reward range exceeds 1e-8. Keep both rollouts,
   including all-wrong groups with different evidence coverage.
5. Refill with new candidate question batches until 8 informative groups exist
   or 3 candidate batches have been generated. At the default sizes this means
   at most 24 question groups / 48 candidate trajectories per update.
6. At the cap, randomly fill remaining slots from finite zero-difference groups
   already generated during this update. The update always has the same size.
7. Questions are unique inside the selected update. A repeated question across
   candidate batches is not admitted twice or merged into a four-rollout group.
8. No optimizer update happens while collecting candidates; buffers are local
   to one update and are never replayed after a policy change.

This is a budget-limited adaptation of DAPO-style current-group filtering, not
full DAPO and not historical difficulty weighting. Unlike uncapped filtering,
some selected groups may remain zero-difference after the budget is exhausted.
No extra answer-validity filter is added. Missing-answer groups can have different
evidence rewards; candidate and selected missing-answer rates must be reviewed.

Uniform control uses the SAME new trainer and numerical guards, but keeps the
first question batch with no filtering/refill. Both arms use identical rewards.

## Data stream, groups and tensors

The question stream restarts with a new shuffle when exhausted, because refill
consumes more examples than an ordinary update. Thus total_epochs remains 1 as
a compatibility setting, while total_training_steps limits optimizer updates.
Track total candidate exposures and unique questions, not epoch count alone.

The historical stopping convention is retained: total_training_steps=625 means
updates 1..624 and final validation labeled 625. Smoke override 3 means 2 updates.
Default validation remains the original 200-question selection. Test fixtures
use synthetic trajectories and do not add an empirical evaluation dataset.

Selected groups from different rounds can have different prompt and response
widths. Prompts are left-padded, responses right-padded, combined masks/positions
are split at the original prompt boundary and padded separately. Non-tensor
references, trajectories and per-row generation statistics remain aligned.
Rewards are independently recomputed on selected, padded/balanced batches and
checked against their candidate scores before the update.

Discarded candidates do not receive actor/reference log-prob recomputation or
backpropagation. Their generation and retrieval consumption still counts.

## Metrics and audit

- traces/candidates.jsonl: every generated candidate; split=candidates.
- traces/train.jsonl: only selected trajectories; split=train.
- traces/validation.jsonl: existing pure-EM validation, no sampling/filtering.
- sampling/sampling_decisions.jsonl: group IDs, candidate rewards, selection and
  fallback decisions, all generated tokens, observed retrieval-token counts,
  actual backend search calls, missing-answer rates and generation time.
- sampling/completed_updates.jsonl: only successfully completed updates.
- numerical_diagnostics/rank_*.jsonl and sampling/failure.json: failures.

Generated tokens are counted using response-region info_mask. Observation tokens
are counted after per-observation truncation and are not claimed to equal FLOPs
or the full context processing cost. Counters include discarded candidates.

Metrics under sampling/ include candidate_groups, selected_informative_groups,
selected_effective_group_fraction, zero_fallback_groups, generation_multiplier,
candidate_generated_tokens, candidate_observation_tokens and retrieval calls.
Compare equal total rollout/token budgets or report wall time and consumed
trajectories explicitly; fewer updates alone does not prove faster training.

## Numerical safeguards

The new worker installs guards only on its own actor instance. The new actor
checks finite policy loss before backward. Forward outputs, input log-probs,
advantages, non-empty microbatch loss masks and clipped gradient norm are checked.
An all-rank failure flag makes every rank stop before proceeding to the next
phase when any rank detects a numerical failure; no rank silently skips an update.

Supported scope: one node, FSDP, GRPO, use_kl_loss=true, state_masking=true,
fixed-size microbatches, sequence/tensor parallel size 1, rollout.n=1.
Dynamic microbatch schedules are rejected because ranks may perform different
numbers of collective guard calls. GPU OOM/deadlock recovery, complete resume,
Clip-Higher and token-level-loss changes are not implemented here.
Numerical guards stop propagation; they do not establish the root cause of the
previous step-588 failure.

## Usage

Run from /g22538014wej/11x22f/Search-R1.

Preview without GPU use:

```bash
bash run_grpo_dynamic_sampling_1gpu_with_bm25.sh --print-config
bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh --print-config
```

Dynamic sampling (starts training, requires the stated number of free GPUs):

```bash
bash run_grpo_dynamic_sampling_1gpu_with_bm25.sh
bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh
```

Matched uniform control:

```bash
SAMPLING_ENABLED=false bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh
```

Change the candidate budget explicitly:

```bash
MAX_CANDIDATE_BATCHES=2 bash run_grpo_dynamic_sampling_2gpu_with_bm25.sh
```

Launchers inherit the existing input/GPU/BM25 preflight. They save a copy of the
actual reward, generation and new sampling source alongside resolved config.
After restoring GPU availability, run a separate isolated two-update smoke test
before a full experiment; CPU fixtures do not verify real vLLM/FSDP execution.

## CPU checks

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export PYTHONPATH=$PWD
export CUDA_VISIBLE_DEVICES=""
export ALGORITHM_A_MODEL=$PWD/models/Qwen2.5-3B
python -m unittest discover -s tests/dynamic_sampling -v
```

Tests include pool budget/fallback, all-wrong evidence differences, duplicate
question handling, metadata/token/position alignment, real-tokenizer visible
evidence reward preservation, simulated full update/validation/save wiring,
nonfinite-loss/gradient blocking, and two-process CPU Gloo failure consensus.
CPU Gloo consensus is not a GPU NCCL/FSDP execution test.
