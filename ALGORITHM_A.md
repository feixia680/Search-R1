# Algorithm A: search cost and repeated-query reward

Worktree: `/g22538014wej/11x22f/Search-R1-algorithm-a`, branch `algorithm-a`.
The original `Search-R1` worktree and its baseline training scripts are unchanged.

## Reward contract

`R = original_EM - search_reward.beta * retrieval_attempts - search_reward.gamma * duplicate_queries`

Both coefficients default to **0.0** in `verl/trainer/config/ppo_trainer.yaml`.
The original `qa_em.compute_score_em` scorer is retained verbatim, including its
prompt/example-dependent answer extraction. `em` is that baseline-compatible
score. `final_answer_em` independently evaluates the final model answer recorded
from its action, making discrepancies with the legacy scorer visible.
Validation always uses pure EM, even when training penalties are enabled.

- `search_actions`: parsed search actions emitted by active trajectories,
  including a search emitted after the final allowed retrieval round.
- `retrieval_attempts`: queries dispatched to the HTTP retriever. Failed requests
  also incur cost. No retry is performed, so one dispatch is charged once.
- `retrieval_successes`: requests with valid HTTP/schema/document results.
  A valid empty result counts as a successful call, not as useful evidence.
- `retrieval_failures`: timeout, non-2xx, invalid JSON/schema/documents.
- `duplicate_queries`: repeated **attempted** queries within one trajectory,
  after Unicode NFKC normalization, case folding, and whitespace normalization.
  Punctuation and word order are preserved; this is not semantic similarity.
- A denied search incurs no call cost or repetition cost, but remains in events.
  Failed calls remain in repetition history.

Trajectory dictionaries travel in `DataProto.non_tensor_batch`, so row reordering
and batch balancing preserve the association with the question. JSONL records
include trajectory/sample IDs, actions, query text, execution status, document IDs,
final answer, termination, EM, each cost term, total reward, step and split.
Per-step metrics use the `algorithm_a/` prefix. Trace writing is opt-in via
`search_reward.trace_dir` (default null).

This is an **outcome / trajectory-level reward**, assigned at the last response
position and consumed by the existing GRPO advantage calculation. It does not
implement per-action credit assignment or evidence-gain reward.

## CPU checks in the server environment

```bash
cd /g22538014wej/11x22f/Search-R1-algorithm-a
source /opt/conda/etc/profile.d/conda.sh
conda activate /g22538014wej/11x22f/envs/search1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ALGORITHM_A_MODEL=/g22538014wej/11x22f/Search-R1/models/Qwen2.5-3B
export ALGORITHM_A_DATA=/g22538014wej/11x22f/Search-R1/data/processed/hotpotqa/train.parquet
python -m unittest discover -s tests/algorithm_a -v
```

The tests use controlled actions, a real tokenizer and a few existing sample labels
for reward correctness. Those constructed correct answers are **not** model EM
results. A controlled GRPO/gradient test checks that a search-cost difference
produces nonzero, opposite-sign within-group advantages and a parameter update.

## Bounded GPU smoke

```bash
bash run_algorithm_a_smoke.sh
# EM-only compatibility run:
SEARCH_BETA=0 SEARCH_GAMMA=0 bash run_algorithm_a_smoke.sh
```

The launcher refuses a GPU with >=500 MiB allocated. It creates its own BM25
service on port 8011, its own Ray instance/short temporary path, and a unique
`algorithm_a_runs/smoke_TIMESTAMP` directory. It never calls `ray stop` or kills
unrelated processes. `PYTHONPATH` explicitly points to this worktree so editable
installation of the original repository does not select the wrong source.
Readiness uses Python requests, since the Java/Conda library path breaks the
system curl on this server (`libp11-kit` / libffi mismatch).

The run samples 2 existing training questions (dataset sampling uses random_state
42), produces 2 rollouts per question, and executes **one optimizer update** with
mini-batch 4 / micro-batch 1. It then saves a checkpoint and validates 1 question.
The upstream loop increments its counter before checking the stop condition;
`total_training_steps=2` intentionally bounds this script to executing step 1.
The branch fixes an upstream scheduler bug: zero warmup previously initialized
step-0 LR to zero, so a single optimizer call did not change weights. Zero warmup
now uses the configured LR immediately; positive-warmup behavior is unchanged.
A regression check covers this behavior. This fix is separate from reward changes.
Other smoke settings: 128 response tokens per round, 512 observation tokens,
2 search rounds, lr 1e-6 without warmup, beta .02 / gamma .05. These nonzero
coefficients are test settings, not tuned algorithm recommendations.

```bash
python evaluation/verify_algorithm_a_smoke.py \
  --run-dir algorithm_a_runs/smoke_TIMESTAMP \
  --base-model /g22538014wej/11x22f/Search-R1/models/Qwen2.5-3B
```

The verifier checks trace/accounting, pure-EM validation, checkpoint config and
tokenizer reloading, and actual changes in a saved attention-weight slice. A
successful smoke proves training integration, not improved answer quality.
The inherited checkpoint saver exports actor weights; it does not establish
full optimizer/scheduler/data-state resume support.

## Formal experiments after the baseline finishes

Start all primary comparisons from the same base checkpoint and use the same
training data, rollout budget, search limits and held-out evaluator:
EM only; EM + search cost; EM + search cost + repetition penalty.
Use validation to tune coefficients, then report EM/F1, calls and repetition on
the untouched test split. Evidence-gain reward requires supporting-fact metadata
which is absent from the current processed parquet; it remains future work.

## Reward interpretation limitation

When every answer in a group is wrong, this initial cost-only formulation can rank
an invalid/no-search trajectory above a wrong/searching trajectory. The smoke run
exposes this behavior; it is not evidence of better task performance. Track invalid
and missing-answer cases during formal experiments before deciding whether an
explicit validity term is needed. No additional reward term is silently enabled.

The inherited `train_grpo_*` / `run_grpo_*` launchers still refer to the original
baseline directory. Use the new `run_algorithm_a_smoke.sh` for this branch;
formal algorithm-A training launchers should be configured separately.
See `ALGORITHM_A_RESULTS.md` for the executed checks and observed limitations.
