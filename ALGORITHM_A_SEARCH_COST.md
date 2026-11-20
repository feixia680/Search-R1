# Algorithm A: causal, evidence-aware search costs

## Reward and measurement
R = legacy_EM + alpha * unique_visible_gold_coverage
    - beta * dispatched_calls - after_full_lambda * calls_after_full
    - gamma * exact_normalized_duplicates - delta * invalid_actions

Defaults for NEW cost launchers: alpha=0.1, beta=0.005, after_full_lambda=0.01,
gamma=delta=0, require_final_answer=true, max_turns=2.
These are initial candidates, not demonstrated optimal hyperparameters.

Every dispatched call, including a failed request, pays beta. Budget-denied actions
are not dispatched and pay no resource cost. Infrastructure failures are not model
format errors. No extra penalty is applied merely because a search has no new gold
facts: useful bridge information and incomplete annotations remain possible.

Additional cost applies only when every reference supporting fact was already seen
in trusted policy-input document spans BEFORE the current action. First-visible
generation timestamps must be <= current action turn; the current search's result
and future input views cannot establish its prior state. Cumulative ever-seen
coverage is retained after context cropping. Full gold support is a proxy, not a
guarantee that the answer can be produced correctly; searching remains allowed.

Reference labels are reward metadata only. Inference still uses native search/answer
actions without a gold-dependent stopping rule. Invalid-annotation rows preserve
EM and base cost but skip evidence bonus and state cost, with eligibility logged.

## Controlled experiment arms
| COST_ARM | alpha | beta | after_full_lambda | require_final_answer |
|---|---:|---:|---:|---|
| evidence | 0.1 | 0 | 0 | true |
| uniform | 0.1 | 0.005 | 0 | true |
| state | 0.1 | 0.005 | 0.01 | true |

Each starts from the same Qwen2.5-3B base with seed 42, global batch 8, topk 3,
observation limit 1024, search budget 2, and the original 625-step schedule.
Compare saved step 500 on the original 200 questions using the existing protocol.
One/two-GPU scripts preserve global batch but hardware can introduce numerical
differences; use the same GPU layout across a matched comparison when possible.

Historical evidence training had require_final_answer=false. It cannot be used as
the sole matched control when both guard and costs change. Train the guarded
evidence arm for attribution, or explicitly report the combined intervention.
The guard puts empty/missing final answers below the worst valid-answer reward;
all NEW arms share it. Its floor now includes the maximum state cost.

## Saved-trajectory calibration (no generation/training)
See analysis/cost_design_20261113/CALIBRATION.md and calibration.json.
9,984 training trajectories were replayed. Positive-gain median is 0.5 (bonus
0.05 with alpha 0.1). Base beta=0.005 is below every observed positive-gain bonus,
while beta=0.05 would offset most such bonuses. This only checks reward scale:
it does not establish a new stopping policy, accuracy, or optimal coefficient.
The original 200 autonomous trajectories contain 24 calls after full evidence.
Their 16 missing final answers end in 15 budget-denied search actions and one
invalid final action, rather than recorded retrieval failures.

At budget 2, beta=.005 and lambda=.01 imply a conservative maximum resource
penalty .03. With alpha=.1, a valid EM=1 trajectory always outranks a valid
EM=0 trajectory under these candidate coefficients. This ordering is not a
guarantee of learned accuracy preservation under GRPO.

## Launch and config preview
Run from /g22538014wej/11x22f/Search-R1.
For current one-GPU server, preview all settings:
    COST_ARM=state bash run_grpo_cost_1gpu_with_bm25.sh --print-config

Run the matched arms in order, each with its own output directory:
    COST_ARM=evidence bash run_grpo_cost_1gpu_with_bm25.sh
    COST_ARM=uniform bash run_grpo_cost_1gpu_with_bm25.sh
    COST_ARM=state bash run_grpo_cost_1gpu_with_bm25.sh

On a two-GPU server use the corresponding *_2gpu*.sh scripts.
Override SEARCH_BETA / AFTER_FULL_LAMBDA explicitly for planned sensitivity tests.
Both launchers permit Hydra overrides; changing guard, budget, data or seed must
be recorded as a protocol change. The wrappers start/reuse BM25 like the existing
evidence wrappers, without adding a judge or a separate model-serving process.

Each run archives the actual modified reward source, resolved Hydra settings,
launcher, annotation audit and trace metrics. New metrics:
algorithm_a/after_full_search_calls, algorithm_a/after_full_cost,
algorithm_a/search_state_eligible.

## Validation scope
CPU tests cover future-visibility leakage, partial/untrusted facts, failed and
budget-denied calls, more than two calls, reward bounds, and RewardManager wiring
with real tokenizer visibility. Config checks cover both GPU layouts and all three
arms. The 1-GPU end-to-end smoke run passed:
analysis/cost_smoke_20261113_093925/SMOKE_REPORT.md.
It completed two GRPO updates (32 trajectories), 23 successful retrieval calls,
14 final-answer guard activations, checkpoint saving and original200 first8
validation. Every training reward component was independently recomputed;
sampled checkpoint parameters changed from the base model.
No fresh smoke trajectory searched after full evidence; the nonzero state-cost
branch remains covered by CPU visibility/RewardManager integration tests.
The 2-GPU layout has config checks only, not a GPU execution test.
No new evaluation dataset was introduced; no training is launched by calibration
or --print-config. Analysis files stay local and are not pushed.
