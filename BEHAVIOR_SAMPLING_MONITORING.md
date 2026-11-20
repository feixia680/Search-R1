# Offline behavior strata and training diagnostics

`bash city_dynamic_1gpu.sh` now enables both features by default. The same
configuration is supported by the two-GPU dynamic-sampling launcher. Existing
reward terms, model initialization, search budget and GRPO objective are unchanged.

## Offline labels and prompt selection

The frozen manifest is `algorithm_a_data/evidence/behavior_strata.json`. It was
built from the **training** trajectories of the historical evidence run:

```bash
source scripts/search_environment.sh
python evaluation/build_behavior_strata.py \
  --dataset algorithm_a_data/evidence/train.parquet \
  --traces algorithm_a_runs/evidence_2gpu_a0.1_s42_20261112_174015/traces/train.jsonl \
  --output algorithm_a_data/evidence/behavior_strata.json
```

The builder refuses to overwrite an existing manifest. For a new reference run,
write a new filename and select it with `BEHAVIOR_STRATA_PATH`.

Each training ID receives one role:

* `first_sufficient`: every observed eligible first retrieval covered all annotated facts.
* `first_insufficient`: incomplete mean first-retrieval coverage, with observed later
  evidence gain or a partially covered first retrieval.
* `query_rewrite`: no later gain observed, and at least half the eligible first
  retrievals had zero coverage, or at least half the traces repeated a query.
* `unobserved`: no eligible first retrieval was available in the reference traces.

These are **historical query/retriever-specific proxies**, not proofs of a
question's required reasoning depth. Only training data are used. Trace reference
IDs must match the training annotations; training parquet SHA-256 is checked on
every launch. No role, gold fact or reference answer is inserted into prompts or
added to the reward. The original parquet files are not rewritten.

At each candidate refill, 25% of question slots are uniformly sampled from all
training IDs. The remaining slots use role weights `1:2:1` for
`first_sufficient:first_insufficient:query_rewrite`, with deterministic integer
rounding. Each candidate batch is unique, and later refill batches avoid IDs
already generated in the same update. Exhausted role pools fall back to uniform
sampling and record the quota shortfall. Uniform slots also expose unobserved IDs.

The existing reward-difference selector still runs **after** candidate generation.
Thus these are candidate-prompt quotas, not constraints forcing a model action or
guarantees of the final selected group composition. Each question's GRPO group
remains intact. Both requested and realized roles are recorded.

## Switches and controls

```bash
# Default: strata + existing dynamic filtering + monitoring
bash city_dynamic_1gpu.sh

# Uniform prompt selection, keep monitoring and existing dynamic filtering
BEHAVIOR_SAMPLING_ENABLED=false bash city_dynamic_1gpu.sh

# Historical prompt selection and no behavior monitoring
BEHAVIOR_SAMPLING_ENABLED=false BEHAVIOR_MONITOR_ENABLED=false bash city_dynamic_1gpu.sh

# Uniform-control trainer: disable reward-difference filtering as well
SAMPLING_ENABLED=false BEHAVIOR_SAMPLING_ENABLED=false bash city_dynamic_1gpu.sh

# Preview config / validate inputs and BM25 without training
bash city_dynamic_1gpu.sh --print-config
bash city_dynamic_1gpu.sh --check-only
```

Hydra overrides remain available, for example
`behavior_sampling.uniform_fraction=0.5`. Model initialization still follows
`BASE_MODEL`; adding these features does not implicitly load an evidence checkpoint.

## Candidate and selected diagnostics

Each successful update writes, under the run's `sampling/` directory:

* `behavior_strata_snapshot.json`: frozen manifest used for this run.
* `behavior_metrics.jsonl`: per-update candidate/selected totals and per-role
  metrics, plus requested/realized prompt draws and explicit fallback counts.
* `behavior_health.json`: rolling health summary and recent observations.

Console and `completed_updates.jsonl` metrics use `behavior/candidate/...`,
`behavior/selected/...` and `behavior/health/...` prefixes. They include:

* actual second-search rate and mean retrieval attempts;
* second-search new-evidence rate (denominator: actual second searches);
* missing-final-answer and blocked-search rates;
* eligible evidence coverage and incomplete-evidence fraction;
* zero-reward-difference group fraction and within-group reward standard deviation;
* identical tool-action-sequence group fraction, which describes actions rather
  than query text or semantic diversity.

Alerts use a 10-update window after 100 warmup updates, on candidate trajectories
from `first_insufficient`. Rates are pooled by trajectory count; evidence-related
rates use evidence-eligible counts. Low search use alone is insufficient: the
underuse alert also requires incomplete current evidence. Easy questions are not
required to use two searches.

Default alerts are `SECOND_SEARCH_UNDERUSE_WITH_INCOMPLETE_EVIDENCE`,
`MISSING_FINAL_ANSWER` and `SEARCH_BUDGET_TRUNCATION`. Thresholds are configurable
under `behavior_monitor`. They are diagnostics, not measures of final model quality.

Default behavior is **warning only**. To stop after a successful update when an
alert fires:

```bash
bash city_dynamic_1gpu.sh behavior_monitor.stop_on_alert=true
```

The completed update and diagnostic records are written before that exception.
The optional stop does not create an extra checkpoint; normal checkpoint cadence
still applies. This legacy trainer does not implement optimizer-state resume.

## Validation scope

CPU checks cover deterministic quotas, uniqueness across refills, shortage
fallback, metadata/prompt separation, dataset-hash rejection, validation-trace
rejection, candidate-versus-selected denominators and rolling alerts. The historical
dynamic-sampling replay first reported second-search underuse at step143.
New GPU smoke runs are stored separately from existing trained models/results.
