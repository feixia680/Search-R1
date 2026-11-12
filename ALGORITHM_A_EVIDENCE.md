# Algorithm A: supporting facts and visible-evidence verification

This extension implements evidence data preparation, trusted visibility tracing,
trajectory-level evidence rewards and CPU verification. It does not establish an
accuracy improvement and does not launch model inference or training.

## Data contract

`evaluation/prepare_algorithm_a_evidence.py` reads the existing processed/raw
HotpotQA data and attaches versioned references under `extra_info.evidence`.
References include the original question ID, sentence IDs, titles, complete
supporting sentences, validity status and explicit errors. Index mapping is checked
against the normalized question. Duplicate supporting-fact IDs are deduplicated;
ambiguous context titles, missing sentences and empty evidence are flagged.

The main `train.parquet` and `test.parquet` outputs preserve all original questions,
their order and their prompt strings. `*_all_audited.parquet` retain the same rows;
`*_evidence_eligible.parquet` provide separately named evidence-only subsets.
No original dataset is overwritten. Input/output SHA256 and anomaly lists are saved.
Reference labels are not passed into the generation worker.

For the current files, 2/5000 training and 3/5000 validation rows have ambiguous or
invalid evidence annotations. The original training/validation sample pools remain
unchanged. The default reward policy errors on invalid annotations when alpha is
nonzero. Explicit `search_reward.invalid_evidence_policy=skip` disables only the
evidence bonus for known `invalid_annotation` rows; answer scoring and other reward
terms remain active. Missing references and question-mapping errors still fail.
`algorithm_a/evidence_eligible` reports the eligible fraction; evidence coverage
metrics exclude ineligible rows.

## Visibility contract

`visibility.py` tracks external-token origins alongside the rolling input. It
mirrors the existing stable padding permutation and context crop without modifying
policy token IDs. Each event records the untruncated observation, document-body
boundaries, tokens before/after observation truncation and actual input views
captured immediately before generation. Both left/right padding are supported.
A fast tokenizer with character offsets is required when tracing is enabled.

Evidence matching requires the full supporting sentence in both a visible trusted
document-body span and the decoded input-token chunk. A matching document title is
also required. This excludes title-only hits, generated `<information>` strings,
query text, partial sentences and partial Unicode tokens. Previously observed facts
remain credited even if a later generation's context drops them. A fact is credited
only once across rounds. Failed, empty and budget-denied searches earn no evidence.

Each trajectory travels with its row in `DataProto.non_tensor_batch`; reference and
trajectory metadata remain aligned through repetition and batch balancing.

## Reward contract

The existing `RewardManager` now supports:

`R = legacy_EM + alpha * coverage - beta * attempts - gamma * duplicates - delta * invalid_actions`

All coefficients default to zero. `coverage` is the sum of per-round novel fact
fractions, equal to final unique reference coverage. This is a trajectory-level
reward, not per-action credit assignment. Invalid actions count unparseable actions
and empty search/answer contents; infrastructure errors are not classified as
model-format errors. Validation remains pure legacy EM and still logs actual-answer
EM independently. No judge is used.

The old cost-only objective has a confirmed loophole: with both outcomes wrong, a
malformed no-search response scores 0 while a legitimate search can score -beta.
The optional `search_reward.require_final_answer=true` places missing/empty-final-
answer trajectories below the worst valid-answer reward within the configured
search budget. The adjustment is logged as `failure_guard_cost`. It also prevents
positive evidence reward from rescuing a trajectory without a final answer.
This guard defaults to false to preserve historical reward compatibility; enabling
it changes the objective and should be kept identical across future matched arms.
It does not solve every form of under-search or make a wrong direct guess correct.

## CPU-only verification

Use the existing search1 environment, this worktree's PYTHONPATH, and disable CUDA:

```bash
export PYTHONPATH=/g22538014wej/11x22f/Search-R1-algorithm-a
export CUDA_VISIBLE_DEVICES=""
export ALGORITHM_A_MODEL=/g22538014wej/11x22f/Search-R1/models/Qwen2.5-3B
export ALGORITHM_A_DATA=/g22538014wej/11x22f/Search-R1/data/processed/hotpotqa/train.parquet
python -m unittest discover -s tests/algorithm_a -v
python evaluation/prepare_algorithm_a_evidence.py \
  --base-root /g22538014wej/11x22f/Search-R1 --out-dir algorithm_a_data/evidence
python evaluation/audit_algorithm_a_evidence.py \
  --base-root /g22538014wej/11x22f/Search-R1 \
  --annotated-data algorithm_a_data/evidence/test.parquet \
  --out-dir algorithm_a_runs/evidence_validation
```

The audit replays previously saved step500 text and exactly reconstructs saved
observation formatting. It is not fresh inference or recovery of an unsaved model.
The 200 replays cover 469 facts, with 280 visible hits and 54 fully covered questions,
matching the prior evidence audit. Six EM/coverage categories are exported in
`CASE_REVIEW.md`; `em0` means an exact-match mismatch, not a semantic verdict.

Manual review flagged validation sample index 589: its
question asks for a Beethoven work published in 1801, the retrieved Violin Sonata
No.5 text explicitly states 1801, but the reference answer is Symphony No.7 and its
support text dates it to 1811–1812. This is a suspected reference conflict requiring
human review. No label was rewritten or row silently removed.
