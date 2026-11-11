# Algorithm A implementation checks — 2026-11-11

This records implementation correctness and bounded training integration only.
It is not a benchmark or evidence of improved answer accuracy.

## Environment and isolation

- Server environment: `/g22538014wej/11x22f/envs/search1`.
- Original worktree: `/g22538014wej/11x22f/Search-R1`, unchanged `main` at `0c21297`.
- Development worktree: `/g22538014wej/11x22f/Search-R1-algorithm-a`.
- One idle NVIDIA A800-SXM4-80GB was available on the connected server.
- Read existing Qwen2.5-3B, train/test parquet, BM25 index and corpus by absolute
  path. No model or data downloads or environment package changes were needed.

## CPU checks

8 checks passed in the server environment, covering failure/duplicate/denied-call
accounting; success, empty and malformed results; HTTP status/timeout handling;
invalid coefficients and missing accounting; zero-warmup scheduling; original EM
compatibility, JSONL and reordered batch alignment; controlled GRPO advantages and
gradient updates; and scoring fixtures built from 3 existing HotpotQA labels.
The last fixtures use labels to construct answers and are not inference results.
`git diff --check` and `bash -n run_algorithm_a_smoke.sh` also passed.

## Actual model smoke

Artifacts: `algorithm_a_runs/smoke_validation_004/` on the server.

- Existing training sample IDs: 1501 and 2586, 2 stochastic rollouts each.
- One optimizer update: mini-batch 4, micro-batch 1, LR 1e-6, no warmup.
- Training reward beta .02 / gamma .05; validation reward remains pure EM.
- 4 training traces and 1 validation trace saved to `traces/*.jsonl`.
- 1 attempted search, successfully retrieved; 0 repeated queries.
- Training mean EM 0.0; training mean composite reward -0.005.
- Validation EM 0.0 on 1 sample. These tiny counts do not estimate accuracy.
- Within-group advantages reached +0.707 / -0.707; gradient norm 3.111.
- Reported step time 105.560 s, including generation 52.805 s, actor update
  18.842 s and checkpoint saving 26.471 s; initialization/evaluation are additional.
- Actor checkpoint: `checkpoints/actor/global_step_1/` under that run directory.
- Checkpoint config/tokenizer reload and safetensors reading passed.
- Compared an 8x128 slice of `model.layers.0.self_attn.q_proj.weight` against
  the original model: 1024/1024 values changed, max absolute difference
  `1.000240445137024e-06`.
- The verifier wrote `verification.json`; run exited with code 0.
- No NaN or OOM was observed in this bounded run.

## Problems found and corrected

1. `smoke_validation_001`: the retriever started, but system curl failed because
   the Java/Conda library path conflicted with libffi. Readiness now uses Python.
2. `smoke_validation_002`: Ray rejected the long Unix socket path. The launcher
   now allocates an independent short `/tmp/sra.*` runtime directory.
3. `smoke_validation_003`: forward/backward and optimizer invocation succeeded,
   but saved weights were unchanged. The inherited zero-warmup scheduler set
   the first optimizer LR to 0. This was fixed only in the algorithm-a worktree,
   with a regression test, before running smoke_validation_004.

The sampled incorrect/no-answer trajectories also demonstrate a limitation of the
requested initial reward: with equal EM=0, a no-search failure can outrank a
searching failure. Keep this visible in diagnostics; no extra validity reward was
added without an explicit experiment design.

## Next experiment

Wait for baseline results, then compare EM only, search cost, and search cost plus
repetition using identical initial weights and budgets. Tune on validation; evaluate
held-out EM/F1 and calls separately from composite reward. Evidence-gain reward,
full resume support and full training are not part of this implementation.
