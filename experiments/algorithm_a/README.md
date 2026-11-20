# Evidence-only candidate protocol

Status: prepared for review; no training launched. Formal hardware: 2 GPUs.

## Main comparison

Reuse the saved pure-EM global_step_500 model. Start the new evidence-reward arm
from the same Qwen2.5-3B Base. Global train/val batch 8, two rollouts per question,
mini-batch 8, global micro-batch 2, LR1e-6, warmup ratio 0.285, KL 0.001, topk 3,
response 256, observation 1024 and maximum 2 searches follow the archived baseline.

Candidate alpha=0.1, beta=gamma=delta=0. Alpha is a starting choice, not a tuned
or validated optimum. The final-answer guard stays off for historical compatibility.
Monitor missing answers closely because evidence-only reward can also credit an
unfinished trajectory. Known invalid annotations skip only evidence credit; all
5000 original training/validation questions and their order remain in the data files.

The baseline driver did not record/set a shuffle seed, so its exact training order
cannot be reconstructed. The new driver seed is 42. The historical comparison is
therefore not a strict single-variable randomized ablation. GPU topology will match
when two GPUs are available. Evaluate both models with the same frozen evaluator
and question manifest; historical online scores should not be mixed with standalone
re-evaluation scores.

## Scheduler and checkpoint budget

Keep trainer.total_training_steps=625, retaining 178 warmup steps. Setting 501 would
change warmup to 142 and confound the first 500 updates. Compare saved step 500 in both
arms. With the inherited loop, executing this entire config would perform 624 actual
updates; no 500-update early-stop mechanism has been added here. This config has not
been launched. The old final624-update weights were not saved, so their reported
validation score cannot be assigned to step500.

## Is a maximum of two searches too low?

Two may restrict exploration for multi-hop questions. However, the saved baseline
used one search on every audited question despite an available second search.
First test the evidence-reward arm with the same maximum 2. Then evaluate both saved
models at budgets 1/2/3/4; no baseline retraining is necessary for this inference sweep.
Report EM/F1, actual calls, limit-hit ratio, missing answers, novel facts and repeated
queries. If budget 2 is frequently reached and later calls add evidence, consider a
higher-budget training extension and report its additional budget explicitly.

Budget sweeps must use a fixed context capacity large enough for the largest
budget in both models. With start 512, response 256 and observation 1024, assuming the initial prompt fits within 512 tokens, the upper
bound on the prompt before final answer generation is 512+B*(256+1024):
B2=3072, B3=4352, B4=5632. Allow another 256 tokens for the final generation, and
check engine limits and memory. These are upper bounds, not measured utilization.
Changing max_turns alone can otherwise retrieve evidence which is immediately
cropped. Re-evaluate all sweep points under the common expanded context protocol.

Review evidence_only.yaml and protocol.json before a future launch. The existing
baseline is reused; this preparation has not started GPU inference or training.
