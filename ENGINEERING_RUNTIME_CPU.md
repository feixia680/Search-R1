# CPU-side rollout runtime foundation

This branch adds the CPU-testable foundation for the engineering runtime. It does not change the active training entrypoint.

- `search_r1/runtime/trace.py`: versioned rollout events, trajectory identity checks, thread-safe JSONL writing, flush/fsync boundaries.
- `search_r1/runtime/token_packer.py`: deterministic token-budget packing, whole-GRPO-group preservation, and greedy rank-load balancing.
- `search_r1/runtime/recovery.py`: bounded OOM retry policy with batch/token shrinking; the policy is independent of CUDA so it can be unit-tested before GPU integration.

The GPU adapter should call these modules from the rollout worker. It must keep `group_id`, reward, masks, and log-probability tensors aligned when batches are repacked. OOM handling should requeue only the failed micro-batch and emit a failed/retry trace event.

CPU validation:

```bash
pytest -q tests/runtime
python -m compileall -q search_r1/runtime
```
