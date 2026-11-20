#!/usr/bin/env python3
"""GPU-side engineering validation without model training.

Checks real CUDA allocation, concurrent Rollout Trace writes, token-budget
packing, NCCL rank balancing (when launched with torchrun), and bounded OOM
recovery. It does not load a model or start a retriever.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from search_r1.runtime import (
    OOMRecoveryController,
    RecoveryState,
    RetryPolicy,
    RolloutTrace,
    RolloutTraceWriter,
    TokenPacker,
    TrajectoryCost,
    is_oom_error,
)


def init_distributed() -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    return world_size


def rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def run_trace_check(report_dir: Path) -> Dict[str, object]:
    path = report_dir / "rollout_trace.jsonl"
    writer = RolloutTraceWriter(path, fsync_every=0)
    errors: List[BaseException] = []

    def write_events(worker_id: int) -> None:
        try:
            trace = RolloutTrace("traj-%d" % worker_id, "question-%d" % worker_id, "group-%d" % worker_id)
            writer.write_many([
                trace.event("query_generation", input_tokens=32, output_tokens=8),
                trace.event("tool_call", query="engineering-%d" % worker_id, cache_hit=False),
                trace.event("answer", evidence_ids=("fact-%d" % worker_id,)),
            ])
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=write_events, args=(index,)) for index in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.close()
    if errors:
        raise errors[0]
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(records) != 96:
        raise AssertionError("concurrent trace writer lost records")
    if len({record["trajectory_id"] for record in records}) != 32:
        raise AssertionError("trace trajectory identities collided")
    if any(record["step_id"] not in (0, 1, 2) for record in records):
        raise AssertionError("trace step id is invalid")
    return {"events": len(records), "trajectories": 32, "path": str(path)}


def run_recovery_check() -> Dict[str, object]:
    attempts: List[tuple] = []

    def operation(state: RecoveryState) -> tuple:
        attempts.append((state.batch_size, state.max_new_tokens))
        if len(attempts) == 1:
            raise RuntimeError("CUDA out of memory (synthetic engineering test)")
        return attempts[-1]

    result = OOMRecoveryController(
        RetryPolicy(max_retries=2, shrink_factor=0.5, min_batch_size=1, min_max_new_tokens=64)
    ).run(operation, RecoveryState(batch_size=4, max_new_tokens=256))
    if result != (2, 128):
        raise AssertionError("OOM recovery did not shrink the retry state")
    return {"attempts": attempts, "recovered": True}


def run_real_oom_check(device: torch.device) -> Dict[str, object]:
    enabled = os.environ.get("ENGINEERING_REAL_OOM", "1") == "1"
    if not enabled:
        return {"enabled": False, "triggered": False}
    free_bytes, _ = torch.cuda.mem_get_info(device)
    request_bytes = int(free_bytes + 128 * 1024 * 1024)
    elements = max(1, request_bytes // 4)
    tensor: Optional[torch.Tensor] = None
    triggered = False
    try:
        tensor = torch.empty((elements,), dtype=torch.float32, device=device)
        tensor.zero_()
    except RuntimeError as error:
        if not is_oom_error(error):
            raise
        triggered = True
    finally:
        del tensor
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    if not triggered:
        raise AssertionError("real OOM probe did not trigger; recovery was not validated")
    return {"enabled": True, "triggered": True, "requested_bytes": request_bytes}


def run_token_balance_check(device: torch.device, world_size: int) -> Dict[str, object]:
    lengths = [137, 512, 271, 1024, 193, 431, 316, 777, 228, 609, 354, 488]
    items = [TrajectoryCost("traj-%d" % index, value, "group-%d" % (index // 2))
             for index, value in enumerate(lengths)]
    batches = TokenPacker(max_tokens=1800).pack(items)
    assignments = TokenPacker.balance(batches, world_size=world_size)
    local = assignments[rank()]
    real_tokens = sum(item.token_count for batch in local.batches for item in batch.items)
    padded_tokens = 0
    elapsed_start = time.perf_counter()
    for batch in local.batches:
        max_len = max(item.token_count for item in batch.items)
        tensor = torch.ones((len(batch.items), max_len), dtype=torch.float16, device=device)
        tensor.sum().item()
        padded_tokens += int(tensor.numel())
        del tensor
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - elapsed_start) * 1000.0
    loads = [torch.zeros(1, dtype=torch.float64, device=device) for _ in range(world_size)]
    local_load = torch.tensor([float(real_tokens)], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_gather(loads, local_load)
    else:
        loads[0].copy_(local_load)
    rank_loads = [float(value.item()) for value in loads]
    nonzero = [value for value in rank_loads if value > 0]
    imbalance = max(nonzero) / min(nonzero) if nonzero else 1.0
    padding_ratio = 1.0 - real_tokens / padded_tokens if padded_tokens else 0.0
    if padding_ratio < 0 or padding_ratio >= 1:
        raise AssertionError("invalid padding ratio")
    return {
        "world_size": world_size,
        "batches": len(batches),
        "local_rank": rank(),
        "rank_loads": rank_loads,
        "load_imbalance_ratio": imbalance,
        "real_tokens_local": real_tokens,
        "padded_tokens_local": padded_tokens,
        "padding_ratio_local": padding_ratio,
        "cuda_pack_ms_local": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    world_size = init_distributed()
    if not torch.cuda.is_available():
        if rank() == 0:
            (report_dir / "gpu_smoke.json").write_text(json.dumps({"gpu_available": False}) + "\n")
            print("GPU_SMOKE_SKIPPED: CUDA is unavailable")
        return
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    result = {
        "gpu_available": True,
        "device": torch.cuda.get_device_name(device),
        "world_size": world_size,
        "trace": run_trace_check(report_dir) if rank() == 0 else None,
        "recovery": run_recovery_check(),
        "token_balance": run_token_balance_check(device, world_size),
        "real_oom": run_real_oom_check(device),
    }
    barrier()
    if rank() == 0:
        (report_dir / "gpu_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
