"""CPU-side runtime utilities for traceable and fault-tolerant rollouts."""
from .recovery import OOMRecoveryController, OOMRecoveryExhausted, RecoveryState, RetryPolicy, is_oom_error
from .token_packer import PackedBatch, RankAssignment, TokenPacker, TrajectoryCost
from .trace import REQUIRED_TRACE_FIELDS, RolloutTrace, RolloutTraceWriter, TraceEvent, validate_trace_record
__all__ = ["OOMRecoveryController", "OOMRecoveryExhausted", "RecoveryState", "RetryPolicy", "is_oom_error", "PackedBatch", "RankAssignment", "TokenPacker", "TrajectoryCost", "REQUIRED_TRACE_FIELDS", "RolloutTrace", "RolloutTraceWriter", "TraceEvent", "validate_trace_record"]
