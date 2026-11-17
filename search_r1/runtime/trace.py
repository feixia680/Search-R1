"""Structured, append-only rollout traces."""
from __future__ import annotations
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union
REQUIRED_TRACE_FIELDS = {"trace_version", "trajectory_id", "question_id", "group_id", "step_id", "event_type", "timestamp_ns", "status"}
def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)
@dataclass(frozen=True)
class TraceEvent:
    trajectory_id: str
    question_id: str
    group_id: str
    step_id: int
    event_type: str
    status: str = "success"
    timestamp_ns: int = field(default_factory=time.time_ns)
    latency_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    query: Optional[str] = None
    evidence_ids: Tuple[str, ...] = ()
    cache_hit: Optional[bool] = None
    retry_count: int = 0
    error_type: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    trace_version: int = 1
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["evidence_ids"] = list(self.evidence_ids)
        return _json_safe(result)
class RolloutTraceWriter:
    """Thread-safe JSONL writer with flush and optional fsync boundaries."""
    def __init__(self, path: Union[str, os.PathLike], fsync_every: int = 0):
        if fsync_every < 0:
            raise ValueError("fsync_every must be non-negative")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync_every = fsync_every
        self._lock = threading.Lock()
        self._count = 0
        self._closed = False
        self._handle = self.path.open("a", encoding="utf-8")
    @property
    def count(self) -> int:
        return self._count
    def write(self, event: Union[TraceEvent, Mapping[str, Any]]) -> Dict[str, Any]:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        record = event.to_dict() if isinstance(event, TraceEvent) else _json_safe(dict(event))
        validate_trace_record(record)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._lock:
            if self._closed:
                raise RuntimeError("trace writer is closed")
            self._handle.write(line + "\n")
            self._handle.flush()
            self._count += 1
            if self.fsync_every and self._count % self.fsync_every == 0:
                os.fsync(self._handle.fileno())
        return record
    def write_many(self, events: Iterable[Union[TraceEvent, Mapping[str, Any]]]) -> int:
        written = 0
        for event in events:
            self.write(event)
            written += 1
        return written
    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._handle.flush()
                self._handle.close()
                self._closed = True
    def __enter__(self) -> "RolloutTraceWriter":
        return self
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
class RolloutTrace:
    """Convenience builder that keeps trajectory identity consistent."""
    def __init__(self, trajectory_id: str, question_id: str, group_id: str):
        self.trajectory_id = trajectory_id
        self.question_id = question_id
        self.group_id = group_id
        self._step_id = 0
    def event(self, event_type: str, status: str = "success", **kwargs: Any) -> TraceEvent:
        event = TraceEvent(trajectory_id=self.trajectory_id, question_id=self.question_id, group_id=self.group_id, step_id=self._step_id, event_type=event_type, status=status, **kwargs)
        self._step_id += 1
        return event
def validate_trace_record(record: Mapping[str, Any]) -> None:
    missing = REQUIRED_TRACE_FIELDS.difference(record)
    if missing:
        raise ValueError(f"trace record missing fields: {sorted(missing)}")
    if not str(record["trajectory_id"]).strip():
        raise ValueError("trajectory_id must be non-empty")
    if not str(record["question_id"]).strip():
        raise ValueError("question_id must be non-empty")
    if not isinstance(record["step_id"], int) or record["step_id"] < 0:
        raise ValueError("step_id must be a non-negative integer")
    if not isinstance(record["timestamp_ns"], int) or record["timestamp_ns"] <= 0:
        raise ValueError("timestamp_ns must be a positive integer")
    if record["status"] not in {"success", "retry", "failed", "cancelled"}:
        raise ValueError(f"unsupported trace status: {record['status']}")
