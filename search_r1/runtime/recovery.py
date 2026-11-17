"""CPU-testable local recovery policy for GPU OOM integration."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Optional, TypeVar
T = TypeVar("T")
class OOMRecoveryExhausted(RuntimeError):
    """Raised when an OOM cannot be recovered within the retry budget."""
@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    shrink_factor: float = 0.5
    min_batch_size: int = 1
    min_max_new_tokens: int = 128
    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if not 0 < self.shrink_factor < 1:
            raise ValueError("shrink_factor must be in (0, 1)")
        if self.min_batch_size <= 0 or self.min_max_new_tokens <= 0:
            raise ValueError("minimum recovery values must be positive")
@dataclass(frozen=True)
class RecoveryState:
    batch_size: int
    max_new_tokens: int
    attempts: int = 0
    def shrink(self, policy: RetryPolicy) -> "RecoveryState":
        return replace(self, batch_size=max(policy.min_batch_size, int(self.batch_size * policy.shrink_factor)), max_new_tokens=max(policy.min_max_new_tokens, int(self.max_new_tokens * policy.shrink_factor)), attempts=self.attempts + 1)
def is_oom_error(error: BaseException) -> bool:
    name = type(error).__name__.lower()
    message = str(error).lower()
    return "outofmemory" in name or "out of memory" in message or "cuda oom" in message
class OOMRecoveryController:
    def __init__(self, policy: Optional[RetryPolicy] = None):
        self.policy = policy or RetryPolicy()
    def run(self, operation: Callable[[RecoveryState], T], state: RecoveryState, on_retry: Optional[Callable[[RecoveryState, BaseException], None]] = None) -> T:
        current = state
        while True:
            try:
                return operation(current)
            except BaseException as error:
                if not is_oom_error(error):
                    raise
                if current.attempts >= self.policy.max_retries:
                    raise OOMRecoveryExhausted(f"OOM recovery exhausted after {current.attempts} retries") from error
                current = current.shrink(self.policy)
                if on_retry is not None:
                    on_retry(current, error)
