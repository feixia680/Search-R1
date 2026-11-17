import pytest
from search_r1.runtime.recovery import OOMRecoveryController, OOMRecoveryExhausted, RecoveryState, RetryPolicy, is_oom_error
def test_oom_recovery_shrinks_and_retries():
    seen = []
    def operation(state):
        seen.append((state.batch_size, state.max_new_tokens))
        if len(seen) < 3:
            raise RuntimeError("CUDA out of memory")
        return state.batch_size, state.max_new_tokens
    result = OOMRecoveryController(RetryPolicy(max_retries=3)).run(operation, RecoveryState(batch_size=8, max_new_tokens=512))
    assert result == (2, 128)
    assert seen == [(8, 512), (4, 256), (2, 128)]
def test_non_oom_errors_are_not_swallowed():
    def operation(_):
        raise ValueError("bad input")
    with pytest.raises(ValueError, match="bad input"):
        OOMRecoveryController().run(operation, RecoveryState(2, 256))
def test_oom_exhaustion_is_explicit():
    def operation(_):
        raise RuntimeError("CUDA out of memory")
    with pytest.raises(OOMRecoveryExhausted):
        OOMRecoveryController(RetryPolicy(max_retries=1)).run(operation, RecoveryState(2, 256))
def test_oom_detection_is_conservative():
    assert is_oom_error(RuntimeError("CUDA out of memory"))
    assert is_oom_error(RuntimeError("CUDA OOM"))
    assert not is_oom_error(RuntimeError("network timeout"))
