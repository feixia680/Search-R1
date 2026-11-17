import json
import pytest
from search_r1.runtime.trace import RolloutTrace, RolloutTraceWriter, validate_trace_record
def test_trace_builder_and_jsonl_roundtrip(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    trace = RolloutTrace("traj-1", "q-1", "group-1")
    events = [trace.event("query_generation", input_tokens=10, output_tokens=4), trace.event("tool_call", query="capital of France", cache_hit=False), trace.event("answer", evidence_ids=("fact-1",), status="success")]
    with RolloutTraceWriter(path) as writer:
        assert writer.write_many(events) == 3
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["step_id"] for record in records] == [0, 1, 2]
    assert records[1]["query"] == "capital of France"
    assert records[2]["evidence_ids"] == ["fact-1"]
def test_trace_writer_rejects_missing_identity(tmp_path):
    with RolloutTraceWriter(tmp_path / "bad.jsonl") as writer:
        with pytest.raises(ValueError, match="missing fields"):
            writer.write({"event_type": "answer"})
def test_trace_validation_rejects_invalid_status():
    record = {"trace_version": 1, "trajectory_id": "t", "question_id": "q", "group_id": "g", "step_id": 0, "event_type": "answer", "timestamp_ns": 1, "status": "unknown"}
    with pytest.raises(ValueError, match="unsupported trace status"):
        validate_trace_record(record)
