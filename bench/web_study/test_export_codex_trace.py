import hashlib
import json
from pathlib import Path

from export_codex_trace import export_turn


def _write(path: Path, records: list[dict]) -> bytes:
    data = b"".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() + b"\n" for record in records)
    path.write_bytes(data)
    return data


def test_exports_original_bytes_and_excludes_other_turn(tmp_path):
    turn = "turn-a"
    records = [
        {"timestamp": "before", "type": "event_msg", "payload": {"turn_id": "turn-z"}},
        {"timestamp": "start", "type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}},
        {"timestamp": "middle", "type": "event_msg", "payload": {"type": "token_count", "info": {"turn_token_usage": {"total_tokens": 7}}}},
        {"timestamp": "end", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn, "last_agent_message": "failure-looking text"}},
        {"timestamp": "after", "type": "event_msg", "payload": {"turn_id": "turn-z"}},
    ]
    source = tmp_path / "rollout.jsonl"
    original = _write(source, records)
    paths = export_turn(source, tmp_path / "out", turn, source="host-rollout")

    expected = b"".join(original.splitlines(keepends=True)[1:4])
    assert paths["trace"].read_bytes() == expected
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["sha256"] == hashlib.sha256(expected).hexdigest()
    assert manifest["line_boundaries"] == {"first": 2, "last": 4, "record_count": 3}
    assert manifest["byte_boundaries"]["end_exclusive"] - manifest["byte_boundaries"]["start"] == len(expected)


def test_deduplicates_usage_by_response_and_maps_tool_sizes(tmp_path):
    turn = "turn-a"
    records = [
        {"timestamp": "t0", "type": "turn_context", "payload": {"type": "turn_context", "turn_id": turn, "model": "gpt-5.6-luna", "effort": "medium"}},
        {"timestamp": "t1", "type": "token_usage_record", "payload": {"type": "token_usage_record", "response_id": "r1", "usage": {"output_tokens": 3}, "turn_id": turn}},
        {"timestamp": "t2", "type": "token_usage_record", "payload": {"type": "token_usage_record", "response_id": "r1", "usage": {"output_tokens": 99}, "turn_token_usage": {"output_tokens": 99}, "turn_id": turn}},
        {"timestamp": "t3", "type": "response_item", "payload": {"type": "function_call", "call_id": "c1", "name": "web", "arguments": {"q": "x"}, "turn_id": turn}},
        {"timestamp": "t4", "type": "response_item", "payload": {"type": "function_call_output", "call_id": "c1", "output": {"answer": "ok"}, "turn_id": turn}},
        {"timestamp": "t5", "type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn}},
    ]
    source = tmp_path / "rollout.jsonl"
    _write(source, records)
    paths = export_turn(source, tmp_path / "out", turn)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["turn_context"] == {"model": "gpt-5.6-luna", "effort": "medium", "source": "line 1: payload"}
    assert len(metadata["token_usage_records"]) == 1
    assert metadata["token_usage_records"][0]["usage"]["output_tokens"] == 99
    assert metadata["token_usage_conflicts"][0]["response_id"] == "r1"
    response = next(item for item in metadata["tool_calls"] if item["kind"] == "response")
    assert response["result_json_bytes"] == len(b'{"answer":"ok"}')
    assert response["result_size_unit"] == "bytes (UTF-8 JSON)"
    request = next(item for item in metadata["tool_calls"] if item["kind"] == "request")
    assert request["arguments"] == {"q": "x"}


def test_missing_usage_and_response_are_explicitly_null_or_empty(tmp_path):
    turn = "turn-a"
    source = tmp_path / "rollout.jsonl"
    _write(source, [{"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}}])
    paths = export_turn(source, tmp_path / "out", turn)
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["turn_context"]["model"] is None
    assert metadata["token_usage_records"] == []
    assert metadata["cumulative_turn_token_usage"] is None
    assert metadata["time_boundaries"]["started"] is None
    assert metadata["completion_boundary"]["status"] == "missing_completion"


def test_audit_turn_reference_in_content_does_not_extend_export(tmp_path):
    turn = "turn-a"
    records = [
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": turn}},
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": turn}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "turn-b"}},
        {"type": "response_item", "payload": {"type": "message", "text": "audit mentions turn-a", "internal_chat_message_metadata_passthrough": {"turn_id": "turn-a"}}},
    ]
    source = tmp_path / "rollout.jsonl"
    _write(source, records)
    paths = export_turn(source, tmp_path / "out", turn)
    assert paths["trace"].read_text().count("turn-b") == 0


def test_usage_record_without_usage_is_not_zero_telemetry(tmp_path):
    source = tmp_path / "rollout.jsonl"
    _write(source, [{"type": "token_usage_record", "payload": {"turn_id": "t", "response_id": "r"}}])
    paths = export_turn(source, tmp_path / "out", "t")
    metadata = json.loads(paths["metadata"].read_text())
    assert metadata["token_usage_records"][0]["usage"] is None
    assert metadata["token_usage_status"]["usage_missing"] is True
    assert metadata["token_usage_status"]["records_missing_usage"] == [1]
