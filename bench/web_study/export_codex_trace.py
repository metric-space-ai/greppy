"""Lossless export and bounded telemetry extraction for one Codex rollout turn.

The JSONL export is deliberately copied as bytes.  Metadata is an index over
that export and must never be used to reconstruct the trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


TURN_KEYS = {"turn_id", "turnId"}


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else {}


def _record_turn_id(record: dict[str, Any]) -> str | None:
    """Read only the record-owned turn id, never IDs mentioned in content."""
    for container in (record, _payload(record)):
        for key in TURN_KEYS:
            value = container.get(key)
            if isinstance(value, str):
                return value
    return None


def _record_kind(record: dict[str, Any]) -> str:
    payload = _payload(record)
    value = payload.get("type", record.get("type", ""))
    return str(value)


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _path_get(value: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _records(lines: list[bytes]) -> list[tuple[int, bytes, dict[str, Any] | None]]:
    result = []
    for index, raw in enumerate(lines, 1):
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        result.append((index, raw, parsed if isinstance(parsed, dict) else None))
    return result


def _extract_metadata(selected: list[tuple[int, bytes, dict[str, Any] | None]], source: str, turn_id: str) -> dict[str, Any]:
    contexts: list[dict[str, Any]] = []
    usage_by_response: dict[str, dict[str, Any]] = {}
    usage_without_response: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    thread_totals: list[dict[str, Any]] = []
    usage_conflicts: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    times: list[dict[str, Any]] = []

    for line_no, _raw, record in selected:
        if record is None:
            continue
        timestamp = record.get("timestamp")
        if timestamp is not None:
            times.append({"value": timestamp, "source": f"line {line_no}: timestamp"})
        payload = _payload(record)
        item_type = _record_kind(record)
        if item_type == "turn_context":
            contexts.append({"value": payload, "source": f"line {line_no}: payload"})
        if item_type == "token_usage_record":
            response_id = _path_get(payload, "response_id", "responseId")
            usage = payload.get("usage")
            usage = usage if isinstance(usage, dict) else None
            entry = {"response_id": response_id, "usage": usage, "source_line": line_no}
            if isinstance(response_id, str):
                prior = usage_by_response.get(response_id)
                if prior is None:
                    usage_by_response[response_id] = entry
                elif prior.get("usage") != usage:
                    usage_conflicts.append({"response_id": response_id, "first": prior, "conflict": entry})
                    usage_by_response[response_id] = entry
            else:
                usage_without_response.append(entry)
            if isinstance(payload.get("turn_token_usage"), dict):
                cumulative.append({"value": payload["turn_token_usage"], "source": f"line {line_no}: payload.turn_token_usage"})
            if isinstance(payload.get("thread_token_usage"), dict):
                thread_totals.append({"value": payload["thread_token_usage"], "source": f"line {line_no}: payload.thread_token_usage"})

        if item_type in {"custom_tool_call", "function_call", "tool_request"}:
                call_id = _path_get(payload, "call_id", "callId", "id")
                request = {"kind": "request", "call_id": call_id, "source_line": line_no}
                if "name" in payload:
                    request["name"] = payload["name"]
                arguments = _path_get(payload, "arguments", "input")
                if arguments is not None:
                    request["arguments"] = arguments
                    request["arguments_json_bytes"] = _json_bytes(arguments)
                tools.append(request)
        elif item_type in {"custom_tool_call_output", "function_call_output", "tool_response"}:
                call_id = _path_get(payload, "call_id", "callId")
                result = _path_get(payload, "output", "result", "content")
                response = {"kind": "response", "call_id": call_id, "source_line": line_no}
                if result is not None:
                    response["result"] = result
                    response["result_json_bytes"] = _json_bytes(result)
                    response["result_size_unit"] = "bytes (UTF-8 JSON)"
                tools.append(response)

    context_value = contexts[0]["value"] if contexts else None
    model = effort = None
    context_source = None
    if isinstance(context_value, dict):
        model = _path_get(context_value, "model", "model_name", "modelName")
        effort = _path_get(context_value, "effort", "reasoning_effort", "reasoningEffort")
        context_source = contexts[0]["source"]
    return {
        "schema_version": "codex-trace-export.v1",
        "turn_id": turn_id,
        "source": source,
        "turn_context": {"model": model, "effort": effort, "source": context_source},
        "token_usage_records": list(usage_by_response.values()),
        "token_usage_conflicts": usage_conflicts,
        "token_usage_records_without_response_id": usage_without_response,
        "token_usage_status": {
            "usage_present": any(entry["usage"] is not None for entry in list(usage_by_response.values()) + usage_without_response),
            "usage_missing": not any(entry["usage"] is not None for entry in list(usage_by_response.values()) + usage_without_response),
            "records_missing_usage": [entry["source_line"] for entry in list(usage_by_response.values()) + usage_without_response if entry["usage"] is None],

            "response_id_missing_count": len(usage_without_response),
            "source": "token_usage_record.payload; no inferred or summed values",
        },
        "cumulative_turn_token_usage": cumulative[-1] if cumulative else None,
        "thread_total_token_usage": thread_totals[-1] if thread_totals else None,
        "tool_calls": tools,
        "tool_response_status": {
            "request_count": sum(item["kind"] == "request" for item in tools),
            "response_count": sum(item["kind"] == "response" for item in tools),
            "unmatched_request_call_ids": sorted({item["call_id"] for item in tools if item["kind"] == "request" and isinstance(item["call_id"], str)} - {item["call_id"] for item in tools if item["kind"] == "response" and isinstance(item["call_id"], str)}),
            "source": "call_id equality in extracted records",
        },
        "time_boundaries": {
            "started": times[0] if times else None,
            "ended": times[-1] if times else None,
            "source": "rollout JSONL record timestamp; no derived wall-clock values",
        },
    }


def export_turn(input_path: str | Path, output_dir: str | Path, turn_id: str, source: str | None = None) -> dict[str, Path]:
    """Export exactly the bounded original records for *turn_id*.

    A turn must be explicitly selected by ID.  Records between the first and
    last matching record are retained to include untagged token-count events.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    raw = input_path.read_bytes()
    lines = raw.splitlines(keepends=True)
    records = _records(lines)
    owned = [(i, record) for i, (_line_no, _raw, record) in enumerate(records) if record and _record_turn_id(record) == turn_id]
    starts = [i for i, record in owned if _record_kind(record) in {"task_started", "turn_context"}]
    completes = [i for i, record in owned if _record_kind(record) == "task_complete"]
    if not owned:
        raise ValueError(f"turn_id not found in rollout: {turn_id}")
    start = min(starts) if starts else min(i for i, _record in owned)
    end = max(completes) if completes else max(i for i, _record in owned)
    selected = records[start : end + 1]
    exported = b"".join(raw_line for _line_no, raw_line, _record in selected)
    byte_start = sum(len(line) for line in lines[:start])
    byte_end = byte_start + len(exported)
    digest = hashlib.sha256(exported).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"turn-{turn_id}"
    trace_path = output_dir / f"{stem}.jsonl"
    metadata_path = output_dir / f"{stem}.metadata.json"
    manifest_path = output_dir / f"{stem}.manifest.json"
    trace_path.write_bytes(exported)
    metadata = _extract_metadata(selected, source or str(input_path), turn_id)
    metadata["line_boundaries"] = {"first": start + 1, "last": end + 1, "record_count": len(selected)}
    metadata["byte_boundaries"] = {"start": byte_start, "end_exclusive": byte_end}
    metadata["completion_boundary"] = {
        "task_complete_present": any(record and _record_kind(record) == "task_complete" and _record_turn_id(record) == turn_id for _line, _raw, record in selected),
        "status": "complete" if any(record and _record_kind(record) == "task_complete" and _record_turn_id(record) == turn_id for _line, _raw, record in selected) else "missing_completion",
        "source": "record-owned event_msg.payload.type/task_complete",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "codex-trace-manifest.v1",
        "turn_id": turn_id,
        "source": source or str(input_path),
        "export": str(trace_path),
        "metadata": str(metadata_path),
        "sha256": digest,
        "byte_length": len(exported),
        "line_boundaries": metadata["line_boundaries"],
        "byte_boundaries": metadata["byte_boundaries"],
        "selection": "contiguous original JSONL bytes bounded by record-owned task_started/turn_context through task_complete; untagged records inside the boundary are retained",
        "limits": ["success is not inferred from agent text", "missing telemetry remains null or explicitly listed as unidentifiable"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"trace": trace_path, "metadata": metadata_path, "manifest": manifest_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source")
    args = parser.parse_args()
    paths = export_turn(args.input, args.output_dir, args.turn_id, args.source)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
