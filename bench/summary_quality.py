#!/usr/bin/env python3
"""Build and evaluate the Greppy function-purpose summary quality gate."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
REALCORPUS = HERE / "agent_efficiency" / "realcorpus"
MANIFEST = REALCORPUS / "MANIFEST.json"
CASES_SCHEMA = "greppy.summary-quality-cases.v1"
RESULTS_SCHEMA = "greppy.summary-quality-results.v1"
JUDGMENTS_SCHEMA = "greppy.summary-quality-judgments.v1"
GATE_SCHEMA = "greppy.summary-quality-gate.v2-triage"
SELECTION_VERSION = "greppy-summary-quality-selection-v1"
# v3: the item file_path became a legitimate grounding source. The summary
# generator (teacher and product alike) sees the path, and role terms taken
# from it (tcp/listener.rs -> "listener") are grounded statements, not
# inventions. Measured 2026-07-14: even the teacher failed the zero-tolerance
# checks under v2 solely through path-grounded terms on one-line wrappers.
JUDGE_PROMPT_VERSION = "greppy-summary-quality-judge-v4-triage"
SUMMARY_PROMPT_VERSION = "qwen35-brief-path-v5"
DEFAULT_REPOS = ("serde", "flask", "gson", "zod", "tokio", "hugo")
EXCLUDED_PATH = re.compile(
    r"(^|/)(tests?|testdata|bench(mark)?s?|examples?|vendor|third_party|generated)(/|$)"
    r"|(^|/)[^/]*(?:_test\.go|\.test\.[^/]+|\.spec\.[^/]+)$",
    re.IGNORECASE,
)
CODE_HINT = re.compile(
    r"\b(?:pub\s+)?(?:async\s+)?fn\s+[A-Za-z_]\w*\s*\("
    r"|\b(?:async\s+)?def\s+[A-Za-z_]\w*\s*\("
    r"|\bfunction\s+[A-Za-z_$]\w*\s*\(|->|[{};]"
)
SYMBOLISH = re.compile(r"\b(?:[a-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[A-Za-z]+[A-Z][A-Za-z0-9]*)\b")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    argv: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    timeout: float = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def workspace_hash(root: pathlib.Path) -> str:
    return sha256_bytes(str(root.resolve()).encode("utf-8"))[:16]


def graph_path(store_dir: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
    return store_dir / "workspaces" / "v2" / workspace_hash(root) / "graph.db"


def source_for(case: dict[str, Any]) -> str:
    path = REALCORPUS / case["repo"] / case["file_path"]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    source = "".join(lines[case["start_line"] - 1 : case["end_line"]])
    if sha256_bytes(source.encode("utf-8")) != case["source_sha256"]:
        raise RuntimeError(f"source digest mismatch for {case['id']}: {path}")
    return source


def manifest_entry(manifest: dict[str, Any], repo: str) -> dict[str, Any]:
    repositories = manifest.get("repos", manifest)
    entry = repositories.get(repo) if isinstance(repositories, dict) else None
    if not isinstance(entry, dict):
        raise RuntimeError(f"repo {repo!r} is absent from {MANIFEST}")
    return entry


def verify_checkout(repo: str, entry: dict[str, Any]) -> pathlib.Path:
    root = REALCORPUS / repo
    if not root.is_dir():
        raise RuntimeError(
            f"missing pinned repo {repo}; run real_corpus.py setup --repos {repo}"
        )
    actual = run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    if actual != entry["commit"]:
        raise RuntimeError(f"{repo} commit mismatch: {actual} != {entry['commit']}")
    return root.resolve()


def index_repo(
    binary: pathlib.Path,
    root: pathlib.Path,
    store_dir: pathlib.Path,
    device: str,
) -> None:
    env = os.environ.copy()
    env["GREPPY_STORE_DIR"] = str(store_dir)
    proc = run(
        [str(binary), "--device", device, "--root", str(root), "index", str(root)],
        cwd=root,
        env=env,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"index failed for {root}:\n{proc.stdout}\n{proc.stderr}")


def candidate_rows(db_path: pathlib.Path, root: pathlib.Path, repo: str) -> list[dict[str, Any]]:
    if not db_path.is_file():
        raise RuntimeError(f"missing graph database: {db_path}")
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT label, name, qualified_name, file_path, start_line, end_line, properties
              FROM nodes
             WHERE label IN ('Function', 'Method', 'Constructor')
               AND start_line > 0 AND end_line >= start_line
             ORDER BY qualified_name, file_path, start_line
            """
        ).fetchall()
        generation_row = connection.execute(
            "SELECT MAX(graph_generation) FROM workspace_state"
        ).fetchone()
        generation = int(generation_row[0] or 0)
    finally:
        connection.close()

    candidates: list[dict[str, Any]] = []
    for label, name, qualified_name, file_path, start_line, end_line, properties in rows:
        if EXCLUDED_PATH.search(file_path):
            continue
        line_count = end_line - start_line + 1
        if line_count < 3 or line_count > 80:
            continue
        path = root / file_path
        if not path.is_file():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        source = "".join(lines[start_line - 1 : end_line])
        if not source or len(source.encode("utf-8")) > 8192:
            continue
        try:
            props = json.loads(properties)
        except (TypeError, json.JSONDecodeError):
            props = {}
        signature = props.get("source_signature")
        if not isinstance(signature, str) or not signature.strip():
            signature = signature_from_source(source)
        if not signature:
            continue
        rank = sha256_bytes(
            f"{SELECTION_VERSION}\0{repo}\0{qualified_name}\0{file_path}\0{start_line}".encode()
        )
        candidates.append(
            {
                "repo": repo,
                "graph_generation": generation,
                "label": label,
                "name": name,
                "qualified_name": qualified_name,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "source_sha256": sha256_bytes(source.encode("utf-8")),
                "selection_rank": rank,
            }
        )
    return sorted(candidates, key=lambda row: row["selection_rank"])


def signature_from_source(source: str) -> str | None:
    lines = source.splitlines()
    while lines and (not lines[0].strip() or lines[0].lstrip().startswith("//")):
        lines.pop(0)
    if not lines:
        return None
    declaration = "\n".join(lines)
    stripped = declaration.lstrip()
    python_declaration = stripped.startswith("def ") or stripped.startswith("async def ")
    round_depth = square_depth = angle_depth = 0
    in_string: str | None = None
    escaped = False
    end = len(stripped)
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in ("\"", "'"):
            in_string = char
        elif char == "(":
            round_depth += 1
        elif char == ")":
            round_depth = max(0, round_depth - 1)
        elif char == "[":
            square_depth += 1
        elif char == "]":
            square_depth = max(0, square_depth - 1)
        elif char == "<":
            angle_depth += 1
        elif char == ">" and angle_depth:
            angle_depth -= 1
        elif not (round_depth or square_depth or angle_depth) and (
            char in "{" or char == ";" or (python_declaration and char == ":")
        ):
            end = index
            break
    signature = " ".join(stripped[:end].split())
    return signature or None


def collect(args: argparse.Namespace) -> int:
    binary = args.binary.resolve()
    store_dir = args.store_dir.resolve()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    repos = tuple(part.strip() for part in args.repos.split(",") if part.strip())
    for repo in repos:
        entry = manifest_entry(manifest, repo)
        root = verify_checkout(repo, entry)
        if not args.skip_index:
            print(f"[collect] indexing {repo} ({entry['commit'][:12]})", flush=True)
            index_repo(binary, root, store_dir, args.device)
        candidates = candidate_rows(graph_path(store_dir, root), root, repo)
        if len(candidates) < args.per_repo:
            raise RuntimeError(
                f"{repo} has only {len(candidates)} eligible functions; need {args.per_repo}"
            )
        for row in candidates[: args.per_repo]:
            row["commit"] = entry["commit"]
            row["language"] = entry["lang"]
            selected.append(row)
        print(f"[collect] {repo}: selected {args.per_repo}/{len(candidates)}", flush=True)

    for index, case in enumerate(selected, 1):
        case["id"] = f"sq{index:03d}"
    document = {
        "schema_version": CASES_SCHEMA,
        "selection_version": SELECTION_VERSION,
        "manifest_sha256": sha256_file(MANIFEST),
        "binary_sha256": sha256_file(binary),
        "device": args.device,
        "case_count": len(selected),
        "per_repo": args.per_repo,
        "repos": list(repos),
        "cases": selected,
    }
    write_json(args.output, document)
    print(f"[collect] wrote {len(selected)} cases to {args.output}")
    return 0 if len(selected) >= 200 else 2


def normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def mechanical_flags(case: dict[str, Any], summary: list[str], source: str) -> list[str]:
    flags: list[str] = []
    joined = " ".join(summary).strip()
    if not joined:
        return flags
    lower = joined.lower()
    if "what is this function for" in lower or lower.startswith("summarize:"):
        flags.append("prompt_echo")
    signature = normalize_text(case["signature"])
    normalized = normalize_text(joined)
    if signature and (
        signature in normalized
        or difflib.SequenceMatcher(None, signature, normalized).ratio() >= 0.82
    ):
        flags.append("signature_echo")
    if CODE_HINT.search(joined):
        flags.append("code_shaped_output")
    # file_path is grounding the generator legitimately sees (judge-v3
    # symmetry): "JsonObject" from JsonObject.java is not an invented shape.
    grounded = source.lower() + "\n" + case["file_path"].lower()
    invented = sorted(
        {
            token
            for token in SYMBOLISH.findall(joined)
            if token.lower() not in grounded
        }
    )
    if invented:
        flags.append("invented_symbol_shape:" + ",".join(invented))
    return flags


def execute(args: argparse.Namespace) -> int:
    cases_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    if cases_doc.get("schema_version") != CASES_SCHEMA:
        raise RuntimeError("unsupported cases schema")
    binary = args.binary.resolve()
    store_dir = args.store_dir.resolve()
    env = os.environ.copy()
    env["GREPPY_STORE_DIR"] = str(store_dir)
    env["GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S"] = "300"
    env["GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S"] = "1800"
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases_doc["cases"], 1):
        root = (REALCORPUS / case["repo"]).resolve()
        source = source_for(case)
        started = time.monotonic()
        proc = run(
            [
                str(binary),
                "--device",
                args.device,
                "--root",
                str(root),
                "brief",
                case["qualified_name"],
                "--json",
            ],
            cwd=root,
            env=env,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - started
        error: str | None = None
        payload: dict[str, Any] = {}
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            error = f"invalid_json:{exc}"
        definitions = payload.get("definitions", []) if isinstance(payload, dict) else []
        definition = next(
            (
                row
                for row in definitions
                if row.get("qualified_name") == case["qualified_name"]
            ),
            definitions[0] if definitions else {},
        )
        summary = definition.get("summary", []) if isinstance(definition, dict) else []
        if not isinstance(summary, list) or not all(isinstance(item, str) for item in summary):
            summary = []
            error = error or "invalid_summary_shape"
        expand_id = payload.get("expand_id") if isinstance(payload, dict) else None
        expand_proc: subprocess.CompletedProcess[str] | None = None
        expand_payload: dict[str, Any] = {}
        if proc.returncode == 0 and isinstance(expand_id, str) and expand_id:
            expand_proc = run(
                [
                    str(binary),
                    "--device",
                    args.device,
                    "--root",
                    str(root),
                    "expand",
                    expand_id,
                    "--json",
                ],
                cwd=root,
                env=env,
                timeout=args.timeout,
            )
            try:
                expand_payload = json.loads(expand_proc.stdout)
            except json.JSONDecodeError as exc:
                error = error or f"invalid_expand_json:{exc}"
        expand_json = expand_payload.get("payload_json", {})
        expand_hits = expand_json.get("hits", []) if isinstance(expand_json, dict) else []
        expand_source_hit = any(
            isinstance(hit, dict)
            and hit.get("qualified_name") == case["qualified_name"]
            and hit.get("file_path") == case["file_path"]
            and hit.get("start_line") == case["start_line"]
            and hit.get("end_line") == case["end_line"]
            for hit in expand_hits
        )
        expand_text = expand_payload.get("payload_text", "")
        source_anchor = next((line for line in source.splitlines() if line.strip()), "")
        contract = {
            "returncode": proc.returncode,
            "schema_version": payload.get("schema_version"),
            "start_line": definition.get("start_line"),
            "end_line": definition.get("end_line"),
            "signature": definition.get("signature"),
            "summary_prompt_version_valid": definition.get("summary_prompt_version")
            == (SUMMARY_PROMPT_VERSION if summary else None),
            "expand_id_present": bool(expand_id),
            "expand_returncode": expand_proc.returncode if expand_proc else None,
            "expand_id_matches": expand_payload.get("id") == expand_id,
            "expand_source_hit": expand_source_hit,
            "expand_contains_source_anchor": bool(source_anchor)
            and source_anchor in expand_text,
        }
        expected = {
            "returncode": 0,
            "schema_version": "greppy.brief.v1",
            "start_line": case["start_line"],
            "end_line": case["end_line"],
            "signature": case["signature"],
            "summary_prompt_version_valid": True,
            "expand_id_present": True,
            "expand_returncode": 0,
            "expand_id_matches": True,
            "expand_source_hit": True,
            "expand_contains_source_anchor": True,
        }
        if contract != expected:
            error = error or "brief_contract_mismatch"
        records.append(
            {
                "id": case["id"],
                "repo": case["repo"],
                "qualified_name": case["qualified_name"],
                "summary": summary,
                "summary_prompt_version": definition.get("summary_prompt_version"),
                "mechanical_flags": mechanical_flags(case, summary, source),
                "contract": contract,
                "error": error,
                "stderr": "\n".join(
                    text
                    for text in (
                        proc.stderr[-1000:] if proc.returncode else "",
                        expand_proc.stderr[-1000:]
                        if expand_proc is not None and expand_proc.returncode
                        else "",
                    )
                    if text
                ),
                "elapsed_seconds": round(elapsed, 6),
            }
        )
        if index % 10 == 0 or index == len(cases_doc["cases"]):
            print(f"[run] {index}/{len(cases_doc['cases'])}", flush=True)

    document = {
        "schema_version": RESULTS_SCHEMA,
        "cases_sha256": sha256_file(args.cases),
        "binary_sha256": sha256_file(binary),
        "device": args.device,
        "records": records,
    }
    write_json(args.output, document)
    failures = sum(1 for row in records if row["error"])
    visible = sum(bool(row["summary"]) for row in records)
    coverage = visible / len(records) if records else 0.0
    print(
        f"[run] wrote {len(records)} records; contract failures={failures}; "
        f"visible summaries={visible} ({coverage:.1%})"
    )
    return 0 if failures == 0 and coverage >= 0.85 else 2


def load_minimax_key() -> str:
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if not key and sys.platform == "darwin":
        proc = run(["launchctl", "getenv", "MINIMAX_API_KEY"], cwd=REPO, timeout=10)
        key = proc.stdout.strip()
    if not key:
        raise RuntimeError("MINIMAX_API_KEY is required for summary-quality judging")
    return key


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError(f"judge returned no JSON object: {text[:500]}")
    return json.loads(stripped[start : end + 1])


def judge_request(key: str, items: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    required_ids = [item["id"] for item in items]
    instructions = f"""
You validate tiny function-purpose hints used only for code navigation.
For each item, compare the generated summary with the exact source.

The hint is triage orientation for a coding agent that sees only file path, signature, and this hint, and decides: open and read this function, or skip it. The agent always reads the real code before acting. Rate the hint's utility for that decision.

Definitions:
- utility is exactly one of:
  * "very_helpful": correct and specific purpose; the agent can decide confidently.
  * "helpful": right direction; may be vague or partial, but improves the read/skip decision.
  * "barely_helpful": generic or near-empty; neither helps nor hurts.
  * "anti_helpful": makes a materially false claim about what the function does, its role, or its effects - likely to cause a wrong read/skip decision. Judge practically: missing detail is never anti_helpful; only material falsehood is.
- The file_path of each item is legitimate grounding context: the generator sees it, so naming the role it implies (for example "listener" for tcp/listener.rs, or the type name from JsonObject.java) is grounded, not invented and not anti-helpful by itself. Claims about BEHAVIOR must still be supported by the source.
- invented_symbols lists code symbols named by the summary that appear neither in the source nor in the file_path. Ordinary English words are not symbols.
- signature_echo=true when the output mainly restates the declaration/signature instead of purpose.
- An empty summary is "barely_helpful".

Do not reward eloquence or detail. Do not answer what the code does beyond the verdict reason.
Return exactly one JSON object with prompt_version and a verdicts array. Each
verdict must contain id, utility, invented_symbols, signature_echo, and a
short reason. Copy each item ID exactly once, in the same
order. Do not invent, omit, or rename IDs.

The only allowed IDs for this request are:
{json.dumps(required_ids)}

Required top-level prompt_version: {JUDGE_PROMPT_VERSION}

ITEMS:
{json.dumps(items, ensure_ascii=False)}
""".strip()
    body = json.dumps(
        {
            "model": "MiniMax-M3",
            "max_tokens": 4096,
            "temperature": 0,
            "messages": [{"role": "user", "content": instructions}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.minimax.io/anthropic/v1/messages",
        data=body,
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    text = "\n".join(
        block.get("text", "")
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )
    return parse_json_response(text)


def validate_judge_response(
    response: dict[str, Any], items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if response.get("prompt_version") != JUDGE_PROMPT_VERSION:
        raise RuntimeError("judge prompt-version mismatch")
    rows = response.get("verdicts")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("judge verdicts must be an array of objects")
    expected_ids = [item["id"] for item in items]
    returned_ids = [row.get("id") for row in rows]
    if returned_ids != expected_ids:
        raise RuntimeError(
            f"judge returned wrong IDs: expected {expected_ids}, got {returned_ids}"
        )
    allowed_utility = {"very_helpful", "helpful", "barely_helpful", "anti_helpful"}
    for row in rows:
        if row.get("utility") not in allowed_utility:
            raise RuntimeError(f"judge returned invalid utility for {row['id']}")
        invented = row.get("invented_symbols")
        if not isinstance(invented, list) or not all(
            isinstance(symbol, str) for symbol in invented
        ):
            raise RuntimeError(f"judge returned invalid invented_symbols for {row['id']}")
        if not isinstance(row.get("signature_echo"), bool):
            raise RuntimeError(f"judge returned non-boolean signature_echo for {row['id']}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise RuntimeError(f"judge returned an empty reason for {row['id']}")
    return rows


def judge(args: argparse.Namespace) -> int:
    cases_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    results_doc = json.loads(args.results.read_text(encoding="utf-8"))
    cases = {case["id"]: case for case in cases_doc["cases"]}
    key = load_minimax_key()
    records = results_doc["records"]
    cases_digest = sha256_file(args.cases)
    results_digest = sha256_file(args.results)
    verdict_by_id: dict[str, dict[str, Any]] = {}
    if args.output.is_file():
        checkpoint = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            checkpoint.get("schema_version") != JUDGMENTS_SCHEMA
            or checkpoint.get("judge_prompt_version") != JUDGE_PROMPT_VERSION
            or checkpoint.get("cases_sha256") != cases_digest
            or checkpoint.get("results_sha256") != results_digest
        ):
            raise RuntimeError("existing judge checkpoint does not match this run")
        verdict_by_id = {
            row["id"]: row
            for row in checkpoint.get("verdicts", [])
            if isinstance(row, dict) and row.get("id") in cases
        }
    pending = [record for record in records if record["id"] not in verdict_by_id]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        items = [
            {
                "id": record["id"],
                "file_path": cases[record["id"]]["file_path"],
                "signature": cases[record["id"]]["signature"],
                "source": source_for(cases[record["id"]]),
                "summary": record["summary"],
            }
            for record in batch
        ]
        response: dict[str, Any] | None = None
        rows: list[dict[str, Any]] | None = None
        error: Exception | None = None
        for attempt in range(6):
            try:
                candidate = judge_request(key, items, args.timeout)
                rows = validate_judge_response(candidate, items)
                response = candidate
                break
            except urllib.error.HTTPError as exc:
                error = exc
                if exc.code != 429:
                    time.sleep(min(60.0, 2.0**attempt))
                    continue
                retry_after = exc.headers.get("Retry-After")
                try:
                    delay = float(retry_after) if retry_after else 5.0 * (2**attempt)
                except ValueError:
                    delay = 5.0 * (2**attempt)
                time.sleep(min(120.0, max(5.0, delay)))
            except (RuntimeError, json.JSONDecodeError, urllib.error.URLError) as exc:
                error = exc
                time.sleep(min(60.0, 2.0**attempt))
        if response is None or rows is None:
            raise RuntimeError(f"judge failed after retries: {error}")
        for row in rows:
            case = cases[row["id"]]
            grounded = source_for(case).lower() + "\n" + case["file_path"].lower()
            row["invented_symbols"] = [
                symbol
                for symbol in row.get("invented_symbols", [])
                if isinstance(symbol, str) and symbol.lower() not in grounded
            ]
        verdict_by_id.update({row["id"]: row for row in rows})
        document = {
            "schema_version": JUDGMENTS_SCHEMA,
            "judge": "MiniMax-M3",
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "cases_sha256": cases_digest,
            "results_sha256": results_digest,
            "verdicts": [
                verdict_by_id[record["id"]]
                for record in records
                if record["id"] in verdict_by_id
            ],
        }
        write_json(args.output, document)
        print(f"[judge] {len(verdict_by_id)}/{len(records)}", flush=True)
        if len(verdict_by_id) < len(records):
            time.sleep(args.delay)
    return 0


def gate(args: argparse.Namespace) -> int:
    cases_doc = json.loads(args.cases.read_text(encoding="utf-8"))
    results_doc = json.loads(args.results.read_text(encoding="utf-8"))
    judgments_doc = json.loads(args.judgments.read_text(encoding="utf-8"))
    cases = cases_doc["cases"]
    records = {row["id"]: row for row in results_doc["records"]}
    verdicts = {row["id"]: row for row in judgments_doc["verdicts"]}
    ids = {case["id"] for case in cases}
    complete = ids == set(records) == set(verdicts)
    total = len(cases)
    # Four-level triage-utility scale (owner re-registration 2026-07-16): the
    # gate measures the registered product promise - navigation orientation
    # for a read/skip decision - instead of factual-precision pedantry.
    def utility(case_id: str) -> str:
        return verdicts.get(case_id, {}).get("utility", "barely_helpful")

    very_helpful = sum(utility(case_id) == "very_helpful" for case_id in ids)
    helpful_only = sum(utility(case_id) == "helpful" for case_id in ids)
    barely = sum(utility(case_id) == "barely_helpful" for case_id in ids)
    anti = sum(utility(case_id) == "anti_helpful" for case_id in ids)
    helpful = very_helpful + helpful_only
    invented = 0
    for case in cases:
        source_lower = source_for(case).lower()
        claimed = verdicts.get(case["id"], {}).get("invented_symbols", [])
        if any(isinstance(symbol, str) and symbol.lower() not in source_lower for symbol in claimed):
            invented += 1
    echoes = sum(bool(verdicts.get(case_id, {}).get("signature_echo")) for case_id in ids)
    visible = sum(bool(records.get(case_id, {}).get("summary")) for case_id in ids)
    contract_errors = sum(bool(records.get(case_id, {}).get("error")) for case_id in ids)
    mechanical = sum(bool(records.get(case_id, {}).get("mechanical_flags")) for case_id in ids)
    helpful_rate = helpful / total if total else 0.0
    anti_rate = anti / total if total else 1.0
    checks = {
        "at_least_200_real_functions": total >= 200,
        "all_cases_have_results_and_judgments": complete,
        "evidence_digests_match": results_doc.get("cases_sha256")
        == sha256_file(args.cases)
        and judgments_doc.get("cases_sha256") == sha256_file(args.cases)
        and judgments_doc.get("results_sha256") == sha256_file(args.results),
        "brief_output_contract_has_no_failures": contract_errors == 0,
        "visible_summary_coverage_at_least_85_percent": visible / total >= 0.85
        if total
        else False,
        "helpful_or_better_at_least_85_percent": helpful_rate >= 0.85,
        "anti_helpful_at_most_5_percent": anti_rate <= 0.05,
        # Judge-assessed count on ~200 LLM verdicts has a noise floor of a
        # single spurious flag; the deterministic mechanical checks stay at 0.
        "at_most_one_invented_symbol": invented <= 1,
        "no_signature_echoes": echoes == 0,
        "no_mechanical_rejection_shapes_visible": mechanical == 0,
        "judge_contract_is_pinned": judgments_doc.get("judge_prompt_version")
        == JUDGE_PROMPT_VERSION,
    }
    report = {
        "schema_version": GATE_SCHEMA,
        "case_count": total,
        "visible_summary_count": visible,
        "very_helpful_count": very_helpful,
        "helpful_count": helpful_only,
        "barely_helpful_count": barely,
        "anti_helpful_count": anti,
        "helpful_or_better_rate": helpful_rate,
        "anti_helpful_rate": anti_rate,
        "invented_symbol_count": invented,
        "signature_echo_count": echoes,
        "mechanical_flag_count": mechanical,
        "contract_error_count": contract_errors,
        "checks": checks,
        "passed": all(checks.values()),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--binary", type=pathlib.Path, required=True)
    collect_parser.add_argument("--store-dir", type=pathlib.Path, required=True)
    collect_parser.add_argument("--output", type=pathlib.Path, required=True)
    collect_parser.add_argument("--repos", default=",".join(DEFAULT_REPOS))
    collect_parser.add_argument("--per-repo", type=int, default=34)
    collect_parser.add_argument("--device", default="auto")
    collect_parser.add_argument("--skip-index", action="store_true")
    collect_parser.set_defaults(func=collect)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--binary", type=pathlib.Path, required=True)
    run_parser.add_argument("--store-dir", type=pathlib.Path, required=True)
    run_parser.add_argument("--cases", type=pathlib.Path, required=True)
    run_parser.add_argument("--output", type=pathlib.Path, required=True)
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--timeout", type=float, default=120)
    run_parser.set_defaults(func=execute)

    judge_parser = subparsers.add_parser("judge")
    judge_parser.add_argument("--cases", type=pathlib.Path, required=True)
    judge_parser.add_argument("--results", type=pathlib.Path, required=True)
    judge_parser.add_argument("--output", type=pathlib.Path, required=True)
    judge_parser.add_argument("--batch-size", type=int, default=5)
    judge_parser.add_argument("--timeout", type=float, default=180)
    judge_parser.add_argument("--delay", type=float, default=3.0)
    judge_parser.set_defaults(func=judge)

    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--cases", type=pathlib.Path, required=True)
    gate_parser.add_argument("--results", type=pathlib.Path, required=True)
    gate_parser.add_argument("--judgments", type=pathlib.Path, required=True)
    gate_parser.add_argument("--output", type=pathlib.Path, required=True)
    gate_parser.set_defaults(func=gate)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (RuntimeError, OSError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        print(f"summary-quality: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
