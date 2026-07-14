#!/usr/bin/env python3
"""Parallel greppy product acceptance runner.

Runs the same product comparison as acceptance_run.py, but fans out task-level
Pi/MiniMax-M3 jobs so a full 100-task run does not serialize one agent loop at a
time. Each worker writes an isolated one-task results file; the orchestrator
merges rows only after all workers finish, then grades and runs forensics for
the single product comparison: greppy vs the explorer gate baseline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
TASKS = HERE / "tasks_v2.json"
BIN = pathlib.Path(os.environ["GREPPY_BENCH_BIN"]) if os.environ.get("GREPPY_BENCH_BIN") else REPO / "target" / "release" / "greppy"
# Quota circuit breaker (mirrors run_bench.py): after this many CONSECUTIVE
# agent sessions dying on a rate-limit/quota error the run aborts with exit 3.
# It is the FINAL BACKSTOP for sustained failure: the first line of defense
# is run_bench.run_pi's in-session 45/90/180s rate-limit backoff (a MiniMax
# 429 "Token Plan rate limit reached" (2062) is a CONCURRENCY/window limit,
# not quota exhaustion), so a session that still errors already failed 4
# attempts across ~5 min of backoff — 8 consecutive such sessions is a dead
# plan or hard cap, not a transient window. On 2026-07-02, 20 parallel
# workers burned 282 sessions (244/282 failed, 429 error 2062) and still
# reported every step exit 0. A worker that trips its own breaker exits 3;
# the orchestrator additionally counts across workers, stops submitting new
# tasks, and marks the run FAILED in SUMMARY.md.
QUOTA_BREAKER_LIMIT = 8
QUOTA_BREAKER_EXIT = 3
QUOTA_ERROR_RE = re.compile(r"rate[ _-]?limit|\b429\b|quota", re.IGNORECASE)
# run_bench.py per-task agent execution order; used to count consecutive
# quota-errored sessions inside each finished worker's one-task results file.
AGENT_RUN_ORDER = ("grep", "greppy", "plus", "gemma", "explorer")
# Product run: greppy (candidate) vs explorer (uncoached PRODUKT-BASELINE,
# the Z1/Z2 gate baseline per PLAN_10X §2) with the coached grep agent as the
# co-reported diagnostic row (R4: no post-hoc baseline disputes).
DEFAULT_AGENTS = "grep,greppy,explorer"
# The product gate is FIXED: greppy vs explorer (uncoached "normales grep").
# It is not a CLI choice — a selectable gate baseline is exactly the R4
# relabeling path the contract forbids (Codex-Review P0-2). The coached grep
# leg is always a diagnostic report, never a gate.
GATE_BASELINE = "explorer"
PROVIDER_KEYS = {"minimax": "MINIMAX_API_KEY"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--candidate", default="greppy")
    ap.add_argument("--tasks", type=pathlib.Path, default=TASKS,
                    help="tasks file (tasks.json = synthetic v1 contract; "
                         "tasks_v2.json = corpus v2 real-repo contract)")
    ap.add_argument("--parallel", type=int, default=5,
                    help="concurrent task workers (default 5). Each worker "
                         "drives ONE sequential pi/MiniMax stream, so this is "
                         "the concurrent-stream count. MiniMax Token Plans "
                         "are CONCURRENCY-sensitive: the 2026-07-02 P1 run "
                         "with 20 workers (~20 parallel streams) tripped "
                         "'Token Plan rate limit reached' (429, code 2062) "
                         "despite the 5h plan window being only 53%% used.")
    ap.add_argument("--llm-provider", default="minimax", choices=sorted(PROVIDER_KEYS),
                    help="minimax = contract model; zai = GLM-5.2 validation axis "
                         "(reported separately, never a gate substitute)")
    ap.add_argument("--repo", help="optional corpus repo filter")
    ap.add_argument("task_ids", nargs="*", help="optional task IDs")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--output-dir", type=pathlib.Path)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--index-only", action="store_true",
                    help="materialize the greppy stores for every selected "
                         "task repo, then exit without running agents. Used "
                         "by CI to prime the store cache in its own job so "
                         "the 6h hosted-runner cap never hits the agent "
                         "phase (indexing alone takes hours on CPU).")
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--allow-unaccepted", action="store_true")
    args = ap.parse_args()

    tasks_path = args.tasks
    if not tasks_path.is_absolute():
        tasks_path = HERE / tasks_path
    if not tasks_path.exists():
        raise SystemExit(f"tasks file not found: {tasks_path}")
    # Corpus v2 (tasks_v2.json): real pinned repos + reused v1 controls. The
    # matching classes doc and verifier differ from v1, so detect it here.
    is_v2 = "_v2" in tasks_path.name
    task_classes_path = (
        HERE / "task_classes_v2.json" if is_v2 else HERE / "task_classes.json"
    )

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if set(agents) != {"grep", "greppy", "explorer"}:
        raise SystemExit(
            "product acceptance runs exactly --agents grep,greppy,explorer: "
            "explorer is the fixed gate baseline, coached grep the diagnostic "
            "row. Diagnostic ablations (gemma/plus) belong in a separate "
            "research run (run_bench.py --diagnostic)."
        )
    if args.candidate != "greppy":
        raise SystemExit("the product candidate is greppy")
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")

    key_env = PROVIDER_KEYS[args.llm_provider]
    ensure_api_key(key_env)
    if not args.skip_bench and not args.index_only and not os.environ.get(key_env):
        raise SystemExit(
            f"{key_env} is missing. Export it or set it with launchctl; "
            "do not pass it on argv."
        )

    run_dir = args.output_dir or (HERE / "acceptance_runs" / args.run_id)
    raw_dir = run_dir / "raw"
    worker_dir = run_dir / "worker-results"
    logs_dir = run_dir / "logs"
    # Run-local greppy store: isolates the benchmark from the shared user
    # cache (~/Library/Caches/greppy), which other sessions have wiped
    # mid-run before. Inherited by verify/index steps and every worker's pi
    # agent via the environment (PLAN_10X §8 Risiko 7).
    # A pre-set GREPPY_STORE_DIR wins: CI primes the store in a separate job
    # (--index-only) and restores it from cache, so both jobs must agree on
    # one fixed path outside the per-run directory.
    store_dir = pathlib.Path(os.environ.get("GREPPY_STORE_DIR") or (run_dir / "store"))
    store_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GREPPY_STORE_DIR"] = str(store_dir)
    # The acceptance run deliberately excludes one-time precomputation from
    # both agent arms. Finish embeddings before the first measured task instead
    # of allowing the interactive lazy-index policy to race agent workers.
    os.environ["GREPPY_LAZY_EMBED_MIN_SPANS"] = str(sys.maxsize)
    print("== embedded EmbeddingGemma + Qwen summaries active", file=sys.stderr)
    results = run_dir / "results.json"
    graded_results = run_dir / "results.mechanical.json"
    aggregate = run_dir / "aggregate.txt"
    forensics = run_dir / f"FORENSICS_{GATE_BASELINE}_VS_greppy.md"
    summary = run_dir / "SUMMARY.md"
    for path in (raw_dir, worker_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    steps: list[tuple[str, int]] = []

    if not args.skip_build:
        steps.append((
            "build-greppy",
            run_logged(
                ["cargo", "build", "--release", "--bin", "greppy"],
                logs_dir / "build-greppy.log",
            ),
        ))
    write_run_manifest(
        run_dir=run_dir,
        tasks_path=tasks_path,
        task_classes_path=task_classes_path,
        llm_provider=args.llm_provider,
        agents=agents,
    )
    if not args.skip_verify:
        if is_v2:
            # Corpus-v2 contract verifier: validates tasks_v2.json +
            # task_classes_v2.json against candidates.json/MANIFEST and
            # re-runs the mechanical gates (firewall, multi-hop, byte
            # reproduction). Deterministic and API-free, but a failure means
            # the contract dataset is broken -- abort before spending tokens.
            verify_rc = run_logged(
                [sys.executable, str(HERE / "verify_real_tasks.py")],
                logs_dir / "verify-real-tasks.log",
            )
            steps.append(("verify-real-tasks", verify_rc))
            if verify_rc != 0:
                write_summary(summary, run_dir, steps, results, graded_results,
                              aggregate, forensics, tasks_path)
                return verify_rc
        else:
            steps.append((
                "verify-task-classes",
                run_logged(
                    [sys.executable, str(HERE / "verify_task_classes.py")],
                    logs_dir / "verify-task-classes.log",
                ),
            ))
            steps.append((
                "verify-tasks-index",
                run_logged(
                    [sys.executable, str(HERE / "verify_tasks.py"), "--index"],
                    logs_dir / "verify-tasks-index.log",
                ),
            ))

    selected_tasks = select_tasks(
        tasks_path, args.repo, set(args.task_ids) if args.task_ids else None
    )
    if not selected_tasks:
        raise SystemExit("no tasks selected")

    if is_v2:
        # Index every repo the selected tasks touch (4 real repos under
        # realcorpus/ plus the reused synthetic control repos under corpus/)
        # into the run-local GREPPY_STORE_DIR BEFORE the agent loop, so no
        # measured agent pays the one-time indexing cost.
        index_rc = index_task_repos(selected_tasks, logs_dir)
        steps.append(("index-task-repos", index_rc))
        if index_rc != 0:
            write_summary(summary, run_dir, steps, results, graded_results,
                          aggregate, forensics, tasks_path)
            return index_rc

    if args.index_only:
        print("== index-only: stores materialized, skipping agent phase",
              file=sys.stderr)
        write_summary(summary, run_dir, steps, results, graded_results,
                      aggregate, forensics, tasks_path)
        return 0

    bench_rc = 0
    if not args.skip_bench:
        bench_rc = run_parallel_bench(
            selected_tasks=selected_tasks,
            parallel=args.parallel,
            worker_dir=worker_dir,
            raw_dir=raw_dir,
            logs_dir=logs_dir,
            rerun=args.rerun,
            agents=",".join(agents),
            llm_provider=args.llm_provider,
            tasks_path=tasks_path,
        )
        steps.append(("run-bench-parallel", bench_rc))
        if bench_rc != 0:
            if bench_rc == QUOTA_BREAKER_EXIT:
                status = (
                    "FAILED — circuit breaker: sustained provider rate-limit/"
                    f"quota failure ({QUOTA_BREAKER_LIMIT}+ consecutive "
                    "rate-limit/429/quota agent sessions, each after "
                    "in-session 45/90/180s backoffs). Remaining tasks were "
                    "cancelled; the partial results are NOT decision-fähig."
                )
            else:
                status = f"FAILED — run-bench-parallel exited {bench_rc}"
            write_summary(summary, run_dir, steps, results, graded_results,
                          aggregate, forensics, tasks_path, status=status)
            return bench_rc

    merge_worker_results(worker_dir, results, selected_tasks)
    steps.append(("merge-results", 0))

    steps.append((
        "aggregate-report",
        run_logged(
            [sys.executable, str(HERE / "run_bench.py"),
             "--results", str(results), "--tasks", str(tasks_path),
             "--report"],
            logs_dir / "aggregate-report.log",
            tee_path=aggregate,
        ),
    ))
    steps.append((
        "mechanical-grade",
        run_logged(
            [
                sys.executable,
                str(HERE / "grade_answers.py"),
                "--mode",
                "mechanical",
                "--accept-mechanical",
                "--results",
                str(results),
                "--tasks",
                str(tasks_path),
                "--output",
                str(graded_results),
                "--agents",
                ",".join(agents),
            ],
            logs_dir / "mechanical-grade.log",
        ),
    ))
    # Gate leg: candidate vs the FIXED gate baseline (explorer). The coached
    # grep leg gets a co-reported diagnostic forensics file, but only the
    # explorer leg decides the exit code (Codex-Review P0-2).
    gate_rc = 0
    for base in (GATE_BASELINE, "grep"):
        is_gate = base == GATE_BASELINE
        out = run_dir / f"FORENSICS_{base}_VS_greppy.md"
        rc = run_logged(
            [
                sys.executable,
                str(HERE / "forensics.py"),
                "--results",
                str(graded_results),
                "--baseline",
                base,
                "--candidate",
                "greppy",
                "--task-classes",
                str(task_classes_path),
                "--output",
                str(out),
                "--enforce",
            ],
            logs_dir / f"forensics-{base}-vs-greppy.log",
            allowed={0, 2},
        )
        steps.append((f"forensics-{base}-vs-greppy"
                      + ("" if is_gate else " (diagnostic)"), rc))
        if is_gate:
            gate_rc = rc
            forensics = out
    release_gate = run_dir / "release-gate.json"
    release_gate_rc = run_logged(
        [
            sys.executable,
            str(HERE / "release_gate.py"),
            "--results",
            str(graded_results),
            "--output",
            str(release_gate),
            "--baseline",
            GATE_BASELINE,
            "--candidate",
            "greppy",
        ],
        logs_dir / "release-gate.log",
        allowed={0, 2},
    )
    steps.append(("release-agent-efficiency-gate", release_gate_rc))
    if release_gate_rc != 0:
        gate_rc = release_gate_rc
    write_summary(summary, run_dir, steps, results, graded_results, aggregate, forensics, tasks_path)
    if gate_rc != 0 and not args.allow_unaccepted:
        return 2
    return 0


def index_task_repos(
    selected_tasks: list[dict[str, Any]], logs_dir: pathlib.Path
) -> int:
    """Index every repo root the selected tasks reference into the (already
    exported) run-local GREPPY_STORE_DIR. Root resolution is shared with
    run_bench.py: real repos -> realcorpus/<repo>, synthetic -> corpus/<repo>."""
    sys.path.insert(0, str(HERE))
    import run_bench  # noqa: PLC0415 (shared repo-root resolution)

    repos = sorted({task["repo"] for task in selected_tasks})
    print(f"== index-task-repos: {', '.join(repos)}", file=sys.stderr)
    for repo in repos:
        root = run_bench.repo_root(repo)
        if not pathlib.Path(root).is_dir():
            print(
                f"missing repo root {root} -- run real_corpus.py setup "
                "(real repos) or gen_corpus.sh (synthetic corpus) first",
                file=sys.stderr,
            )
            return 1
        cmd = [str(BIN), "index", root, "--root", root]
        rc = run_logged(cmd, logs_dir / f"index-{repo}.log")
        if rc != 0:
            return rc
    return 0


def run_parallel_bench(
    selected_tasks: list[dict[str, Any]],
    parallel: int,
    worker_dir: pathlib.Path,
    raw_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    rerun: bool,
    agents: str = DEFAULT_AGENTS,
    llm_provider: str = "minimax",
    tasks_path: pathlib.Path = TASKS,
) -> int:
    max_workers = min(parallel, len(selected_tasks))
    print(
        f"== run-bench-parallel: {len(selected_tasks)} tasks, {max_workers} workers",
        file=sys.stderr,
    )
    failures: list[tuple[str, int]] = []
    consecutive_quota = 0
    breaker_reason: str | None = None
    submitted = 0
    done = 0
    # Tasks are submitted INCREMENTALLY (max_workers up front, one more per
    # completion) so that a tripped circuit breaker really stops submission:
    # with an up-front bulk submit, pool worker threads grab queued tasks
    # before Future.cancel() can land.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending: dict[concurrent.futures.Future, str] = {}

        def submit_next() -> None:
            nonlocal submitted
            if submitted >= len(selected_tasks):
                return
            task = selected_tasks[submitted]
            submitted += 1
            future = pool.submit(run_one_task, task, worker_dir, raw_dir,
                                 logs_dir, rerun, agents, llm_provider,
                                 tasks_path)
            pending[future] = task["id"]

        for _ in range(max_workers):
            submit_next()
        while pending:
            done_futures, _ = concurrent.futures.wait(
                list(pending), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done_futures:
                tid = pending.pop(future)
                done += 1
                try:
                    rc = future.result()
                except Exception as exc:  # pragma: no cover - defensive orchestrator path
                    rc = 99
                    (logs_dir / f"{tid}.exception.log").write_text(str(exc), encoding="utf-8")
                status = "ok" if rc == 0 else f"exit={rc}"
                print(f"== [{done:03d}/{len(selected_tasks):03d}] {tid}: {status}", file=sys.stderr)
                if rc != 0:
                    failures.append((tid, rc))
                if rc == QUOTA_BREAKER_EXIT:
                    # A worker tripped its own in-process breaker: quota dead.
                    breaker_reason = (
                        f"worker {tid} exited {QUOTA_BREAKER_EXIT} (its "
                        "in-process quota circuit breaker tripped)"
                    )
                elif breaker_reason is None:
                    consecutive_quota = count_consecutive_quota_errors(
                        worker_dir / f"{tid}.json", consecutive_quota
                    )
                    if consecutive_quota >= QUOTA_BREAKER_LIMIT:
                        breaker_reason = (
                            f"{consecutive_quota} consecutive rate-limit/quota "
                            "(429) agent sessions across workers"
                        )
                if breaker_reason is None:
                    submit_next()
            if breaker_reason:
                remaining = len(selected_tasks) - submitted
                print(
                    f"CIRCUIT BREAKER: {breaker_reason} -- sustained provider "
                    "failure: each of those sessions already sat out "
                    "run_bench.run_pi's in-session 45/90/180s rate-limit "
                    "backoffs and still died. NOT submitting the "
                    f"remaining {remaining} tasks (waiting for "
                    f"{len(pending)} in-flight), aborting with exit "
                    f"{QUOTA_BREAKER_EXIT}. Fix the quota/concurrency, then "
                    "re-run with the same run dir to resume (errored "
                    "sessions re-run automatically).",
                    file=sys.stderr,
                )
                break
    if breaker_reason:
        return QUOTA_BREAKER_EXIT
    if failures:
        print(
            "failed tasks: "
            + ", ".join(f"{tid}:{rc}" for tid, rc in failures[:20]),
            file=sys.stderr,
        )
        return 1
    return 0


def count_consecutive_quota_errors(
    worker_result: pathlib.Path, consecutive: int
) -> int:
    """Fold one finished worker's one-task results file into the running count
    of consecutive quota-errored agent sessions. Any healthy session resets
    the streak; unreadable/missing files leave it unchanged (the worker's own
    exit code already reports those failures)."""
    try:
        rows = json.loads(worker_result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return consecutive
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        for agent in AGENT_RUN_ORDER:
            res = row.get(agent)
            if not isinstance(res, dict) or "wall_s" not in res:
                continue
            if QUOTA_ERROR_RE.search(str(res.get("error") or "")):
                consecutive += 1
            else:
                consecutive = 0
    return consecutive


def run_one_task(
    task: dict[str, Any],
    worker_dir: pathlib.Path,
    raw_dir: pathlib.Path,
    logs_dir: pathlib.Path,
    rerun: bool,
    agents: str = DEFAULT_AGENTS,
    llm_provider: str = "minimax",
    tasks_path: pathlib.Path = TASKS,
) -> int:
    tid = task["id"]
    cmd = [
        sys.executable,
        str(HERE / "run_bench.py"),
        "--results",
        str(worker_dir / f"{tid}.json"),
        "--tasks",
        str(tasks_path),
        "--agents",
        agents,
        "--llm-provider",
        llm_provider,
        "--save-raw",
        "--raw-dir",
        str(raw_dir),
    ]
    if rerun:
        cmd.append("--rerun")
    cmd.append(tid)
    return run_logged(cmd, logs_dir / f"{tid}.log")


def select_tasks(
    tasks_path: pathlib.Path, repo: str | None, task_ids: set[str] | None
) -> list[dict[str, Any]]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    out = []
    for task in tasks:
        if repo and task.get("repo") != repo:
            continue
        if task_ids and task.get("id") not in task_ids:
            continue
        out.append(task)
    return out


def merge_worker_results(
    worker_dir: pathlib.Path,
    results: pathlib.Path,
    selected_tasks: list[dict[str, Any]],
) -> None:
    by_id: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        path = worker_dir / f"{task['id']}.json"
        if not path.exists():
            raise SystemExit(f"missing worker result: {path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != 1 or rows[0].get("id") != task["id"]:
            raise SystemExit(f"invalid worker result: {path}")
        by_id[task["id"]] = rows[0]
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(
        json.dumps([by_id[t["id"]] for t in selected_tasks], indent=2),
        encoding="utf-8",
    )


def run_logged(
    cmd: list[str],
    log_path: pathlib.Path,
    allowed: set[int] | None = None,
    tee_path: pathlib.Path | None = None,
) -> int:
    allowed = allowed or {0}
    rendered = shlex.join(cmd)
    print(f"== {rendered}", file=sys.stderr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if tee_path:
        tee_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    log_path.write_text(f"$ {rendered}\n\n{proc.stdout}\nexit={proc.returncode}\n", encoding="utf-8")
    if tee_path:
        tee_path.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode not in allowed:
        print(f"command failed: {rendered} -> {proc.returncode}", file=sys.stderr)
        # CI keeps only the step transcript, not the run directory: without
        # the captured tail a failure here is undiagnosable (the 2026-07-13
        # exit-73 index failure burned a full runner-hour with zero output).
        tail = proc.stdout.splitlines()[-40:]
        if tail:
            print(f"--- last {len(tail)} log lines ({log_path.name}) ---", file=sys.stderr)
            for line in tail:
                print(f"| {line}", file=sys.stderr)
            print("--- end log tail ---", file=sys.stderr)
    return proc.returncode


def ensure_api_key(key_env: str) -> None:
    if os.environ.get(key_env):
        return
    try:
        proc = subprocess.run(
            ["launchctl", "getenv", key_env],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, ValueError):
        return
    value = proc.stdout.strip()
    if value:
        os.environ[key_env] = value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    run_dir: pathlib.Path,
    tasks_path: pathlib.Path,
    task_classes_path: pathlib.Path,
    llm_provider: str,
    agents: list[str],
) -> None:
    """Pin every input needed to audit or reproduce one acceptance run."""
    if not BIN.is_file():
        raise SystemExit(f"benchmark binary missing after build: {BIN}")
    sys.path.insert(0, str(HERE))
    import run_bench  # noqa: PLC0415

    version = subprocess.run(
        [str(BIN), "--version"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout.strip()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout.strip()
    repo_manifest = HERE / "realcorpus" / "MANIFEST.json"
    manifest = {
        "schema_version": "greppy.agent-benchmark-run.v1",
        "product_candidate": "greppy",
        "gate_baseline": GATE_BASELINE,
        "diagnostic_baseline": "grep",
        "agents": agents,
        "provider": llm_provider,
        "model": run_bench.PROVIDERS[llm_provider]["model"],
        "prompt_contract": run_bench.prompt_contract(),
        "greppy": {
            "git_commit": git_head,
            "version": version,
            "binary_sha256": sha256_file(BIN),
        },
        "tasks": {
            "path": str(tasks_path.relative_to(REPO)),
            "sha256": sha256_file(tasks_path),
            "count": len(json.loads(tasks_path.read_text(encoding="utf-8"))),
        },
        "task_classes": {
            "path": str(task_classes_path.relative_to(REPO)),
            "sha256": sha256_file(task_classes_path),
        },
        "repository_manifest": {
            "path": str(repo_manifest.relative_to(REPO)),
            "sha256": sha256_file(repo_manifest),
        },
        "raw_traces_published": False,
        "publishable_artifacts": [
            "RUN_MANIFEST.json",
            "results.json",
            "results.mechanical.json",
            "aggregate.txt",
            f"FORENSICS_{GATE_BASELINE}_VS_greppy.md",
            "release-gate.json",
            "SUMMARY.md",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_summary(
    summary: pathlib.Path,
    run_dir: pathlib.Path,
    steps: list[tuple[str, int]],
    results: pathlib.Path,
    graded_results: pathlib.Path,
    aggregate: pathlib.Path,
    forensics: pathlib.Path,
    tasks_path: pathlib.Path | None = None,
    status: str | None = None,
) -> None:
    lines = [
        f"# greppy Parallel Acceptance Run - {run_dir.name}",
        "",
    ]
    if status:
        lines += [
            "## Status",
            "",
            f"**RUN {status}**",
            "",
        ]
    lines += [
        "## Configuration",
        "",
        "- Product comparison: `greppy` (candidate) vs gate baseline "
        "(uncoached explorer = 'normales grep' by default); coached grep "
        "co-reported as diagnostic",
        f"- Tasks file: `{tasks_path or TASKS}`",
        f"- GREPPY_STORE_DIR (run-local, isolated): `{os.environ.get('GREPPY_STORE_DIR', '')}`",
        "",
        "## Artifacts",
        "",
        f"- Results: `{results}`",
        f"- Run manifest: `{run_dir / 'RUN_MANIFEST.json'}`",
        f"- Mechanical results: `{graded_results}`",
        f"- Aggregate report: `{aggregate}`",
        f"- Forensics: `{forensics}`",
        f"- Release gate: `{run_dir / 'release-gate.json'}`",
        "",
        "## Steps",
        "",
        "| Step | Exit |",
        "|---|---:|",
    ]
    for name, rc in steps:
        lines.append(f"| `{name}` | {rc} |")
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
