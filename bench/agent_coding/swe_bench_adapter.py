#!/usr/bin/env python3
"""Harvest real edit tasks from merged PRs that close issues (SWE-bench style).

The synthetic v1 bank injects a mutation and asks the agent to revert it with a
leading prompt. That is not what a coding agent is really given. This adapter
derives tasks the honest way, mirroring how the discovery bench used SWE-QA-Pro:

  * SOURCE   a merged PR that closes a real GitHub issue, touching BOTH source
             and test files.
  * STATE    the repo at the PR's pre-change commit (first parent of the merge,
             M^1 — the pre-PR state for both merge- and squash-merged PRs).
  * PROMPT   the issue title + body, verbatim. Not authored, not leading.
  * VERIFY   the PR's own test patch. FAIL_TO_PASS tests fail at base+test_patch
             and pass after the gold source patch; PASS_TO_PASS tests pass in
             both states (regression guard). No LLM judge.

Grading discipline matches swe_qa_adapter.py: a task is kept ONLY if it
mechanically validates (builds at base, the FAIL_TO_PASS set genuinely flips
red->green with the gold patch, PASS_TO_PASS stays green). Nothing is trusted
on faith.

Subcommands:
  harvest   gh API -> candidate PRs with diffs + linked issue  (candidates.json)
  validate  clone@base, build, run tests before/after gold      (validated.json)
  build     emit run_benchmark-compatible tasks                 (tasks_real.json)

This file only implements harvest + the shared model; validate/build land next
once harvest output is inspected on one repo.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from typing import Any

# ---- repo registry: url, pinned "near" commit (for reference), toolchain id --

# Broad multi-language registry for the forensics bench. Concentrated in the
# five toolchains already proven by tasks_v1 (python/rust/go/java/ts) so
# validation reuses working build recipes; breadth comes from more repos per
# language, each with clean issue/PR hygiene. `toolchain` selects the
# per-language build+test recipe in TOOLCHAINS.
REPOS = {
    # python (venv + pytest)
    "flask": {"owner": "pallets", "name": "flask", "lang": "python", "toolchain": "python-pip"},
    "requests": {"owner": "psf", "name": "requests", "lang": "python", "toolchain": "python-pip"},
    "click": {"owner": "pallets", "name": "click", "lang": "python", "toolchain": "python-pip"},
    "werkzeug": {"owner": "pallets", "name": "werkzeug", "lang": "python", "toolchain": "python-pip"},
    "jinja": {"owner": "pallets", "name": "jinja", "lang": "python", "toolchain": "python-pip"},
    # rust (cargo)
    "serde": {"owner": "serde-rs", "name": "serde", "lang": "rust", "toolchain": "rust-cargo"},
    "tokio": {"owner": "tokio-rs", "name": "tokio", "lang": "rust", "toolchain": "rust-cargo"},
    "clap": {"owner": "clap-rs", "name": "clap", "lang": "rust", "toolchain": "rust-cargo"},
    "anyhow": {"owner": "dtolnay", "name": "anyhow", "lang": "rust", "toolchain": "rust-cargo"},
    # go (go test)
    "hugo": {"owner": "gohugoio", "name": "hugo", "lang": "go", "toolchain": "go-test"},
    "cobra": {"owner": "spf13", "name": "cobra", "lang": "go", "toolchain": "go-test"},
    "viper": {"owner": "spf13", "name": "viper", "lang": "go", "toolchain": "go-test"},
    # java (maven)
    "gson": {"owner": "google", "name": "gson", "lang": "java", "toolchain": "java-maven"},
    "jackson-databind": {"owner": "FasterXML", "name": "jackson-databind", "lang": "java", "toolchain": "java-maven"},
    # typescript / javascript (pnpm + vitest/jest)
    "zod": {"owner": "colinhacks", "name": "zod", "lang": "typescript", "toolchain": "ts-pnpm"},
    "date-fns": {"owner": "date-fns", "name": "date-fns", "lang": "typescript", "toolchain": "ts-pnpm"},
    "dayjs": {"owner": "iamkun", "name": "dayjs", "lang": "typescript", "toolchain": "ts-pnpm"},
}

# Per-language build recipe: `setup` runs once after checkout (before the
# agent), `test_runner` names the framework so the validator can extract and
# run the PR's specific FAIL_TO_PASS tests. Filled in / refined during the
# validate phase against each pinned base_commit.
TOOLCHAINS = {
    "python-pip": {
        # uv, not raw venv: it pins a stable interpreter (the system python may
        # be a bleeding-edge build with broken ensurepip/pip, and old repo deps
        # will not build on it anyway) and installs fast and reliably.
        # Install the repo only; its test EXTRA (below) pins the framework
        # version the tests were written against. Forcing latest pytest breaks
        # repos whose conftest uses internals removed in newer pytest
        # (flask: _pytest.monkeypatch.notset, gone in pytest > 8.4).
        "setup": [["uv", "venv", "--python", "3.12", "--clear", ".venv"],
                  ["uv", "pip", "install", "--python", ".venv/bin/python3", "-q", "-e", "."]],
        "test_runner": "pytest",
    },
    "rust-cargo": {"setup": [], "test_runner": "cargo"},
    "go-test": {"setup": [], "test_runner": "gotest"},
    "java-maven": {"setup": [], "test_runner": "maven"},
    "ts-pnpm": {"setup": [["pnpm", "install", "--frozen-lockfile", "--prefer-offline"]],
                "test_runner": "vitest"},
}

# A path is a TEST file (vs. source) — used to split the PR diff into the
# test_patch (applied before the agent runs) and the gold_patch (the reference
# fix, never shown to the agent). Conservative: when unsure, treat as source.
TEST_PATH_RE = {
    "python": re.compile(r"(^|/)(tests?/|conftest\.py$|test_[^/]+\.py$|[^/]+_test\.py$)"),
    "java": re.compile(r"(^|/)src/test/|[^/]+Test\.java$|[^/]+Tests\.java$"),
    "go": re.compile(r"[^/]+_test\.go$"),
    "rust": re.compile(r"(^|/)tests/"),
    "typescript": re.compile(r"\.(test|spec)\.(ts|tsx|js)$|(^|/)tests?/"),
}


# Bookkeeping files that ride along in a PR but are neither the fix nor a
# test: changelogs, docs, lockfiles. The agent is not asked to touch these,
# so they count as neither source nor test — a PR whose only "source" is a
# changelog entry is not a code task.
IGNORE_PATH_RE = re.compile(
    r"(^|/)(CHANGES|CHANGELOG|HISTORY|AUTHORS|NEWS)[^/]*$"
    r"|\.(rst|md|txt|lock)$"
    r"|(^|/)docs?/"
    r"|(^|/)\.github/"
    r"|(^|/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock)$",
    re.IGNORECASE,
)


def is_test_path(lang: str, path: str) -> bool:
    rx = TEST_PATH_RE.get(lang)
    return bool(rx and rx.search(path))


def is_ignored_path(path: str) -> bool:
    return bool(IGNORE_PATH_RE.search(path))


def sh(args: list[str], cwd: str | None = None, check: bool = True) -> str:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:6])} -> rc={r.returncode}\n{r.stderr[:400]}")
    return r.stdout


def gh_json(path: str, attempts: int = 4) -> Any:
    # Resilient to transient gh/network failures: a single blip must not
    # abort a broad multi-repo harvest. Linear backoff; re-raise only after
    # exhausting attempts.
    last = None
    for i in range(attempts):
        try:
            return json.loads(sh(["gh", "api", "-H", "Accept: application/vnd.github+json", path]))
        except RuntimeError as e:
            last = e
            msg = str(e)
            # rate-limit / auth errors are not transient — fail fast
            if "rate limit" in msg.lower() or "bad credentials" in msg.lower():
                raise
            import time as _t
            _t.sleep(3 * (i + 1))
    raise last  # type: ignore[misc]


ISSUE_LINK_RE = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\b[:\s]+#(\d+)", re.IGNORECASE
)


@dataclass
class Candidate:
    repo: str
    lang: str
    pr_number: int
    title: str
    merge_commit: str
    base_commit: str            # M^1: pre-PR state
    issue_number: int
    issue_title: str
    issue_body: str
    changed_source: list[str] = field(default_factory=list)
    changed_tests: list[str] = field(default_factory=list)
    # diffs are fetched during validate (they need a local clone), not here


def linked_issue(repo_cfg: dict, pr: dict) -> int | None:
    """The issue this PR closes, from body keywords or the timeline. Real,
    issue-driven work only — a PR that closes nothing is dropped."""
    body = pr.get("body") or ""
    m = ISSUE_LINK_RE.search(body)
    if m:
        return int(m.group(1))
    # timeline cross-reference fallback (closed-via events)
    owner, name = repo_cfg["owner"], repo_cfg["name"]
    try:
        events = gh_json(f"/repos/{owner}/{name}/issues/{pr['number']}/timeline?per_page=100")
    except RuntimeError:
        return None
    for ev in events:
        if ev.get("event") == "connected" and ev.get("source", {}).get("issue"):
            iss = ev["source"]["issue"]
            if "pull_request" not in iss:
                return int(iss["number"])
    return None


def harvest_repo(repo: str, want: int, scan: int, args: argparse.Namespace) -> list[Candidate]:
    cfg = REPOS[repo]
    owner, name, lang = cfg["owner"], cfg["name"], cfg["lang"]
    out: list[Candidate] = []
    page = 1
    seen = 0
    while len(out) < want and seen < scan:
        prs = gh_json(
            f"/repos/{owner}/{name}/pulls?state=closed&per_page=50&page={page}"
            "&sort=updated&direction=desc"
        )
        if not prs:
            break
        for pr in prs:
            if len(out) >= want or seen >= scan:
                break
            if not pr.get("merged_at"):
                continue
            if args.since and pr["merged_at"] < args.since:
                continue
            seen += 1
            issue_no = linked_issue(cfg, pr)
            if not issue_no:
                continue
            merge = pr.get("merge_commit_sha")
            if not merge:
                continue
            # first parent = pre-PR base state (works for merge & squash)
            try:
                commit = gh_json(f"/repos/{owner}/{name}/commits/{merge}")
                parents = commit.get("parents", [])
                if not parents:
                    continue
                base = parents[0]["sha"]
            except RuntimeError:
                continue
            # split the PR's files into test vs source
            try:
                files = gh_json(f"/repos/{owner}/{name}/pulls/{pr['number']}/files?per_page=100")
            except RuntimeError:
                continue
            tests, source = [], []
            for f in files:
                fn = f["filename"]
                if is_ignored_path(fn):
                    continue  # changelog/docs/lockfile: neither fix nor test
                if is_test_path(lang, fn):
                    tests.append(fn)
                elif f["status"] != "removed":
                    source.append(fn)
            # Must touch BOTH real source and tests, and stay focused enough
            # for one agent turn: a 30-file refactor is not a benchmark task.
            if not tests or not source:
                continue
            if len(source) > args.max_source or len(tests) > args.max_tests:
                continue
            try:
                issue = gh_json(f"/repos/{owner}/{name}/issues/{issue_no}")
            except RuntimeError:
                continue
            if "pull_request" in issue:  # the "issue" is actually a PR
                continue
            out.append(
                Candidate(
                    repo=repo, lang=lang, pr_number=pr["number"], title=pr["title"],
                    merge_commit=merge, base_commit=base,
                    issue_number=issue_no,
                    issue_title=issue.get("title", ""),
                    issue_body=(issue.get("body") or "").strip(),
                    changed_source=sorted(source), changed_tests=sorted(tests),
                )
            )
        page += 1
    return out


def cmd_harvest(args: argparse.Namespace) -> int:
    repos = args.repos.split(",") if args.repos else list(REPOS)
    all_c: list[dict] = []
    # Incremental, crash-safe: write after every repo so a late failure never
    # loses earlier work, and a re-run resumes repos not yet in the file.
    done = set()
    if args.resume:
        try:
            all_c = json.load(open(args.out))["candidates"]
            done = {c["repo"] for c in all_c}
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
    for repo in repos:
        if repo in done:
            print(f"{repo:16s} übersprungen (schon geerntet)", file=sys.stderr)
            continue
        try:
            cands = harvest_repo(repo, args.per_repo, args.scan, args)
        except Exception as e:  # one repo's failure must not sink the corpus
            print(f"{repo:16s} FEHLER, übersprungen: {str(e)[:80]}", file=sys.stderr)
            continue
        print(f"{repo:16s} {len(cands):3d} kandidaten (aus {args.scan} PRs, {REPOS[repo]['lang']})", file=sys.stderr)
        all_c.extend(asdict(c) for c in cands)
        json.dump({"candidates": all_c}, open(args.out, "w"), indent=1)
    print(f"-> {len(all_c)} kandidaten gesamt in {args.out}", file=sys.stderr)
    return 0


# ------------------------------- validate --------------------------------
#
# A candidate becomes a task ONLY if it mechanically validates against the
# pinned base_commit: build succeeds, the PR's test files FAIL before the fix
# and PASS after it. SWE-bench discipline the owner insisted on — nothing
# trusted on faith. Runs locally (all five toolchains present). pytest and
# cargo are implemented; other frameworks report "unsupported" and skip.

import os
import shutil  # noqa: F401  (reserved for worktree cleanup paths)
import tempfile


def git(args: list[str], cwd: str, timeout: int = 900, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])} -> {r.returncode}: {r.stderr[:200]}")
    return r


def clone_at_base(cand: dict, workdir: str) -> str:
    """Clone the repo and check out base_commit; return the repo path. Both
    base and merge commits are fetched so patches can be diffed locally."""
    cfg = REPOS[cand["repo"]]
    url = f"https://github.com/{cfg['owner']}/{cfg['name']}"
    path = os.path.join(workdir, cfg["name"])
    git(["clone", "--quiet", url, path], cwd=workdir)
    for sha in (cand["base_commit"], cand["merge_commit"]):
        git(["fetch", "--quiet", "origin", sha], cwd=path, check=False)
    git(["checkout", "--quiet", "--force", cand["base_commit"]], cwd=path)
    return path


def split_patches(repo: str, base: str, merge: str, lang: str) -> tuple[str, str]:
    """Unified diff base..merge, split into (test_patch, gold_patch) by path.
    Bookkeeping files are dropped from both."""
    names = git(["diff", "--name-only", f"{base}..{merge}"], cwd=repo).stdout.split()
    test_files = [f for f in names if not is_ignored_path(f) and is_test_path(lang, f)]
    gold_files = [f for f in names if not is_ignored_path(f) and not is_test_path(lang, f)]
    test_patch = git(["diff", f"{base}..{merge}", "--", *test_files], cwd=repo).stdout if test_files else ""
    gold_patch = git(["diff", f"{base}..{merge}", "--", *gold_files], cwd=repo).stdout if gold_files else ""
    return test_patch, gold_patch


def apply_patch(repo: str, patch: str) -> bool:
    if not patch.strip():
        return True
    p = subprocess.run(["git", "apply", "--3way", "-"], cwd=repo, input=patch,
                       capture_output=True, text=True)
    return p.returncode == 0


def run_setup(repo: str, toolchain: str) -> bool:
    for cmd in TOOLCHAINS[toolchain]["setup"]:
        r = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            return False
    if toolchain == "python-pip":
        py = ".venv/bin/python3"

        def uvpip(*args: str, check: bool = False) -> bool:
            r = subprocess.run(["uv", "pip", "install", "--python", py, "-q", *args],
                               cwd=repo, capture_output=True, text=True, timeout=1800)
            return r.returncode == 0

        # Repo test deps: pinned in optional-dependency groups (names vary) or
        # requirements files. Install ALL declared extras best-effort — a test
        # can need any of them (flask's test_async_view needs the [async] extra
        # for asgiref; without it the test errors on a missing dep, not the bug,
        # and never flips). This brings the pinned framework + all test deps.
        extras = set()
        pp = os.path.join(repo, "pyproject.toml")
        if os.path.exists(pp):
            import tomllib
            try:
                data = tomllib.load(open(pp, "rb"))
                extras |= set(data.get("project", {}).get("optional-dependencies", {}))
            except (tomllib.TOMLDecodeError, OSError):
                pass
        extras |= {"test", "tests", "dev", "testing", "async"}
        got_extra = any(uvpip("-e", f".[{x}]") for x in sorted(extras))
        for req in ("requirements/tests.txt", "requirements/test.txt",
                    "test-requirements.txt", "requirements-test.txt", "requirements-dev.txt"):
            if os.path.exists(os.path.join(repo, req)):
                got_extra = uvpip("-r", req) or got_extra
        # Ensure a test runner + JSON reporter exist. If the repo pinned pytest,
        # this adds only the reporter (uv resolves a compatible version); if it
        # pinned nothing, pytest comes in too.
        # Pin the framework to the era of the harvested PRs (merged >= 2024):
        # pytest 9 removed internals conftests of that era rely on
        # (_pytest.monkeypatch.notset) and pytest-json-report does not support
        # it. `pytest<9` is the compatible band for 2024-2025 repos.
        _ = got_extra
        uvpip("pytest>=8,<9", "pytest-json-report")
    return True


def run_tests(repo: str, toolchain: str, test_files: list[str]) -> dict[str, bool] | None:
    """Return {test_id: passed}. None = framework not yet supported / harness
    error (candidate then skipped, never silently counted)."""
    runner = TOOLCHAINS[toolchain]["test_runner"]
    if runner == "pytest":
        py = os.path.join(repo, ".venv/bin/python3")
        py = py if os.path.exists(py) else "python3"
        rep = os.path.join(repo, ".pytest_report.json")
        subprocess.run(
            [py, "-m", "pytest", "-p", "no:cacheprovider", "--json-report",
             f"--json-report-file={rep}", "-q", *test_files],
            cwd=repo, capture_output=True, text=True, timeout=1200,
        )
        if not os.path.exists(rep):
            return None  # pytest-json-report missing or collection crashed
        data = json.load(open(rep))
        return {t["nodeid"]: t["outcome"] == "passed" for t in data.get("tests", [])}
    if runner == "cargo":
        r = subprocess.run(["cargo", "test", "--no-fail-fast", "--", "--format=terse"],
                           cwd=repo, capture_output=True, text=True, timeout=1800)
        out = {}
        for line in (r.stdout + r.stderr).splitlines():
            m = re.match(r"test (\S+) \.\.\. (ok|FAILED)", line)
            if m:
                out[m.group(1)] = m.group(2) == "ok"
        return out or None
    return None  # gotest / maven / vitest: TODO, skip for now


def validate_candidate(cand: dict, workdir: str) -> dict | None:
    cfg = REPOS[cand["repo"]]
    tc = cfg["toolchain"]
    repo = clone_at_base(cand, workdir)
    test_patch, gold_patch = split_patches(repo, cand["base_commit"], cand["merge_commit"], cand["lang"])
    if not test_patch or not gold_patch:
        return None
    if not run_setup(repo, tc):
        return None
    if not apply_patch(repo, test_patch):
        return None
    before = run_tests(repo, tc, cand["changed_tests"])
    if before is None:
        return None
    if not apply_patch(repo, gold_patch):
        return None
    after = run_tests(repo, tc, cand["changed_tests"])
    if after is None:
        return None
    fail_to_pass = sorted(t for t, ok in after.items() if ok and not before.get(t, False))
    pass_to_pass = sorted(t for t, ok in after.items() if ok and before.get(t, False))
    if not fail_to_pass:
        return None  # the PR's tests do not actually flip: not a real, gated task
    return {
        **cand,
        "test_patch": test_patch,
        "gold_patch": gold_patch,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
        "toolchain": tc,
        "setup_commands": TOOLCHAINS[tc]["setup"],
    }


def cmd_validate(args: argparse.Namespace) -> int:
    cands = json.load(open(args.candidates))["candidates"]
    validated: list[dict] = []
    done_prs = set()
    if args.resume:
        try:
            validated = json.load(open(args.out))["tasks"]
            done_prs = {(t["repo"], t["pr_number"]) for t in validated}
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
            pass
    for cand in cands:
        key = (cand["repo"], cand["pr_number"])
        if key in done_prs:
            continue
        if args.repos and cand["repo"] not in args.repos.split(","):
            continue
        with tempfile.TemporaryDirectory(prefix="swebench-val-") as wd:
            try:
                task = validate_candidate(cand, wd)
            except Exception as e:
                print(f"  {cand['repo']}#{cand['pr_number']}: FEHLER {str(e)[:70]}", file=sys.stderr)
                task = None
        mark = "OK" if task else "verworfen"
        n_ftp = len(task["fail_to_pass"]) if task else 0
        print(f"  {cand['repo']}#{cand['pr_number']}: {mark}" + (f" ({n_ftp} FAIL_TO_PASS)" if task else ""), file=sys.stderr)
        if task:
            validated.append(task)
            json.dump({"tasks": validated}, open(args.out, "w"), indent=1)
    print(f"-> {len(validated)} validierte tasks in {args.out}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="gh API -> candidate PRs")
    h.add_argument("--repos", default="", help="comma list; default all")
    h.add_argument("--per-repo", type=int, default=12, help="candidates to keep per repo")
    h.add_argument("--scan", type=int, default=300, help="merged PRs to scan per repo")
    h.add_argument("--since", default="2024-01-01T00:00:00Z", help="only PRs merged after this ISO date")
    h.add_argument("--max-source", type=int, default=3, help="max non-doc source files")
    h.add_argument("--max-tests", type=int, default=3, help="max test files")
    h.add_argument("--resume", action="store_true", help="skip repos already in --out")
    h.add_argument("--out", default="candidates_real.json")
    h.set_defaults(func=cmd_harvest)

    v = sub.add_parser("validate", help="clone@base, build, run tests before/after gold")
    v.add_argument("--candidates", required=True)
    v.add_argument("--repos", default="", help="comma list to validate; default all in candidates")
    v.add_argument("--resume", action="store_true", help="skip PRs already in --out")
    v.add_argument("--out", default="tasks_real.json")
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
