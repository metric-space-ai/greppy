#!/usr/bin/env python3
"""Multi-repo, multi-language A/B token+time benchmark for grepplus.

For every task in ``tasks_v2.json`` a minimal pi.dev agent (default model
MiniMax-M3) answers the question about the task's corpus repo:

  * grep-agent      -- restricted to /usr/bin/grep + plain file viewing,
  * grepplus-agent  -- told to use the single grepplus product surface.

Additional `plus` and `explorer` agents are diagnostic ablations only.
They help identify which internal mechanism helps or hurts, but they are not
separate grepplus products and are not the product acceptance status.

It records, per task and per agent, the model's own INPUT and OUTPUT token
usage SEPARATELY, the number of tool-call loops, wall-clock, and the final
answer; rows are written to ``results.json``. ``--report`` prints an aggregate
with MEDIAN and MEAN grep/grepplus factors for INPUT and OUTPUT tokens
separately, broken down by repo-size and by task-type.

The API key is read from ``$MINIMAX_API_KEY`` at runtime, with a launchd
fallback on macOS, and is NEVER stored.
The runner is orchestrator-friendly: no key in argv, deterministic task order,
incremental saves.

Usage:
    export MINIMAX_API_KEY=sk-...
    python3 bench/agent_efficiency/run_bench.py                 # run all tasks
    python3 bench/agent_efficiency/run_bench.py t001 t042       # run a subset
    python3 bench/agent_efficiency/run_bench.py --repo go_small # one repo
    python3 bench/agent_efficiency/run_bench.py --agents grep --save-raw t001
    python3 bench/agent_efficiency/run_bench.py --agents plus --save-raw  # diagnostic ablation only
    python3 bench/agent_efficiency/run_bench.py --results /tmp/results.json
    python3 bench/agent_efficiency/run_bench.py --results /tmp/results.json --rerun
    python3 bench/agent_efficiency/run_bench.py --report        # aggregate only

After every run, execute ``forensics.py`` for each candidate comparison. The
aggregate report is not an acceptance gate: speed wins without machine-readable
quality evidence remain optimization hints only.
"""
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXT = str(HERE / "minimax-provider.js")

# Contract model for all gate decisions is MiniMax-M3 (BENCHMARK_CONTRACT).
PROVIDERS = {
    "minimax": {"ext": str(HERE / "minimax-provider.js"), "model": "MiniMax-M3",
                "key_env": "MINIMAX_API_KEY"},
}
PROVIDER = "minimax"  # overridden by --llm-provider in main()
BIN = os.environ.get("GREPPLUS_BENCH_BIN") or str(REPO / "target" / "release" / "grepplus")
CORPUS = HERE / "corpus"
REALCORPUS = HERE / "realcorpus"
TASKS = HERE / "tasks_v2.json"
RESULTS = HERE / "results.json"
RAW_ROOT = HERE / "raw_runs"
PROMPT_USAGE_KEYS = ("input", "cacheRead", "cacheWrite", "cacheWrite1h", "cacheWrite5m")


def ensure_provider_key(provider: str = "minimax") -> None:
    """Load the provider key from launchd if this process did not inherit it."""
    key_env = PROVIDERS[provider]["key_env"]
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


def ensure_minimax_api_key() -> None:
    """Back-compat alias used by acceptance orchestrators."""
    ensure_provider_key("minimax")


# repo -> (language, size-class). Mirrors gen_corpus.py for the synthetic v1
# corpus; the four pinned real repos (corpus v2, realcorpus/MANIFEST.json) get
# size class "real".
REPO_META = {
    "rust_medium": ("rust", "medium"),
    "python_large": ("python", "large"),
    "go_small": ("go", "small"),
    "java_medium": ("java", "medium"),
    "js_small": ("javascript", "small"),
    "ts_large": ("typescript", "large"),
    # corpus v2 real repos (pinned clones under realcorpus/, see MANIFEST.json)
    "serde": ("rust", "real"),
    "flask": ("python", "real"),
    "gson": ("java", "real"),
    "zod": ("ts", "real"),
}

# Real-repo roots resolve to realcorpus/<repo>; synthetic ones to corpus/<repo>.
REAL_REPOS = {"serde", "flask", "gson", "zod"}


def repo_root(repo: str) -> str:
    if repo in REAL_REPOS:
        return str(REALCORPUS / repo)
    return str(CORPUS / repo)

GREP_SYS = (
    "You are answering a question about the code in this repository. You may use "
    "the shell ONLY for /usr/bin/grep and plain file viewing (cat/head/sed/wc), "
    "plus the read tool. You do NOT have grepplus, ripgrep, ctags, or any code "
    "index -- work like a plain-grep user. Be efficient: stop as soon as you can "
    "answer. End with the final answer."
)

# Realistic baseline: a normal coding agent with grep + read and NO efficiency
# coaching — how an agent without code-intelligence actually explores code.
EXPLORER_SYS = (
    "You are a coding agent answering a question about this repository. You have "
    "the shell (grep, cat, sed, head, find, wc) and a read tool. Explore the code "
    "as needed to give a thorough, correct answer. End with the final answer."
)


def gp_sys(root: str) -> str:
    """grepplus-agent system prompt. Kept deliberately CONCISE (a realistic
    tool description, not a tutorial): an early, over-verbose version of this
    prompt was ~530 tokens and, re-sent every turn with no prompt caching, was
    itself a large share of the agent's input tokens — penalising grepplus on
    cheap tasks where it should at least tie grep. It still names the
    code-returning queries (`context`, `--code`) and the graph commands, and
    carries the generalisable anti-flood guidance the forensics surfaced
    (trust returned code; don't reach for --all; summarise the 'N more' count).
    """
    g = f"{BIN}"
    return (
        f"Answer the question about the code at {root} using the grepplus "
        f"code-intelligence CLI ({g}), always with `--root {root}`. It returns "
        f"real source, not just pointers — prefer it over reading files.\n"
        f"- context \"X\"      : source of the most relevant definitions (trust "
        f"it; don't re-open the file to confirm). For a CONCEPTUAL / "
        f"behavioural question — 'which function does X', 'where is the logic "
        f"that …', 'what handles …' — pass the WHOLE natural-language phrase "
        f"as X: `context` matches meaning, not just words (native vector "
        f"similarity), so it finds the routine even when the code uses "
        f"different vocabulary than your question — run `context` with the "
        f"phrase FIRST, do not pre-guess symbol names. Fallback: search-code "
        f"\"X\".\n"
        f"- who-calls SYM / callees SYM / find-usages SYM : callers / callees / "
        f"references as file:line, resolved from the code graph — the "
        f"AUTHORITATIVE set, not a text match. Add --code for ONE symbol's "
        f"bodies only. The trailing `— N …` line is the exact count and IS the "
        f"complete answer: do not re-run, pass --all, or cross-check with grep "
        f"(only the `— N more … of T total` form was capped — it already states "
        f"the full T).\n"
        f"- brief SYM : definition (with source) + direct callers + direct "
        f"callees in ONE call. For any 'how does SYM work / what is its role / "
        f"what depends on it' question, run `brief` ONCE — do not iterate "
        f"context+who-calls+callees separately.\n"
        f"- impact SYM [--direction incoming|outgoing] : the WHOLE transitive "
        f"set in ONE call (incoming = everything that breaks if you change SYM; "
        f"outgoing = everything SYM reaches). For any 'what breaks / what does X "
        f"reach / transitive' question, run `impact` ONCE.\n"
        f"- path --from A --to B : the ORDERED call chain from A to B in ONE "
        f"call. For any 'trace how A reaches B / give the chain of calls in "
        f"order / path from A to B' question, run `path` ONCE — do NOT "
        f"hand-walk who-calls. A and B may be bare or Owner.method names.\n"
        f"- search-symbols NAME, stats.\n"
        f"Be efficient: prefer ONE impact/context/who-calls call over many. "
        f"Stop as soon as you can answer. End with the final answer."
    )


def plus_sys(root: str) -> str:
    g = f"{BIN}"
    return (
        f"Answer the question about the code at {root} using grepplus ({g}), "
        f"always with `--root {root}`. Treat `plus` as a better grep, not as "
        f"RAG and not as an answer generator.\n"
        f"- plus \"QUERY\" --k 3 : precision-first grep-like fused search. It prints "
        f"`file:line:snippet` rows, using literal/full-text plus fuzzy/symbol/"
        f"graph/vector ranking internally. Use it to find the right files, "
        f"symbols, and lines faster than plain grep.\n"
        f"- After `plus` finds candidate code, read or inspect only the relevant "
        f"span/file you need. You must still interpret the code yourself.\n"
        f"- For exact graph facts, use who-calls/callees/find-usages/path/impact "
        f"when you know the symbol. The 'N more … of T total' footer contains "
        f"exact counts; do not pass --all unless every row is truly required.\n"
        f"- Use context/brief only when reading the definition/body is needed; "
        f"do not treat them as final answers.\n"
        f"Be efficient: use `plus` to avoid exploratory grep/read loops, then "
        f"perform the minimum exact read or graph command needed. End with the "
        f"final answer."
    )


# Per-session rate-limit backoff. The MiniMax "Token Plan rate limit reached"
# 429 (code 2062) is a CONCURRENCY/window limit, NOT quota exhaustion — the
# 2026-07-02 P1 run hit it with the 5h plan window only 53% used, because
# ~20 concurrent MiniMax streams were too many. pi's built-in 3x retry with
# 2/4/8s backoff cannot ride out a rate-limit window; a rate-limited session
# dies in milliseconds, so failed attempts cost almost nothing. run_pi retries
# the WHOLE session with these sleeps before accepting the errored result.
RATE_LIMIT_BACKOFFS_S = (45, 90, 180)


def run_pi(
    system: str,
    question: str,
    cwd: str,
    timeout: int = 240,
    raw_path: pathlib.Path | None = None,
) -> dict:
    """Drive the pi.dev agent once. Returns separated input/output token usage,
    tool-call count, wall-clock, answer, and any error.

    Rate-limit-aware: when the session comes back INVALID with a rate-limit/
    429/quota error, the whole session is retried up to
    len(RATE_LIMIT_BACKOFFS_S) times with 45/90/180s sleeps (logged to
    stderr). Only after the last backoff is the errored result returned. The
    attempt count is recorded in the returned row and in the raw .meta.json.
    wall_s measures the FINAL attempt only — backoff sleeps are harness
    throttling, never agent time.

    When `raw_path` is set, the exact pi JSONL stdout of the final attempt is
    written before parsing. This is required for path forensics: the aggregate
    row cannot reconstruct which grep/read commands the model actually ran.
    """
    cmd = [
        "pi", "-p", "--extension", PROVIDERS[PROVIDER]["ext"],
        "--provider", PROVIDER,
        "--model", PROVIDERS[PROVIDER]["model"], "--mode", "json", "--no-session",
        "--thinking", "off", "--tools", "bash,read",
        "--append-system-prompt", system, question,
    ]
    attempts = 0
    for backoff_s in RATE_LIMIT_BACKOFFS_S + (None,):
        attempts += 1
        t0 = time.time()
        try:
            p = subprocess.run(
                cmd, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
            )
            out = p.stdout.decode("utf-8", "replace")
            return_code = p.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            return_code = None
        wall = time.time() - t0
        parsed = parse_pi_jsonl(out)
        if agent_valid(parsed) or not is_quota_error(parsed.get("error")):
            break
        if backoff_s is None:
            break  # backoffs exhausted -- return the errored result
        print(
            f"   rate-limited, backing off {backoff_s}s "
            f"(attempt {attempts}/{len(RATE_LIMIT_BACKOFFS_S) + 1})",
            file=sys.stderr,
        )
        time.sleep(backoff_s)
    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(out, encoding="utf-8")
        meta = {
            "cwd": cwd,
            "timeout_s": timeout,
            "return_code": return_code,
            "wall_s": round(wall, 3),
            "attempts": attempts,
            "question": question,
            "raw_jsonl": raw_path.name,
        }
        raw_path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    parsed.update({
        "wall_s": round(wall, 1),
        "return_code": return_code,
        "attempts": attempts,
    })
    return parsed


def parse_pi_jsonl(out: str) -> dict:
    inp = outp = tools = 0
    cache_read = cache_write = cache_write_1h = 0
    ctx_chars = 0  # total chars of tool-result content the agent had to ingest
    turn_prompt_inputs: list[int] = []
    turn_reported_inputs: list[int] = []
    turn_outputs: list[int] = []
    answer = ""
    err = None
    for line in out.splitlines():
        try:
            o = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        # The agent's *search-context burden* — the bytes it had to read back
        # from grep / file-reads / grepplus to answer — is the cleanest measure
        # of the user's intent ("tokens an agent burns to FIND the code"). It is
        # free of the two confounds that wreck the model-reported `input` count:
        # (1) pi's fixed ~3.4K-token base system prompt, identical for both
        # agents, which compresses the input ratio toward 1.0, and (2) MiniMax
        # prompt-caching, whose warmth varies run-to-run. We sum every tool
        # result's content length (deterministic given the tool output).
        if o.get("type") == "turn_end":
            for tr in o.get("toolResults", []) or []:
                for c in tr.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "text":
                        ctx_chars += len(c.get("text", ""))
            m = o.get("message", {})
            u = m.get("usage", {}) or {}
            turn_input = int(u.get("input", 0) or 0)
            turn_output = int(u.get("output", 0) or 0)
            turn_cache_read = int(u.get("cacheRead", 0) or 0)
            turn_cache_write = int(u.get("cacheWrite", 0) or 0)
            turn_cache_write_1h = int(u.get("cacheWrite1h", 0) or 0)
            turn_prompt_input = sum(int(u.get(k, 0) or 0) for k in PROMPT_USAGE_KEYS)
            inp += turn_input
            outp += turn_output
            cache_read += turn_cache_read
            cache_write += turn_cache_write
            cache_write_1h += turn_cache_write_1h
            turn_reported_inputs.append(turn_input)
            turn_outputs.append(turn_output)
            turn_prompt_inputs.append(turn_prompt_input)
            tools += len(o.get("toolResults", []) or [])
            txt = "".join(
                c.get("text", "") for c in m.get("content", [])
                if c.get("type") == "text"
            )
            if txt.strip():
                answer = txt
            if m.get("errorMessage"):
                err = m["errorMessage"]
    prompt_input = sum(turn_prompt_inputs)
    first_prompt_input = turn_prompt_inputs[0] if turn_prompt_inputs else 0
    loop_prompt_input = sum(turn_prompt_inputs[1:])
    first_reported_input = turn_reported_inputs[0] if turn_reported_inputs else 0
    loop_reported_input = sum(turn_reported_inputs[1:])
    return {
        "input": inp, "output": outp, "total": inp + outp,
        "cache_read": cache_read, "cache_write": cache_write,
        "cache_write_1h": cache_write_1h,
        "prompt_input": prompt_input,
        "first_turn_prompt_input": first_prompt_input,
        "loop_prompt_input": loop_prompt_input,
        "variable_input": loop_prompt_input,
        "first_turn_input": first_reported_input,
        "loop_input": loop_reported_input,
        "turns": len(turn_prompt_inputs),
        "ctx_chars": ctx_chars, "ctx_tok": round(ctx_chars / 4),
        "tool_calls": tools,
        "answer": answer.strip(), "error": err,
    }


def enrich_metrics_from_raw(row: dict, agent: str) -> None:
    result = row.get(agent)
    if not isinstance(result, dict):
        return
    if result.get("prompt_input") is not None and result.get("variable_input") is not None:
        return
    raw_paths = row.get("raw_paths", {})
    if not isinstance(raw_paths, dict):
        return
    raw_path = raw_paths.get(agent)
    if not raw_path:
        return
    path = pathlib.Path(raw_path)
    if not path.exists():
        return
    parsed = parse_pi_jsonl(path.read_text(encoding="utf-8", errors="replace"))
    for key in (
        "cache_read",
        "cache_write",
        "cache_write_1h",
        "prompt_input",
        "first_turn_prompt_input",
        "loop_prompt_input",
        "variable_input",
        "first_turn_input",
        "loop_input",
        "turns",
    ):
        result[key] = parsed.get(key)


def load_rows(results_path: pathlib.Path) -> dict:
    if results_path.exists():
        return {r["id"]: r for r in json.load(open(results_path))}
    return {}


def save_rows(by: dict, results_path: pathlib.Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(list(by.values()), open(results_path, "w"), indent=2)


# --------------------------------------------------------------------------
# session validity + quota circuit breaker
# --------------------------------------------------------------------------
# Consecutive quota-errored agent sessions after which a run aborts (exit 3).
# This is the FINAL BACKSTOP for sustained provider failure: every session
# that still errors already sat out the in-session 45/90/180s rate-limit
# backoffs (RATE_LIMIT_BACKOFFS_S), i.e. 8 consecutive dead sessions means
# 32 attempts across ~40 min of backoff — not a transient rate-limit window.
# On 2026-07-02 the harness burned 282 sessions against a rate-limited
# MiniMax plan (429 rate_limit_error 2062) and still reported every step as
# exit 0; per-session 2/4/8s pi retries could not outlast the window.
QUOTA_BREAKER_LIMIT = 8
QUOTA_ERROR_RE = re.compile(r"rate[ _-]?limit|\b429\b|quota", re.IGNORECASE)


def is_quota_error(error) -> bool:
    return bool(error and QUOTA_ERROR_RE.search(str(error)))


def agent_valid(result) -> bool:
    """A results row is INVALID for an agent when the session errored
    (rate limit, provider failure, ...) or produced neither an answer nor a
    single tool call: its ctx/output/wall numbers measure a dead session, not
    the tool, and must never enter factor aggregates."""
    if not isinstance(result, dict):
        return False
    if result.get("error"):
        return False
    if not str(result.get("answer") or "").strip() and not result.get("tool_calls"):
        return False
    return True


def agent_status_line(result: dict) -> str:
    """Per-task stderr status: real numbers for a valid session, an explicit
    ERROR(...) marker for a dead one (never ctx=0 masquerading as data)."""
    if not agent_valid(result):
        err = str(result.get("error") or "empty answer, 0 tool calls")
        err = " ".join(err.split())
        return f"ERROR({err[:60]})"
    return (f"ctx={result['ctx_tok']:>6} out={result['output']:>5} "
            f"in={result['input']:>6} {result['tool_calls']:>2} calls "
            f"{result['wall_s']:>5}s")


# --------------------------------------------------------------------------
# aggregate reporter
# --------------------------------------------------------------------------
def _factors(rows, field, baseline="grep"):
    """baseline/grepplus per-task factors for a field (skip zero divisors and
    pairs where either side is an errored/dead session)."""
    out = []
    for r in rows:
        b_row, p_row = r.get(baseline), r.get("grepplus")
        if not agent_valid(b_row) or not agent_valid(p_row):
            continue
        b = b_row.get(field, 0)
        p = p_row.get(field, 0)
        if p:
            out.append(b / p)
    return out


def _cost_factors(rows, cost_fn, baseline="grep"):
    out = []
    for r in rows:
        b_row, p_row = r.get(baseline), r.get("grepplus")
        if not agent_valid(b_row) or not agent_valid(p_row):
            continue
        b = cost_fn(b_row)
        p = cost_fn(p_row)
        if p:
            out.append(b / p)
    return out


def weighted_variable_cost(agent_row: dict, output_weight: float = 4.0) -> float:
    return float(agent_row.get("variable_input", 0) or 0) + output_weight * float(
        agent_row.get("output", 0) or 0
    )


def weighted_raw_cost(agent_row: dict, output_weight: float = 4.0) -> float:
    return float(agent_row.get("input", 0) or 0) + output_weight * float(
        agent_row.get("output", 0) or 0
    )


def _fmt(vals):
    if not vals:
        return "   n/a"
    return f"{statistics.median(vals):5.2f}x med / {statistics.mean(vals):5.2f}x mean"


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def _gate_h(vals):
    """Gate-H Pflichtdimensionen: median, mean, p90/p95, best/worst, n>=10x."""
    if not vals:
        return "n/a"
    s = sorted(vals)
    return (f"{statistics.median(s):6.2f}x med {statistics.mean(s):6.2f}x mean "
            f"p90={_pct(s, 90):6.2f}x p95={_pct(s, 95):6.2f}x "
            f"best={s[-1]:.2f}x worst={s[0]:.2f}x "
            f">=10x: {sum(1 for v in s if v >= 10)}/{len(s)}")


def _joint_10x(subset, base):
    """Tasks >=10x on ctx tokens AND wall-clock jointly vs `base`."""
    hit = tot = 0
    for r in subset:
        b, p = r.get(base), r.get("grepplus")
        if not agent_valid(b) or not agent_valid(p):
            continue
        if not p.get("ctx_tok") or not p.get("wall_s"):
            continue
        tot += 1
        if (b.get("ctx_tok", 0) / p["ctx_tok"] >= 10
                and b.get("wall_s", 0) / p["wall_s"] >= 10):
            hit += 1
    return hit, tot


def default_task_classes_path(tasks_path: pathlib.Path) -> pathlib.Path:
    """Auto-detect the classes doc matching a tasks file: tasks_v2.json pairs
    with task_classes_v2.json (corpus v2 contract)."""
    candidate = tasks_path.parent / "task_classes_v2.json"
    if candidate.exists():
        return candidate
    return HERE / "task_classes_v2.json"


def load_task_classes(path: pathlib.Path | None = None) -> dict:
    """task id -> router/regression class (R7 contract; corpus v2 doc)."""
    path = path or (HERE / "task_classes_v2.json")
    if not path.exists():
        return {}
    id2cls = {}
    for name, spec in json.load(open(path)).get("classes", {}).items():
        for tid in spec.get("ids", []):
            id2cls[tid] = name
    return id2cls


def report(results_path: pathlib.Path,
           classes_path: pathlib.Path | None = None) -> None:
    if not results_path.exists():
        sys.exit(f"no results file at {results_path} -- run the benchmark first")
    all_rows = json.load(open(results_path))
    rows = [r for r in all_rows
            if r.get("grepplus") and (r.get("grep") or r.get("explorer"))]
    if not rows:
        sys.exit("results file has no completed grepplus rows")
    for row in rows:
        for agent in ("grep", "grepplus", "explorer"):
            enrich_metrics_from_raw(row, agent)

    id2cls = load_task_classes(classes_path)

    def block(label, subset):
        if not subset:
            return
        print(f"\n{label}  (n={len(subset)})")
        # explorer first: it is THE product baseline ("plain grep", uncoached).
        # The coached grep agent is a co-reported diagnostic row (PLAN_10X §2, R4).
        for base in ("explorer", "grep"):
            if not any(r.get(base) for r in subset):
                continue
            tag = ("PRODUCT-BASELINE grep-agent (uncoached)" if base == "explorer"
                   else "diagnostic: coached efficient-grep (not product status)")
            print(f"  vs {tag}:")
            # Aggregation honesty: a pair with an errored/dead side is not a
            # measurement. Report how many pairs actually carry data, and how
            # many were excluded, BEFORE any factor line.
            pairs = sum(1 for r in subset
                        if isinstance(r.get(base), dict)
                        and isinstance(r.get("grepplus"), dict))
            valid = sum(1 for r in subset
                        if agent_valid(r.get(base))
                        and agent_valid(r.get("grepplus")))
            excluded = pairs - valid
            print(f"    valid pairs: {valid}/{pairs} (excluded: {excluded} errored)")
            if pairs and valid / pairs < 0.7:
                bang = "!" * 70
                print(f"    {bang}")
                print(f"    !!! RUN NOT DECISION-CAPABLE: only {valid}/{pairs} "
                      f"valid pairs (<70%) — errored/")
                print(f"    !!! dead sessions dominate this block. The "
                      f"factors below are")
                print(f"    !!! NOT measurements and must not gate any "
                      f"decision.")
                print(f"    {bang}")
            print(f"    SEARCH-CONTEXT tokens (gated metric)    : "
                  f"{_gate_h(_factors(subset, 'ctx_tok', base))}")
            print(f"    LOOP PROMPT tokens (base turn removed)  : "
                  f"{_gate_h(_factors(subset, 'variable_input', base))}")
            print(f"    OUTPUT tokens                           : "
                  f"{_gate_h(_factors(subset, 'output', base))}")
            print(f"    WEIGHTED LOOP cost (input + 4*output)   : "
                  f"{_gate_h(_cost_factors(subset, weighted_variable_cost, base))}")
            print(f"    WALL-CLOCK time (gated metric)          : "
                  f"{_gate_h(_factors(subset, 'wall_s', base))}")
            print(f"    TOOL CALLS (rounds)                     : "
                  f"{_gate_h(_factors(subset, 'tool_calls', base))}")
            print(f"    RAW MODEL-INPUT tokens (diagnostic only): "
                  f"{_gate_h(_factors(subset, 'input', base))}")
            joint, jtot = _joint_10x(subset, base)
            print(f"    >=10x TOKENS AND TIME jointly           : {joint}/{jtot}")

    print("=" * 72)
    print("PRODUCT REPORT grepplus vs explorer gate baseline (higher = grepplus better)")
    print("=" * 72)
    if id2cls:
        target = [r for r in rows if id2cls.get(r["id"]) != "literal_control"]
        block(f"TARGET AGGREGATE {len(target)}/{len(rows)} (all without literal_control; "
              f"Z1/Z2 gate, PLAN_10X §2)", target)
        block(f"FULL MIX {len(rows)}/{len(rows)} (Z5: always co-reported)", rows)
        classes_name = classes_path.name if classes_path else "task_classes_v2.json"
        print("\n" + "-" * 72 + f"\nBY ROUTER CLASS ({classes_name})\n" + "-" * 72)
        seen = set()
        class_order = [c for c in id2cls.values()
                       if not (c in seen or seen.add(c))]
        for cls in class_order:
            block(f"class={cls}", [r for r in rows if id2cls.get(r["id"]) == cls])
    else:
        block("ALL TASKS (task classes doc missing -- no target aggregate!)", rows)

    print("\n" + "-" * 72 + "\nBY REPO SIZE\n" + "-" * 72)
    for size in ("small", "medium", "large", "real"):
        block(f"size={size}", [r for r in rows if r.get("size") == size])

    print("\n" + "-" * 72 + "\nBY TASK TYPE\n" + "-" * 72)
    for ttype in ("locate", "research"):
        block(f"type={ttype}", [r for r in rows if r.get("type") == ttype])

    print("\n" + "-" * 72 + "\nBY LANGUAGE\n" + "-" * 72)
    for lang in sorted({r.get("lang") for r in rows}):
        block(f"lang={lang}", [r for r in rows if r.get("lang") == lang])

    print("\n" + "=" * 72)
    if any(r.get("size") == "real" for r in rows):
        print("SCOPE: corpus v2 -- 4 real pinned repos (serde/flask/gson/zod,")
        print("realcorpus/MANIFEST.json) + synthetic v1 control tasks; floor")
        print("semantics from candidates.json. Contract model: Pi Code + MiniMax-M3.")
    else:
        print("SCOPE: synthetic 100-task LLM corpus, 6 languages; total language support:")
        print("0/159 accepted at C parity. Contract model: Pi Code + MiniMax-M3.")
    print("=" * 72)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    args = sys.argv[1:]
    results_path = RESULTS
    if "--results" in args:
        i = args.index("--results")
        results_path = pathlib.Path(args[i + 1])
        del args[i:i + 2]

    tasks_path = TASKS
    if "--tasks" in args:
        i = args.index("--tasks")
        tasks_path = pathlib.Path(args[i + 1])
        del args[i:i + 2]
        if not tasks_path.is_absolute():
            tasks_path = HERE / tasks_path
        if not tasks_path.exists():
            sys.exit(f"tasks file not found: {tasks_path}")

    classes_path = None
    if "--task-classes" in args:
        i = args.index("--task-classes")
        classes_path = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    else:
        classes_path = default_task_classes_path(tasks_path)

    if "--report" in args:
        report(results_path, classes_path)
        return

    global PROVIDER
    if "--llm-provider" in args:
        i = args.index("--llm-provider")
        PROVIDER = args[i + 1]
        if PROVIDER not in PROVIDERS:
            sys.exit(f"unknown --llm-provider {PROVIDER}; "
                     f"known: {', '.join(sorted(PROVIDERS))}")
        del args[i:i + 2]
    ensure_provider_key(PROVIDER)
    key_env = PROVIDERS[PROVIDER]["key_env"]
    if not os.environ.get(key_env):
        sys.exit(f"set {key_env}")

    repo_filter = None
    if "--repo" in args:
        i = args.index("--repo")
        repo_filter = args[i + 1]
        del args[i:i + 2]

    agents = {"grep", "grepplus"}
    if "--agents" in args:
        i = args.index("--agents")
        requested = {a.strip() for a in args[i + 1].split(",") if a.strip()}
        allowed = {"grep", "grepplus", "explorer", "plus"}
        unknown = requested - allowed
        if unknown:
            sys.exit(f"unknown --agents values: {', '.join(sorted(unknown))}")
        if not requested:
            sys.exit("--agents requires at least one of grep,grepplus,explorer,plus")
        agents = requested
        del args[i:i + 2]

    # One-product rule (R1): grep (coached, diagnostic row) and explorer
    # (uncoached = PRODUCT-BASELINE "plain grep") are baselines; grepplus is
    # the one candidate. plus is a grepplus ABLATION — research only,
    # never product status, locked behind --diagnostic.
    diagnostic = False
    if "--diagnostic" in args:
        diagnostic = True
        args.remove("--diagnostic")
    ablations = agents - {"grep", "grepplus", "explorer"}
    if ablations and not diagnostic:
        sys.exit(
            "plus is a grepplus ablation DIAGNOSTIC, never a product "
            f"(requested: {', '.join(sorted(ablations))}). Pass --diagnostic to run "
            "it for research; it must not appear in product status reports."
        )

    rerun = False
    if "--rerun" in args:
        rerun = True
        args.remove("--rerun")

    save_raw = False
    if "--save-raw" in args:
        save_raw = True
        args.remove("--save-raw")

    raw_root = None
    if "--raw-dir" in args:
        i = args.index("--raw-dir")
        raw_root = pathlib.Path(args[i + 1])
        del args[i:i + 2]
    elif save_raw:
        raw_root = RAW_ROOT / time.strftime("%Y%m%d-%H%M%S")

    ids = set(args) or None

    tasks = []
    for t in json.load(open(tasks_path)):
        if ids and t["id"] not in ids:
            continue
        if repo_filter and t["repo"] != repo_filter:
            continue
        tasks.append(t)

    by = load_rows(results_path)
    # Circuit breaker state: consecutive FRESHLY-RUN agent sessions that died
    # on a rate-limit/quota error (resumed rows do not count — they spent no
    # quota now). See QUOTA_BREAKER_LIMIT.
    consecutive_quota = 0
    agent_prompts = {
        "grep": lambda root: GREP_SYS,
        "grepplus": gp_sys,
        "plus": plus_sys,
        "explorer": lambda root: EXPLORER_SYS,
    }
    run_order = ("grep", "grepplus", "plus", "explorer")
    for t in tasks:
        repo = t["repo"]
        if repo not in REPO_META:
            sys.exit(f"{t['id']}: unknown repo {repo!r} (not in REPO_META)")
        lang, size = REPO_META[repo]
        root = repo_root(repo)
        if not pathlib.Path(root).is_dir():
            sys.exit(
                f"{t['id']}: repo root missing: {root} "
                "(run real_corpus.py setup for real repos / gen_corpus.sh for "
                "the synthetic corpus)"
            )
        print(f"== {t['id']} [{repo}/{t['type']}] {t['q'][:60]}", file=sys.stderr)

        row = by.get(t["id"], {
            "id": t["id"], "repo": repo, "lang": lang, "size": size,
            "type": t["type"], "q": t["q"], "ground_truth": t["ground_truth"],
        })
        row.update({
            "id": t["id"], "repo": repo, "lang": lang, "size": size,
            "type": t["type"], "q": t["q"], "ground_truth": t["ground_truth"],
        })
        raw_paths = row.get("raw_paths", {})

        def raw(agent: str) -> pathlib.Path | None:
            if not raw_root:
                return None
            return raw_root / t["id"] / f"{agent}.jsonl"

        def has_agent_result(agent: str) -> bool:
            # An errored/dead session is NOT a result: resuming without
            # --rerun re-runs it (that is how a quota-aborted run recovers).
            result = row.get(agent)
            return (isinstance(result, dict) and "wall_s" in result
                    and agent_valid(result))

        def persist() -> None:
            if raw_paths:
                row["raw_paths"] = raw_paths
            by[t["id"]] = row
            save_rows(by, results_path)

        for agent in run_order:
            if agent not in agents:
                continue
            label = f"{agent}:"
            if not rerun and has_agent_result(agent):
                res = row[agent]
                print(f"   {label:<10}resume existing result", file=sys.stderr)
            else:
                res = run_pi(agent_prompts[agent](root), t["q"], cwd=root,
                             raw_path=raw(agent))
                row[agent] = res
                if raw_root:
                    raw_paths[agent] = str(raw(agent))
                persist()
                if is_quota_error(res.get("error")):
                    consecutive_quota += 1
                else:
                    consecutive_quota = 0
            print(f"   {label:<10}{agent_status_line(res)}", file=sys.stderr)
            if consecutive_quota >= QUOTA_BREAKER_LIMIT:
                print(
                    f"CIRCUIT BREAKER: {consecutive_quota} consecutive "
                    "rate-limit/quota (429) agent sessions, each already "
                    "retried with 45/90/180s in-session backoffs -- this is "
                    "sustained provider failure, not a transient rate-limit "
                    "window. Aborting instead of burning every remaining "
                    f"session. Partial results saved to {results_path}; fix "
                    "the quota/concurrency and re-run to resume (errored "
                    "sessions re-run automatically).",
                    file=sys.stderr,
                )
                sys.exit(3)

        g = row.get("grep")
        p = row.get("grepplus")
        e = row.get("explorer")
        if agent_valid(g) and agent_valid(p):
            fc = g["ctx_tok"] / p["ctx_tok"] if p["ctx_tok"] else 0
            msg = f"   FACTOR    ctx vs grep={fc:.1f}x"
            if agent_valid(e):
                ec = e["ctx_tok"] / p["ctx_tok"] if p["ctx_tok"] else 0
                msg += f"  vs explorer={ec:.1f}x"
            print(msg, file=sys.stderr)
        persist()  # keep task metadata and factor-visible row fresh
    print(f"done -- run with --results {results_path} --report for aggregate", file=sys.stderr)


if __name__ == "__main__":
    main()
