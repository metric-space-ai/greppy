#!/usr/bin/env python3
"""corpus-v2 task generator (REALCORPUS_TASKGEN_SPEC.md).

Generates ``tasks_v2.json`` + ``task_classes_v2.json`` from
``realcorpus/candidates.json`` (the audited C-oracle ground truth) plus a
deterministic reuse of the synthetic v1 control tasks.

DESIGN CONTRACT
  * Pure deterministic function of the input files: no randomness, no
    timestamps, no environment-dependent values in the outputs. Re-running
    after a candidates.json audit round reproduces byte-identical files
    (given the same pinned repos + greppy binary + rg).
  * Grading semantics are the candidates.json floor semantics verbatim:
    expect_members = real caller-name subset floor, file_evidence = path-only
    match, min_count = floor (the reference oracle undercounts; an agent that finds
    MORE true callers must pass).
  * Mechanical gates (enforced here, re-run by verify_real_tasks.py):
      1. vocabulary firewall on every fuzzy_discovery question
      2. multi-hop gate (impact reach@1 / reach@total < 0.3) on every
         research_multihop target, measured with the greppy release binary
         in an ISOLATED store (GREPPY_STORE_DIR).

Task mix (denominators stay open until user sign-off):
  graph_discovery      ~40  who-calls straight from CALLS targets, spread
                            across repos proportional to CALLS survivors
  fuzzy_discovery      <=20 authored synonym-vocabulary (lexicon B) questions,
                            mechanically firewalled (survivor count reported);
                            surviving surplus cut by largest-remainder quota
                            + even spacing per repo, never prefix truncation
  research_multihop    ~15  impact / chain questions on gate-passing targets
  literal_control      <=15 reused verbatim from v1 tasks.json (class
                            literal_control has only 9 ids -> all 9)
  graph_control_synth  ~10  reused from v1 graph_control with a deterministic
                            natural-language question frame, evenly spaced
                            over sorted ids

Usage:
    python3 bench/agent_efficiency/gen_real_tasks.py
    REALTASKS_WORK_DIR=/path/to/scratch python3 .../gen_real_tasks.py

The work dir holds gitignore-free mirrors of the pinned repos (the outer
realcorpus/.gitignore `*/` hides the clones from greppy/rg discovery) and
the isolated greppy store. It is a cache, never an input: outputs depend
only on repo content.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = HERE / "realcorpus" / "candidates.json"
MANIFEST = HERE / "realcorpus" / "MANIFEST.json"
TASKS_V1 = HERE / "tasks.json"
CLASSES_V1 = HERE / "task_classes.json"
OUT_TASKS = HERE / "tasks_v2.json"
OUT_CLASSES = HERE / "task_classes_v2.json"
BIN = HERE.parents[1] / "target" / "release" / "greppy"

REPO_ORDER = ["serde", "flask", "gson", "zod", "tokio", "django"]
LANG_FALLBACK = {
    "serde": "rust",
    "flask": "python",
    "gson": "java",
    "zod": "ts",
    "tokio": "rust",
    "django": "python",
}

WORK_DIR = pathlib.Path(
    os.environ.get("REALTASKS_WORK_DIR")
    or pathlib.Path(tempfile.gettempdir()) / "greppy_realtasks_v2"
)
STORE_DIR = pathlib.Path(os.environ.get("GREPPY_STORE_DIR") or WORK_DIR / "gpstore")

# ----------------------------------------------------------------- knobs
# 140-task expansion (2026-07-06): tokio + django join the pool (their
# candidates are audited in candidates.json); graph/fuzzy/research quotas
# grow proportionally. Target total ≈ 140 incl. the synthetic v1 controls.
N_GRAPH = 60
N_FUZZY_MAX = 30
N_RESEARCH = 25
# PRE-REGISTRATION (BENCHMARK_CONTRACT.md, 2026-07-06): every real repo
# contributes at most REPO_CAP tasks across graph+fuzzy+research combined,
# so no single repo dominates the median (django was 51/134 before). Class
# budget order is fuzzy -> graph -> research: the authored fuzzy bank is the
# scarcest asset, graph is the headline class, research takes the remainder.
REPO_CAP = 16
N_LITERAL = 15          # capped by the 9 ids the v1 class actually has
N_GRAPH_CONTROL = 10
MULTIHOP_MAX_RATIO = 0.3
IMPACT_DEPTH_DIRECT = 1
IMPACT_DEPTH_TOTAL = 12  # "total" transitive reach proxy (default CLI depth is 6)
CHAIN_TASK_MAX = 5       # at most this many research tasks phrased as A->..->B traces
INDEX_TIMEOUT_SECONDS = 1800
INDEX_DIAGNOSTIC_LIMIT = 4096

# ------------------------------------------------------- firewall pieces
# English function words + the fixed task scaffolding vocabulary that appears
# in EVERY fuzzy question ("name the function/method and the file ...") and
# is therefore not discriminative. Symbol-derived vocabulary is NEVER
# stopworded: it is checked via the forbidden-stem rule below.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "can", "could", "may", "might", "must",
    "shall", "should", "will", "would", "where", "what", "which", "who",
    "whom", "whose", "how", "when", "why", "in", "on", "of", "off", "to",
    "for", "and", "or", "nor", "but", "that", "this", "these", "those",
    "it", "its", "itself", "they", "them", "their", "there", "then", "than",
    "not", "no", "any", "all", "every", "each", "one", "ones", "once",
    "from", "with", "without", "within", "into", "onto", "at", "as", "so",
    "if", "else", "other", "another", "etc", "via", "per", "vs", "over",
    "under", "between", "while", "during", "before", "after", "against",
    "through", "up", "down", "out", "about", "above", "below", "same",
    "own", "such", "only", "also", "e", "g", "eg", "ie", "by", "you",
    "your", "we", "our", "us", "he", "she", "his", "her", "have", "has",
    "had", "gets", "get", "got", "make", "makes", "made", "let", "lets",
    "rather", "instead", "actually", "back", "next", "later", "thereafter",
    "elsewhere", "anything", "something", "everything", "nothing",
    # fixed scaffolding present in every fuzzy question (v3 natural
    # suffixes included: "where does that happen in the code", "point me
    # to the function that does this", "which function implements this")
    "name", "names", "function", "functions", "method", "methods",
    "file", "files", "implements", "implement", "lives", "live",
    "repository", "repo", "codebase", "code", "happen", "happens",
    "point", "me",
}

_SUFFIXES = sorted(
    (
        "izations", "ization", "iveness", "fulness", "ations", "ements",
        "ingly", "ation", "ities", "ously", "ement", "ally", "ies", "ied",
        "ers", "ing", "est", "ous", "ive", "able", "ible", "ly", "ed",
        "er", "es", "al", "s",
    ),
    key=len,
    reverse=True,
)


def stem(word: str) -> str:
    """Tiny deterministic suffix-stripper (single pass, longest suffix first).

    Not a linguistic stemmer -- a mechanical normalizer so that morphological
    variants of a symbol part ("signing"/"signer" -> "sign") collide. The
    same function is the generator gate and the verifier gate.
    """
    w = word.lower()
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: len(w) - len(suf)]
    return w


def split_ident(name: str) -> list[str]:
    """snake_case + camelCase split, lowered, parts of length >= 3."""
    parts: list[str] = []
    for chunk in re.split(r"[^0-9A-Za-z]+", name):
        if not chunk:
            continue
        for m in re.finditer(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", chunk):
            parts.append(m.group(0).lower())
    return [p for p in parts if len(p) >= 3]


def content_words(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*", question.lower())
    seen: list[str] = []
    for w in words:
        if len(w) <= 2 or w in STOPWORDS:
            continue
        if w not in seen:
            seen.append(w)
    return seen


def forbidden_stems(symbol: str, target_file: str) -> dict[str, str]:
    """stem -> origin for the target symbol name, its camel/snake parts and
    the target filename stem (+ its parts)."""
    out: dict[str, str] = {}
    fname_stem = pathlib.Path(target_file).stem

    def add(raw: str, origin: str) -> None:
        s = stem(raw.lower())
        if s and s not in out:
            out[s] = origin

    add(re.sub(r"[^0-9A-Za-z]+", "", symbol), "symbol")
    for p in split_ident(symbol):
        add(p, f"symbol part '{p}'")
    add(re.sub(r"[^0-9A-Za-z]+", "", fname_stem), "filename stem")
    for p in split_ident(fname_stem):
        add(p, f"filename part '{p}'")
    return out


def _stems_collide(a: str, b: str) -> bool:
    if a == b:
        return True
    if len(a) >= 4 and len(b) >= 3 and a.startswith(b):
        return True
    if len(b) >= 4 and len(a) >= 3 and b.startswith(a):
        return True
    return False


_RG_CACHE: dict[tuple[str, str], list[tuple[str, int]]] = {}


def rg_file_counts(mirror: pathlib.Path, word: str) -> list[tuple[str, int]]:
    """``rg -i --count-matches`` per file, sorted by (-count, path)."""
    key = (str(mirror), word)
    if key in _RG_CACHE:
        return _RG_CACHE[key]
    p = subprocess.run(
        ["rg", "-i", "--count-matches", "--no-messages", "--", word],
        cwd=mirror, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,  # never let rg fall back to reading stdin
    )
    rows: list[tuple[str, int]] = []
    for line in p.stdout.decode("utf-8", "replace").splitlines():
        path, _, cnt = line.rpartition(":")
        if path and cnt.isdigit():
            rows.append((path, int(cnt)))
    rows.sort(key=lambda r: (-r[1], r[0]))
    _RG_CACHE[key] = rows
    return rows


def firewall_check(
    question: str, symbol: str, target_file: str, mirror: pathlib.Path
) -> tuple[bool, list[str]]:
    """Vocabulary firewall (lexicon-B gate). Returns (ok, violations).

    REJECT when any stemmed content word of the question
      (a) collides with the target symbol name / its camel-snake parts /
          the target filename stem, or
      (b) rg -i of that word (its stem when the stem is >= 4 chars, so
          morphological variants are covered as substrings) puts the target
          file among the top-3 files by hit count.
    """
    violations: list[str] = []
    forb = forbidden_stems(symbol, target_file)
    for raw in content_words(question):
        s = stem(raw)
        for f, origin in forb.items():
            if _stems_collide(s, f):
                violations.append(f"lexical: '{raw}' collides with {origin}")
                break
        else:
            probe = s if len(s) >= 4 else raw
            rows = rg_file_counts(mirror, probe)
            target_count = 0
            for path, cnt in rows:
                if path == target_file:
                    target_count = cnt
                    break
            if target_count > 0:
                stronger = sum(1 for _, cnt in rows if cnt > target_count)
                if stronger < 3:
                    violations.append(
                        f"rg-top3: '{raw}' (probe '{probe}') ranks target "
                        f"{target_file} at #{stronger + 1} "
                        f"({target_count} hits)"
                    )
    return (not violations, violations)


# ------------------------------------------------- repo mirrors + greppy
def ensure_mirrors(manifest: dict) -> dict[str, pathlib.Path]:
    """Copy the pinned repos out from under realcorpus/.gitignore (`*/` hides
    every clone from gitignore-respecting walkers) and index them into the
    isolated store. Mirrors are keyed by the pinned commit -> idempotent."""
    mirrors: dict[str, pathlib.Path] = {}
    (WORK_DIR / "repos").mkdir(parents=True, exist_ok=True)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    for name in REPO_ORDER:
        src = HERE / "realcorpus" / name
        dst = WORK_DIR / "repos" / name
        commit = manifest["repos"][name]["commit"]
        marker = dst / ".realtasks_commit"
        if not (marker.exists() and marker.read_text().strip() == commit):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git"))
            marker.write_text(commit)
        mirrors[name] = dst
    env = dict(os.environ, GREPPY_STORE_DIR=str(STORE_DIR))
    for name, dst in mirrors.items():
        try:
            completed = subprocess.run(
                [str(BIN), "index", str(dst), "--root", str(dst)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=INDEX_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"greppy index timed out for {name} after "
                f"{INDEX_TIMEOUT_SECONDS}s"
            ) from error
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            if len(diagnostic) > INDEX_DIAGNOSTIC_LIMIT:
                diagnostic = diagnostic[-INDEX_DIAGNOSTIC_LIMIT:]
            raise RuntimeError(
                f"greppy index failed for {name} with exit "
                f"{completed.returncode}: {diagnostic or '<no diagnostic>'}"
            )
    return mirrors


def _impact_total(mirror: pathlib.Path, symbol: str, depth: int) -> tuple[bool, int]:
    env = dict(os.environ, GREPPY_STORE_DIR=str(STORE_DIR))
    p = subprocess.run(
        [str(BIN), "impact", symbol, "--root", str(mirror),
         "--depth", str(depth), "--json"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
        stdin=subprocess.DEVNULL,
    )
    try:
        d = json.loads(p.stdout)
    except json.JSONDecodeError:
        return False, 0
    return bool(d.get("symbol_found")), int(d.get("total_exact") or 0)


def impact_ratio(mirror: pathlib.Path, symbol: str) -> dict:
    f1, r1 = _impact_total(mirror, symbol, IMPACT_DEPTH_DIRECT)
    f2, rt = _impact_total(mirror, symbol, IMPACT_DEPTH_TOTAL)
    found = f1 and f2 and rt > 0
    ratio = (r1 / rt) if rt else None
    return {"found": found, "reach1": r1, "reach_total": rt, "ratio": ratio}


def gate_pass(m: dict) -> bool:
    return bool(m["found"]) and m["ratio"] is not None and m["ratio"] < MULTIHOP_MAX_RATIO


def measure_all_calls(
    cands: dict, mirrors: dict[str, pathlib.Path]
) -> dict[tuple[str, str], dict]:
    jobs = []
    for name in REPO_ORDER:
        for t in cands["repos"][name]["edge_types"]["CALLS"]["targets"]:
            jobs.append((name, t["symbol"]))

    def work(job):
        name, sym = job
        return job, impact_ratio(mirrors[name], sym)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(work, jobs))


# --------------------------------------------------- deterministic slicing
def largest_remainder(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """Proportional apportionment, deterministic tiebreak by REPO_ORDER."""
    pool = sum(sizes.values())
    if pool == 0:
        return {k: 0 for k in sizes}
    shares = {k: total * v / pool for k, v in sizes.items()}
    out = {k: min(math.floor(s), sizes[k]) for k, s in shares.items()}
    remaining = total - sum(out.values())
    order = sorted(
        sizes,
        key=lambda k: (-(shares[k] - math.floor(shares[k])), REPO_ORDER.index(k)),
    )
    i = 0
    while remaining > 0 and any(out[k] < sizes[k] for k in sizes):
        k = order[i % len(order)]
        if out[k] < sizes[k]:
            out[k] += 1
            remaining -= 1
        i += 1
    return out


def evenly_spaced(items: list, k: int) -> list:
    n = len(items)
    if k >= n:
        return list(items)
    return [items[(i * n) // k] for i in range(k)]


# ------------------------------------------------------------ fuzzy bank
# Authored ONCE, before any benchmark run, in synonym vocabulary (lexicon B):
# each question describes the BEHAVIOR/PURPOSE of the target without using
# the symbol name, its camel/snake parts, or the target filename stem. The
# mechanical firewall above decides survival; rejects are reported, never
# silently reworded at run time.
FUZZY_BANK: list[dict] = [
    # ---- serde (rust)
    {"repo": "serde", "symbol": "apply_to_field",
     "q": "Which routine converts a struct member's identifier into the "
          "spelling used on the wire?"},
    {"repo": "serde", "symbol": "error_spanned_by",
     "q": "Where do the derive macros stash the problems found while "
          "checking attributes so all of them can be reported at the end?"},
    {"repo": "serde", "symbol": "ungroup",
     "q": "Which helper peels away invisible delimiter wrappers from a type "
          "expression so later pattern matching sees the underlying type?"},
    {"repo": "serde", "symbol": "wrap_in_const",
     "q": "Where does the generated impl get enclosed in a hidden anonymous "
          "scope so its imports cannot leak into the caller's namespace?"},
    {"repo": "serde", "symbol": "borrowable_lifetimes",
     "q": "Which function works out whether deserialized values may point "
          "straight into the original text rather than owning their own "
          "copies?"},
    {"repo": "serde", "symbol": "replace_receiver",
     "q": "Where does the derive rewrite mentions of `Self` inside a remote "
          "type definition into the concrete named target?"},
    {"repo": "serde", "symbol": "with_bound",
     "q": "Where does the derive widen the trait obligations of the emitted "
          "impl so every touched type variable satisfies them?"},
    {"repo": "serde", "symbol": "field_i",
     "q": "Which helper manufactures the numbered placeholder identifiers "
          "used for tuple positions during code generation?"},
    {"repo": "serde", "symbol": "contains_deprecated",
     "q": "Which scan looks through a token stream to spot uses of retired "
          "attribute spellings?"},
    # ---- flask (python)
    {"repo": "flask", "symbol": "get_signing_serializer",
     "q": "Which factory builds the object that cryptographically protects "
          "the data kept in the user's browser between requests?"},
    {"repo": "flask", "symbol": "_find_error_handler",
     "q": "When a request raises a failure, which method checks blueprint "
          "scope and then the project-wide table to pick the function meant "
          "to deal with it?"},
    {"repo": "flask", "symbol": "update_template_context",
     "q": "Where are the standard entries folded into the namespace a page "
          "render receives, keeping anything the view supplied?"},
    {"repo": "flask", "symbol": "explain_template_loading_attempts",
     "q": "Which helper composes the human-readable rundown of every place "
          "the renderer looked for a page source that turned out to be "
          "missing?"},
    {"repo": "flask", "symbol": "should_set_cookie",
     "q": "Which predicate decides if the client-side state needs to go "
          "back to the browser with the current response?"},
    {"repo": "flask", "symbol": "_untag_scan",
     "q": "Where does the reader walk a nested structure to turn marked "
          "placeholder entries back into rich Python objects?"},
    {"repo": "flask", "symbol": "auto_find_instance_path",
     "q": "Which method computes where the per-deployment folder should sit "
          "next to the package when the user did not supply one?"},
    {"repo": "flask", "symbol": "make_aborter",
     "q": "Which method constructs the object used to signal HTTP failures "
          "from a numeric status code?"},
    {"repo": "flask", "symbol": "has_level_handler",
     "q": "Which check climbs the record-emitter hierarchy to see if "
          "anything will actually write out entries of the given severity?"},
    {"repo": "flask", "symbol": "get_flashed_messages",
     "q": "Which function hands the one-time notices queued for the user "
          "over to the page being rendered and clears them from their "
          "state?"},
    {"repo": "flask", "symbol": "_split_blueprint_path",
     "q": "Which routine expands a dotted nesting name into all of its "
          "ancestor prefixes?"},
    # ---- gson (java)
    {"repo": "gson", "symbol": "fillBuffer",
     "q": "Where does the tokenizer replenish its window of unread text "
          "from the wrapped source?"},
    {"repo": "gson", "symbol": "canonicalize",
     "q": "Which utility rewrites a reflective description into an "
          "equivalent normal form so two descriptions of the same construct "
          "compare as identical?"},
    {"repo": "gson", "symbol": "makeAccessible",
     "q": "Where does the library try to open up a normally hidden member "
          "for use, turning failures into a friendlier explanation?"},
    {"repo": "gson", "symbol": "excludeField",
     "q": "Which routine decides whether a class member should be skipped "
          "during serialization according to the configured rules and "
          "annotations?"},
    {"repo": "gson", "symbol": "getTimePartOfDateTimePattern",
     "q": "Which method builds the clock portion of the locale-dependent "
          "display style used on legacy runtimes?"},
    {"repo": "gson", "symbol": "isWrapperType",
     "q": "Which predicate reports whether a class is one of the boxed "
          "counterparts of the built-in scalar kinds?"},
    {"repo": "gson", "symbol": "calculateHashMapCapacity",
     "q": "Where is the initial sizing figure computed so a lookup container "
          "can hold an expected number of entries without resizing?"},
    {"repo": "gson", "symbol": "checkInstantiable",
     "q": "Where does the library verify that a requested class is concrete "
          "enough to actually build an object of, rather than a purely "
          "declarative kind?"},
    # ---- zod (ts)
    {"repo": "zod", "symbol": "addIssueToContext",
     "q": "Where do validation failures get recorded onto the shared state "
          "while a schema evaluates its input?"},
    {"repo": "zod", "symbol": "unwrapMessage",
     "q": "Which helper accepts either a fixed string or a factory for the "
          "human-readable complaint text and normalizes it to a string?"},
    {"repo": "zod", "symbol": "extractDefs",
     "q": "Where does the converter hoist repeatedly referenced fragments so "
          "each is emitted once and referred to from elsewhere in the "
          "output?"},
    {"repo": "zod", "symbol": "getRussianPlural",
     "q": "Which helper picks the correct noun form for a count in the "
          "Slavic locale that distinguishes three grammatical variants?"},
    {"repo": "zod", "symbol": "fetchStars",
     "q": "Where does the documentation site retrieve the repository's "
          "GitHub popularity figure for display?"},
    {"repo": "zod", "symbol": "compile",
     "q": "Where does the internal source generator assemble its "
          "accumulated body into a callable it can hand back?"},
    {"repo": "zod", "symbol": "capitalizeFirstCharacter",
     "q": "Which tiny utility upper-cases the leading letter of a phrase in "
          "one of the Baltic language packs?"},
    # ---- tokio (rust) — 140-expansion 2026-07-06
    {"repo": "tokio", "symbol": "enable_all",
     "q": "Which configuration call switches on both the clock and the "
          "network drivers at once while assembling an executor?"},
    {"repo": "tokio", "symbol": "unbounded_channel",
     "q": "Which constructor creates a queue pair whose producer never has "
          "to wait because capacity is limitless?"},
    {"repo": "tokio", "symbol": "sleep_until",
     "q": "Which timer primitive suspends a task up to a specific instant "
          "rather than for a relative amount of time?"},
    {"repo": "tokio", "symbol": "insert_at",
     "q": "Which method schedules an element to surface from the queue at "
          "an exact moment instead of after a relative wait?"},
    {"repo": "tokio", "symbol": "notified_owned",
     "q": "Which variant hands back a wakeup future holding its own handle "
          "to the signal source so it can outlive any borrow?"},
    {"repo": "tokio", "symbol": "interval_at",
     "q": "Which factory builds a repeating ticker whose first firing "
          "happens at a chosen starting moment?"},
    {"repo": "tokio", "symbol": "try_acquire_owned",
     "q": "Which non-blocking call claims a permit carrying its own handle "
          "to the limiter, failing immediately when none are free?"},
    {"repo": "tokio", "symbol": "poll_reserve",
     "q": "Which readiness probe waits for free capacity in the outgoing "
          "queue before a value gets committed to it?"},
    {"repo": "tokio", "symbol": "send_item",
     "q": "Which helper pushes the previously reserved value into the "
          "queue slot that was set aside for it?"},
    {"repo": "tokio", "symbol": "open_receiver",
     "q": "Which constructor attaches the reading end to an existing named "
          "FIFO on the filesystem?"},
    # ---- django (python) — 140-expansion 2026-07-06
    {"repo": "django", "symbol": "mark_safe",
     "q": "Which utility flags text as trusted markup so the template "
          "engine will not encode it a second time?"},
    {"repo": "django", "symbol": "force_bytes",
     "q": "Which coercion helper guarantees its argument comes back as raw "
          "octets, unwrapping lazily promised objects on the way?"},
    {"repo": "django", "symbol": "import_string",
     "q": "Which utility resolves a dotted path into the live Python object "
          "it names?"},
    {"repo": "django", "symbol": "make_password",
     "q": "Which function turns a plain secret into the salted, hashed form "
          "stored for a user account?"},
    {"repo": "django", "symbol": "get_permission_codename",
     "q": "Which helper composes the short action identifier that the "
          "authorization layer checks for a model operation?"},
    {"repo": "django", "symbol": "format_html",
     "q": "Which builder interpolates values into a markup fragment after "
          "escaping every argument first?"},
    {"repo": "django", "symbol": "method_decorator",
     "q": "Which adapter lets a wrapper written for plain callables be "
          "applied to a function bound to a class instead?"},
    {"repo": "django", "symbol": "from_model",
     "q": "Which constructor captures a live ORM class into the frozen "
          "representation used while computing schema changes?"},
    {"repo": "django", "symbol": "normalize_choices",
     "q": "Which helper flattens the many accepted shapes of an options "
          "list into canonical value-label pairs?"},
    {"repo": "django", "symbol": "deprecate_posargs",
     "q": "Which wrapper warns callers that passing certain arguments "
          "without keywords will stop working in a future release?"},
]


# ------------------------------------------------------------------ build
def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs():
    cands = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tasks_v1 = json.loads(TASKS_V1.read_text(encoding="utf-8"))
    classes_v1 = json.loads(CLASSES_V1.read_text(encoding="utf-8"))
    return cands, manifest, tasks_v1, classes_v1


def _lang(manifest: dict, repo: str) -> str:
    return manifest["repos"].get(repo, {}).get("lang", LANG_FALLBACK[repo])


def _calls_map(cands: dict, repo: str) -> dict[str, dict]:
    return {t["symbol"]: t
            for t in cands["repos"][repo]["edge_types"]["CALLS"]["targets"]}


def _locate_map(cands: dict, repo: str) -> dict[str, dict]:
    """CALLS + USAGE targets by symbol (fuzzy tasks may use either)."""
    out: dict[str, dict] = {}
    for et in ("CALLS", "USAGE"):
        for t in cands["repos"][repo]["edge_types"].get(et, {}).get("targets", []):
            out.setdefault(t["symbol"], t)
    return out


def _who_calls_check(t: dict) -> dict:
    return {
        "kind": "who_calls",
        "symbol": t["symbol"],
        "expect_members": list(t["expect_members"]),
        "file_evidence": list(t["file_evidence"]),
        "min_count": t["min_count"],
        "semantics": "floor",
    }


# ------------------------------------------------- natural phrasing (v3)
# Owner directive 2026-07-06: benchmark questions must read like what a
# developer actually types at a coding agent — terse, goal-driven, NO
# output-format coaching ("list the callers", "give the total number") and
# no command-shaped wording that telegraphs a specific tool. Ground truth
# and grading are UNCHANGED: the mechanical grader checks the audited
# caller/file floors inside the free-form answer, and the count floor is
# inferred from the named members (grade_answers.count_check_mechanical),
# so dropping the format coaching does not break gradeability — it makes
# the task harder for BOTH arms equally.
#
# Frames rotate deterministically per (repo, symbol) so the mix is a pure
# function of candidates.json (stable across regenerations, no RNG).

GRAPH_FRAMES = [
    "I want to change the signature of `{name}` — what do I need to touch?",
    "is `{name}` dead code, or is it still used somewhere?",
    "before I refactor `{name}`: what depends on it?",
    "I'm reviewing a PR that touches `{name}`. Which callers could be affected?",
    "quick check: where is `{name}` called from?",
    "I need to deprecate `{name}`. What still calls it?",
    "trying to figure out why `{name}` seems to run twice — what invokes it?",
    "if `{name}` starts returning an error, what breaks upstream?",
]

CHAIN_FRAMES = [
    "I'm debugging something odd: a change in `{frm}` ends up affecting "
    "`{to}`. How does it get there?",
    "somehow `{frm}` influences what `{to}` does. Can you explain the "
    "connection through the code?",
    "does `{frm}` ever end up reaching `{to}`? How?",
]

IMPACT_FRAMES = [
    "we're planning to change what `{sym}` ({file}) does. How far could "
    "that ripple through the codebase?",
    "a bug was just found in `{sym}` ({file}). What could be affected by "
    "it, including indirectly?",
    "how risky is it to touch `{sym}` in {file}? What depends on it, "
    "directly or not?",
]

FUZZY_SUFFIXES = [
    " Where does that happen in the code?",
    " Point me to the function that does this.",
    " Which function implements this?",
]


def _frame_pick(frames: list[str], repo: str, symbol: str) -> str:
    """Deterministic frame choice — a pure function of (repo, symbol)."""
    h = int(hashlib.sha256(f"{repo}/{symbol}".encode()).hexdigest()[:8], 16)
    return frames[h % len(frames)]


def control_payload(source: dict, task_class: str) -> dict:
    """Derive a control payload exactly as emitted by ``build``."""
    reused = {key: value for key, value in source.items() if key != "id"}
    if task_class != "graph_control_synth":
        return reused
    check = reused.get("check", {})
    if check.get("kind") == "who_calls" and check.get("symbol"):
        reused["q"] = _frame_pick(
            GRAPH_FRAMES, "synth", check["symbol"]
        ).format(name=check["symbol"])
    elif check.get("kind") == "path" and check.get("frm") and check.get("to"):
        reused["q"] = _frame_pick(
            CHAIN_FRAMES, "synth", f"{check['frm']}->{check['to']}"
        ).format(frm=check["frm"], to=check["to"])
    return reused


def _graph_question(repo: str, t: dict) -> str:
    owner = t.get("filters", {}).get("owner_class")
    name = (
        f"{owner}.{t['symbol']}"
        if t.get("filters", {}).get("is_method") and owner
        else t["symbol"]
    )
    return _frame_pick(GRAPH_FRAMES, repo, t["symbol"]).format(name=name)


def _graph_ground_truth(t: dict) -> str:
    parts = [f"Oracle floor: at least {t['min_count']} caller(s)."]
    if t["expect_members"]:
        parts.append("Known callers include: " + ", ".join(t["expect_members"]) + ".")
    if t["file_evidence"]:
        parts.append("Caller file evidence: " + ", ".join(t["file_evidence"]) + ".")
    parts.append("An agent finding MORE true callers than the floor must pass.")
    return " ".join(parts)


def build() -> dict:
    """Returns {"tasks": [...], "classes": {...}, "report": {...}}."""
    for path in (CANDIDATES, MANIFEST, TASKS_V1, CLASSES_V1):
        if not path.exists():
            sys.exit(f"[gen] missing input: {path}")
    if not BIN.exists():
        sys.exit(f"[gen] greppy release binary missing: {BIN}")
    if not shutil.which("rg"):
        sys.exit("[gen] rg (ripgrep) not on PATH -- required by the firewall")

    cands, manifest, tasks_v1, classes_v1 = _load_inputs()
    mirrors = ensure_mirrors(manifest)

    fuzzy_symbols = {r: {e["symbol"] for e in FUZZY_BANK if e["repo"] == r}
                     for r in REPO_ORDER}

    # ---- graph_discovery: CALLS targets, proportional across repos,
    # evenly spaced over sorted symbols (fuzzy-authored symbols excluded
    # from the pool regardless of firewall outcome -> pure candidates.json
    # function).
    graph_pool: dict[str, list[dict]] = {}
    for repo in REPO_ORDER:
        calls = _calls_map(cands, repo)
        pool = [t for s, t in sorted(calls.items())
                if s not in fuzzy_symbols[repo] and t["expect_members"]]
        graph_pool[repo] = pool

    # ---- fuzzy_discovery: authored bank -> mechanical firewall. Survivors
    # above the ~20 cap are cut by the SAME deterministic apportionment used
    # for the graph classes (largest remainder across repos, even spacing
    # over the symbol-sorted survivor list) so no repo is prefix-starved.
    fuzzy_surv: dict[str, list[tuple[str, dict, dict]]] = {
        r: [] for r in REPO_ORDER}
    fuzzy_rejections: list[dict] = []
    bank = sorted(FUZZY_BANK, key=lambda e: (REPO_ORDER.index(e["repo"]), e["symbol"]))
    for entry in bank:
        repo = entry["repo"]
        target = _locate_map(cands, repo).get(entry["symbol"])
        if target is None:
            fuzzy_rejections.append({
                "repo": repo, "symbol": entry["symbol"],
                "reason": "target no longer in candidates.json",
            })
            continue
        ok, violations = firewall_check(
            entry["q"], target["symbol"], target["file"], mirrors[repo])
        if not ok:
            fuzzy_rejections.append({
                "repo": repo, "symbol": entry["symbol"],
                "reason": "vocabulary firewall", "violations": violations,
            })
            continue
        fuzzy_surv[repo].append((repo, target, entry))
    fquotas = largest_remainder(
        min(N_FUZZY_MAX, sum(len(v) for v in fuzzy_surv.values())),
        {r: min(len(fuzzy_surv[r]), REPO_CAP) for r in REPO_ORDER})
    fuzzy_sel: list[tuple[str, dict, dict]] = []
    for repo in REPO_ORDER:
        fuzzy_sel.extend(evenly_spaced(fuzzy_surv[repo], fquotas[repo]))

    # ---- graph_discovery quotas, under the per-repo cap remaining after
    # the fuzzy class (see REPO_CAP pre-registration note).
    cap_left = {r: REPO_CAP - fquotas[r] for r in REPO_ORDER}
    quotas = largest_remainder(
        N_GRAPH,
        {r: min(len(graph_pool[r]), cap_left[r]) for r in REPO_ORDER})
    graph_sel: list[tuple[str, dict]] = []
    for repo in REPO_ORDER:
        for t in evenly_spaced(graph_pool[repo], quotas[repo]):
            graph_sel.append((repo, t))
    cap_left = {r: cap_left[r] - quotas[r] for r in REPO_ORDER}

    # ---- research_multihop: multi-hop gate over ALL CALLS targets
    measured = measure_all_calls(cands, mirrors)
    used = {(r, t["symbol"]) for r, t in graph_sel}
    research_pool: dict[str, list[dict]] = {}
    gate_stats = {r: {"measured": 0, "unresolved": 0, "gate_pass": 0}
                  for r in REPO_ORDER}
    for repo in REPO_ORDER:
        calls = _calls_map(cands, repo)
        pool = []
        for sym, t in sorted(calls.items()):
            m = measured[(repo, sym)]
            gate_stats[repo]["measured"] += 1
            if not m["found"]:
                gate_stats[repo]["unresolved"] += 1
                continue
            if gate_pass(m):
                gate_stats[repo]["gate_pass"] += 1
                if ((repo, sym) not in used and sym not in fuzzy_symbols[repo]
                        and t["expect_members"]):
                    pool.append(t)
        research_pool[repo] = pool
    rquotas = largest_remainder(
        N_RESEARCH,
        {r: min(len(research_pool[r]), cap_left[r]) for r in REPO_ORDER})
    research_sel: list[tuple[str, dict]] = []
    for repo in REPO_ORDER:
        for t in evenly_spaced(research_pool[repo], rquotas[repo]):
            research_sel.append((repo, t))

    # chain sub-form: X gate-passed; C in X.expect_members is itself a CALLS
    # target; D in C.expect_members (D not in {X, C}) -> "trace D -> C -> X".
    def find_chain(repo: str, t: dict):
        calls = _calls_map(cands, repo)
        best = None
        for c in sorted(t["expect_members"]):
            ct = calls.get(c)
            if ct is None or c == t["symbol"]:
                continue
            for dname in sorted(ct["expect_members"]):
                if dname in (t["symbol"], c):
                    continue
                cand = (c, dname)
                if best is None or cand < best:
                    best = cand
                break  # sorted -> first D is the smallest for this C
        return best

    # ---- literal_control + graph_control_synth from the frozen v1 corpus
    v1_by_id = {t["id"]: t for t in tasks_v1}
    lit_ids = evenly_spaced(
        sorted(classes_v1["classes"]["literal_control"]["ids"]), N_LITERAL)
    gc_ids = evenly_spaced(
        sorted(classes_v1["classes"]["graph_control"]["ids"]), N_GRAPH_CONTROL)

    # ------------------------------------------------------------ assemble
    tasks: list[dict] = []
    class_ids: dict[str, list[str]] = {
        "graph_discovery": [], "fuzzy_discovery": [], "research_multihop": [],
        "literal_control": [], "graph_control_synth": [],
    }

    def next_id() -> str:
        return f"r{len(tasks) + 1:03d}"

    for repo, t in graph_sel:
        tid = next_id()
        tasks.append({
            "id": tid, "repo": repo, "lang": _lang(manifest, repo),
            "type": "locate", "class": "graph_discovery",
            "q": _graph_question(repo, t),
            "ground_truth": _graph_ground_truth(t),
            "target": {"file": t["file"], "line": t["line"],
                       "owner_class": t.get("filters", {}).get("owner_class")},
            "check": _who_calls_check(t),
        })
        class_ids["graph_discovery"].append(tid)

    for repo, target, entry in fuzzy_sel:
        tid = next_id()
        tasks.append({
            "id": tid, "repo": repo, "lang": _lang(manifest, repo),
            "type": "locate", "class": "fuzzy_discovery",
            "q": entry["q"] + _frame_pick(FUZZY_SUFFIXES, repo, target["symbol"]),
            "ground_truth": (
                f"{target['symbol']} in {target['file']} "
                f"(line {target['line']})."
            ),
            "target": {"file": target["file"], "line": target["line"],
                       "edge_type": target["edge_type"]},
            "check": {
                "kind": "search_symbols",
                "query": target["symbol"],
                "expect_file": target["file"],
                "semantics": "floor",
            },
        })
        class_ids["fuzzy_discovery"].append(tid)

    chains_used = 0
    for repo, t in research_sel:
        tid = next_id()
        m = measured[(repo, t["symbol"])]
        gate = {
            "reach1": m["reach1"], "reach_total": m["reach_total"],
            "ratio": round(m["ratio"], 4), "max_ratio": MULTIHOP_MAX_RATIO,
            "measured_by": (
                f"greppy impact --depth {IMPACT_DEPTH_DIRECT} vs "
                f"--depth {IMPACT_DEPTH_TOTAL} (isolated store)"
            ),
        }
        chain = find_chain(repo, t) if chains_used < CHAIN_TASK_MAX else None
        if chain is not None:
            c, dname = chain
            chains_used += 1
            tasks.append({
                "id": tid, "repo": repo, "lang": _lang(manifest, repo),
                "type": "research", "class": "research_multihop",
                "q": _frame_pick(CHAIN_FRAMES, repo, t["symbol"]).format(
                    frm=dname, to=t["symbol"]
                ),
                "ground_truth": (
                    f"{dname} -> {c} -> {t['symbol']} is one witnessed "
                    f"chain (oracle floor; other chains may exist)."
                ),
                "target": {"file": t["file"], "line": t["line"]},
                "multihop_gate": gate,
                "check": {
                    "kind": "path", "frm": dname, "to": t["symbol"],
                    "via": [c], "semantics": "floor",
                },
            })
        else:
            tasks.append({
                "id": tid, "repo": repo, "lang": _lang(manifest, repo),
                "type": "research", "class": "research_multihop",
                "q": _frame_pick(IMPACT_FRAMES, repo, t["symbol"]).format(
                    sym=t["symbol"], file=t["file"]
                ),
                "ground_truth": _graph_ground_truth(t) + (
                    " Transitive reach is strictly larger than the direct "
                    "floor (multi-hop gate: reach@1/reach@total "
                    f"= {round(m['ratio'], 4)} < {MULTIHOP_MAX_RATIO})."
                ),
                "target": {"file": t["file"], "line": t["line"]},
                "multihop_gate": gate,
                "check": {
                    "kind": "impact",
                    "symbol": t["symbol"],
                    "direction": "incoming",
                    "expect_members": list(t["expect_members"]),
                    "file_evidence": list(t["file_evidence"]),
                    "min_count": t["min_count"],
                    "semantics": "floor",
                },
            })
        class_ids["research_multihop"].append(tid)

    for src_ids, cls in ((lit_ids, "literal_control"),
                         (gc_ids, "graph_control_synth")):
        for sid in src_ids:
            tid = next_id()
            src = v1_by_id[sid]
            # v3 natural phrasing also applies to the synthetic graph
            # controls ("Who calls X? List the calling functions." is the
            # same command-shaped voice the real classes dropped). Ground
            # truth / checks stay verbatim; literal_control keeps its v1
            # wording (already how a user types a literal lookup, and it is
            # the grep-favoring control).
            reused = control_payload(src, cls)
            tasks.append({"id": tid, "class": cls, "source_id": sid, **reused})
            class_ids[cls].append(tid)

    classes_doc = {
        "schema_version": 1,
        "purpose": (
            "Machine-readable corpus-v2 router/regression classes for the "
            "six-repository real-code agent benchmark plus deterministic "
            "synthetic v1 controls."
        ),
        "source_evidence": [
            "bench/agent_efficiency/realcorpus/candidates.json",
            "bench/agent_efficiency/realcorpus/MANIFEST.json",
            "bench/agent_efficiency/tasks.json",
            "bench/agent_efficiency/task_classes.json",
        ],
        "generator": (
            "bench/agent_efficiency/gen_real_tasks.py -- deterministic "
            "function of the inputs; no randomness, no timestamps."
        ),
        "input_candidates_sha256": _sha256(CANDIDATES),
        "grading_semantics": (
            "floor: expect_members = real caller-name subset floor, "
            "file_evidence = path-only match, min_count = floor; the C "
            "oracle undercounts, so an agent finding MORE true callers "
            "must pass."
        ),
        "classes": {
            "graph_discovery": {
                "role": "avoid_embedding",
                "description": (
                    "Who-calls questions built 1:1 from audited CALLS "
                    "targets in the real repos; exact graph/literal search "
                    "should answer without vector retrieval."
                ),
                "ids": class_ids["graph_discovery"],
                "acceptance": {
                    "embedding_must_not_be_invoked": True,
                    "direct_scope_must_not_be_reported_as_transitive": True,
                    "quality_must_not_regress": True,
                },
            },
            "fuzzy_discovery": {
                "role": "embedding_candidate",
                "description": (
                    "Synonym-vocabulary (lexicon B) behaviour descriptions "
                    "with a mechanical firewall guaranteeing the question "
                    "shares no stemmed content word with the target symbol/"
                    "filename and is not rg-solvable to the target file "
                    "(top-3 rule)."
                ),
                "ids": class_ids["fuzzy_discovery"],
                "acceptance": {
                    "median_tool_calls_max": 2.5,
                    "context_median_must_not_regress_vs_current_greppy": True,
                    "quality_must_not_regress": True,
                },
            },
            "research_multihop": {
                "role": "avoid_embedding",
                "description": (
                    "Impact/blast-radius and call-chain questions on targets "
                    "passing the multi-hop gate (impact reach@1/reach@total "
                    "< 0.3); graph traversal, not vector similarity, is the "
                    "intended tool."
                ),
                "ids": class_ids["research_multihop"],
                "acceptance": {
                    "embedding_must_not_be_invoked": True,
                    "direct_scope_must_not_be_reported_as_transitive": True,
                    "quality_must_not_regress": True,
                },
            },
            "literal_control": {
                "role": "avoid_embedding",
                "description": (
                    "Literal/local/exact tasks reused verbatim from the "
                    "synthetic v1 corpus (regression control; vector search "
                    "must not be invoked)."
                ),
                "ids": class_ids["literal_control"],
                "acceptance": {
                    "embedding_must_not_be_invoked": True,
                    "candidate_must_not_expand_transitively_unless_asked": True,
                    "quality_must_not_regress": True,
                },
            },
            "graph_control_synth": {
                "role": "avoid_embedding",
                "description": (
                    "Exact graph tasks derived from the synthetic v1 "
                    "graph_control class with deterministic natural-language "
                    "question frames (regression control for the shared "
                    "resolver; must not regress on rust/python)."
                ),
                "ids": class_ids["graph_control_synth"],
                "acceptance": {
                    "embedding_must_not_be_invoked": True,
                    "direct_scope_must_not_be_reported_as_transitive": True,
                    "quality_must_not_regress": True,
                },
            },
        },
    }

    report = {
        "fuzzy_authored": len(FUZZY_BANK),
        "fuzzy_survivors": sum(len(v) for v in fuzzy_surv.values()),
        "fuzzy_selected": len(fuzzy_sel),
        "fuzzy_quotas": fquotas,
        "fuzzy_rejections": fuzzy_rejections,
        "multihop_gate_stats": gate_stats,
        "graph_quotas": quotas,
        "research_quotas": rquotas,
        "research_pool_sizes": {r: len(research_pool[r]) for r in REPO_ORDER},
        "literal_available": len(classes_v1["classes"]["literal_control"]["ids"]),
        "literal_requested": N_LITERAL,
    }
    return {"tasks": tasks, "classes": classes_doc, "report": report}


def serialize(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    built = build()
    OUT_TASKS.write_text(serialize(built["tasks"]), encoding="utf-8")
    OUT_CLASSES.write_text(serialize(built["classes"]), encoding="utf-8")

    tasks, rep = built["tasks"], built["report"]
    print(f"[gen] wrote {OUT_TASKS.name}: {len(tasks)} tasks")
    by_class: dict[str, int] = {}
    by_repo: dict[str, int] = {}
    for t in tasks:
        by_class[t["class"]] = by_class.get(t["class"], 0) + 1
        by_repo[t["repo"]] = by_repo.get(t["repo"], 0) + 1
    for c in sorted(by_class):
        print(f"[gen]   class {c:20s} {by_class[c]:3d}")
    for r in sorted(by_repo):
        print(f"[gen]   repo  {r:20s} {by_repo[r]:3d}")
    print(f"[gen] fuzzy firewall: {rep['fuzzy_authored']} authored, "
          f"{rep['fuzzy_survivors']} survived, "
          f"{len(rep['fuzzy_rejections'])} rejected; "
          f"{rep['fuzzy_selected']} selected {json.dumps(rep['fuzzy_quotas'])}")
    for rj in rep["fuzzy_rejections"]:
        why = rj.get("violations", [rj["reason"]])
        print(f"[gen]   REJECT {rj['repo']}/{rj['symbol']}: {why[0]}")
    print(f"[gen] multi-hop gate stats (all CALLS targets): "
          f"{json.dumps(rep['multihop_gate_stats'])}")
    print(f"[gen] literal_control: v1 class has {rep['literal_available']} "
          f"ids (requested ~{rep['literal_requested']}; all reused)")
    print(f"[gen] sha256 {OUT_TASKS.name}    = {_sha256(OUT_TASKS)}")
    print(f"[gen] sha256 {OUT_CLASSES.name} = {_sha256(OUT_CLASSES)}")
    print(f"[gen] sha256 candidates.json (input) = {_sha256(CANDIDATES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
