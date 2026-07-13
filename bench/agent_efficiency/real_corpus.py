#!/usr/bin/env python3
"""REAL-REPO ground-truth pipeline for the greppy benchmark (corpus v2).

The synthetic corpus is void for fuzzy/research/graph task classes (template
degeneracy, zero vocabulary gap, hop saturation).  This pipeline builds
ground truth from 4 pinned real repos, using a pinned reference oracle binary
(reference oracle, f0c9be1) as the oracle for CALLS/USAGE/IMPORTS edges,
then hard-filters everything the audits flagged as unreliable:

  (a) unique-name targets only  — the C resolver suffix-matches, so common
      names ('get') over-link into stdlib; we additionally require length>=4,
      no dunders, a stoplist of ubiquitous method names, and NO target whose
      name collides with a Class name (constructor overloads: `new MyObject()`
      textually matches every `MyObject(...)` ctor — probe finding, gson).
  (b) no Module@line-0 sources  — macro/module attribution artifacts.
  (c) ripgrep cross-check       — every claimed caller must have a textual
      occurrence of the target name inside the caller's line span; for CALLS
      the evidence line must look like an actual call (`name(`), not a bare
      identifier/type annotation (probe finding, zod 'defaulter'); targets
      where rg finds call sites the oracle missed beyond a tolerance are
      rejected too (the floor would be too far below the truth).
  (d) floor semantics only      — candidates carry min_count + expect_members
      (a SUBSET of true callers); the reference oracle under-counts sometimes
      (get_root_path), so an agent finding MORE true callers must not fail.

Adversarial-audit hardening (2026-07-02; 3/6 audited candidates were broken):
  (e) dependency direction      — a target defined under a test/bench/docs/
      examples subtree is REJECTED unless ALL its callers live inside the
      SAME aux subtree; targets outside aux subtrees must be defined under
      the repo's primary src tree (serde*/src, src/flask, gson/src/main,
      packages/zod/src).  Kills e.g. zod USAGE 'data' (bench helper with
      src "callers") and serde USAGE 'option' (a #[test] fn whose 4/4
      "users" were word collisions in serde_core).
  (f) stdlib-name defense       — per-language stoplist of common stdlib /
      prototype names (java.lang.Math.toIntExact, Number.prototype.toFixed,
      Rust unwrap/as_str, Python builtins/str methods ...); for Java a
      claimed caller whose file has 'import static X.<name>;' pointing to a
      different class is rejected; a 'private' (java/ts) target cannot have
      callers in other files; a plain Function (non-Method) target only
      accepts call evidence NOT of the dotted-receiver form 'x.name('.
  (g) USAGE reference shape     — USAGE/IMPORTS evidence lines must contain
      the name in call position 'name(', import/use position, or qualified
      reference ('::name'; '.name' for methods only); parameter names,
      struct field defs and macro token lists never count; quoted module
      strings ('import {type} from "arktype"') never count; Python
      from-imports must resolve to the target's own module ('from
      flask.globals import app_ctx' is not evidence for a same-named test
      fixture).  If a repo ends up with fewer than 15 USAGE survivors,
      USAGE is DROPPED for that repo (recorded in MANIFEST.json).
  (h) no pseudo-callers         — File/Module oracle sources are moved out
      of expect_members into a separate 'file_evidence' field (matched by
      file path only); expect_members contains ONLY real function/method
      caller names.

Round-2 adversarial-audit hardening (2026-07-02; 3/725 targets broken, all
through ONE systematic gap: "unique_name" was relative to the ORACLE's def
dump, which misses parameters, local closures, getter/setter pairs and
sometimes whole methods):
  (i) textual def-multiplicity + shadowing — every candidate is re-checked
      with a per-language TEXTUAL definition scan over the repo (named
      functions/methods; TS/JS const/let/var '= (' and '=>' closures,
      shorthand methods, get/set accessors, class-field arrows; java
      methods incl. annotation elements and overloads; rust fn; python def
      incl. nested).  textual def count > oracle def count => REJECT
      (kills zod 'processError': 3 local closures vs 1 oracle def; gson
      'serializeNulls': Gson.serializeNulls() missed by the oracle's def
      dump).  Additionally, a parameter/local binding named like the
      target declared near an evidence line (window: 120 lines)
      REJECTS the target (kills zod v3 'message': the sole evidence
      'return message(val)' calls a shadowing refine() PARAMETER).
      Rejection key: 'shadowed_or_multidef'.
  (j) Method-target evidence tightening — for Method targets, cross-file
      call evidence must be dotted/qualified ('.name(' / '::name(' /
      'super().name('); bare 'name(' only counts when the evidence file IS
      the target's file (and the same-file shadowing scan is clean).

Round-3 adversarial-audit hardening (2026-07-02; 4/678 targets CONFIRMED
broken via 2 attack classes + 1 determinism failure):
  (k) TS hidden-def forms       — TEXT_DEF_RE now also counts
      PROPERTY-ASSIGNED callables ('inst.extract = (values, params) => {'),
      property ALIASES ('inst.removeDefault = inst.unwrap;'),
      object-literal / interface callable properties ('name: (…) =>'),
      single-line interface method signatures ('removeDefault(): T;') and
      multiline generic signature openers ('extract<const U …>(').  The
      zod v4 classic API assigns its methods onto instances inside
      $constructor closures — invisible to every def form counted before.
  (l) cross-version attribution — a target and ALL its claimed callers
      must resolve to the SAME version subtree (zod: v3 | v4/classic |
      v4/mini | v4 shared core; classic and mini are sibling islands over
      the shared core, so a classic/mini target only accepts callers from
      its own island while a shared-core target accepts any v4 caller).
      Kills zod v3 'extract'/'exclude'/'removeDefault', whose "callers"
      import 'zod/v4' and dispatch to the v4 property closures/aliases.
  (m) Java external-override    — a unique-name Method carrying @Override
      necessarily overrides a NON-repo (JDK/external) super-method (a
      repo-side super def would already break unique_name / textual
      def-multiplicity), so dotted evidence like 'number.longValue()' can
      statically bind to java.lang.Number instead of the repo override.
      Such targets are REJECTED unless EVERY evidence receiver is
      statically the owner class or a repo type related to it (rejection
      key 'external_override_receiver'; kills gson
      LazilyParsedNumber.longValue).  Cheap backstop: the java stoplist
      now carries the java.lang.Number accessors.
  (n) oracle determinism        — EDGE_INTERSECT_RUNS raised to 3 AND the
      whole candidate build now runs STABILITY_BUILDS=3 independent times;
      a target that is not byte-identical across ALL builds is
      flicker-dropped (rejection key 'nondeterministic_oracle'; known
      flickerer: serde CALLS 'ident' @ serde_derive/src/lib.rs:99).
      Volatile diagnostic fields are listed under 'volatile_diagnostics'
      in candidates.json; the target arrays themselves must be (and are
      verified to be) byte-identical across consecutive regenerations.

HANDLES / SEMANTICALLY_RELATED / SIMILAR_TO are nondeterministic run-to-run
and are never queried here.

Subcommands
  setup   shallow-clone the 4 repos at pinned commits into realcorpus/<name>
          (prefers existing local checkouts as clone source; skips if already
          present at the right commit) and write realcorpus/MANIFEST.json.
  oracle  index each repo with the reference oracle in an ISOLATED store
          (CBM_CACHE_DIR under realcorpus/.cbm-store/<name>), dump
          CALLS/USAGE/IMPORTS for all Function/Method symbols, apply filters
          (a)-(c), emit realcorpus/candidates.json with filter provenance.
  probe   for gson + zod ONLY (java/ts oracle quality is UNPROBED): report
          how many targets survive, print 5 spot-check targets with code
          snippets for human verification, and mark a repo 'unreliable' in
          MANIFEST.json if <20 reliable CALLS targets survive.
  audit   automated adversarial gate over candidates.json: re-runs the
          systemic scans (dependency direction, stdlib-name grep, private
          visibility, File/Module names inside expect_members) and exits
          non-zero on ANY hit.
  all     setup + oracle + probe + audit (audit failure fails 'all').

License note: repos are cloned locally for benchmarking only, never
redistributed; MANIFEST.json records each upstream license.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
_SEED_DIR = os.environ.get("GREPPY_BENCH_SEED_DIR")
ROOT = HERE / "realcorpus"
REPO_ROOT = HERE.parents[1]
CBM = Path(os.environ.get("GREPPY_REFERENCE_ORACLE", "reference-oracle"))
MANIFEST = ROOT / "MANIFEST.json"
CANDIDATES = ROOT / "candidates.json"

REPOS = {
    "serde": {
        "url": "https://github.com/serde-rs/serde",
        "commit": "1023d077510b4aef36a41ef56fdb7798568a2654",
        "lang": "rust",
        "license": "MIT OR Apache-2.0",
        "local_seed": HERE / "realcorpus-eval" / "serde",
        "exts": [".rs"],
    },
    "flask": {
        "url": "https://github.com/pallets/flask",
        "commit": "36e4a824f340fdee7ed50937ba8e7f6bc7d17f81",
        "lang": "python",
        "license": "BSD-3-Clause",
        "local_seed": HERE / "realcorpus-eval" / "flask",
        "exts": [".py"],
    },
    "gson": {
        "url": "https://github.com/google/gson",
        "commit": "c9f3fd55854a743b66f857ace3c7b268ea3e2ef7",
        "lang": "java",
        "license": "Apache-2.0",
        "local_seed": None,
        "exts": [".java"],
    },
    "zod": {
        "url": "https://github.com/colinhacks/zod",
        "commit": "912f0f51b0ced654d0069741e7160834dca742ee",
        "lang": "ts",
        "license": "MIT",
        "local_seed": None,
        "exts": [".ts", ".tsx", ".mts", ".cts"],
    },
    # ---- larger-repo expansion: optional local seed via GREPPY_BENCH_SEED_DIR ----
    "tokio": {
        "url": "https://github.com/tokio-rs/tokio",
        "commit": "c637f6e73d06f36d933cc3edaf45111c06b79c18",
        "lang": "rust",
        "license": "MIT",
        "local_seed": (Path(_SEED_DIR) / "tokio") if _SEED_DIR else None,
        "exts": [".rs"],
    },
    "django": {
        "url": "https://github.com/django/django",
        "commit": "318a316a4c86a65bede68144f9546a6056d91379",
        "lang": "python",
        "license": "BSD-3-Clause",
        "local_seed": (Path(_SEED_DIR) / "django") if _SEED_DIR else None,
        "exts": [".py"],
    },
    "hugo": {
        "url": "https://github.com/gohugoio/hugo",
        "commit": "dfb35dcd7a9ab9a6d8b6c0829c312f2e4d5f8b0d",
        "lang": "go",
        "license": "Apache-2.0",
        "local_seed": (Path(_SEED_DIR) / "hugo") if _SEED_DIR else None,
        "exts": [".go"],
    },
    "elasticsearch": {
        "url": "https://github.com/elastic/elasticsearch",
        "commit": "fc85b9c0492265a20e90e18db4f496c5e2e4bf4a",
        "lang": "java",
        "license": "Elastic-2.0 / SSPL-1.0",
        "local_seed": (Path(_SEED_DIR) / "elasticsearch") if _SEED_DIR else None,
        "exts": [".java"],
    },
}

PROBE_REPOS = ["gson", "zod"]  # java/ts oracle quality is UNPROBED before this
MIN_RELIABLE_TARGETS = 20  # per repo, CALLS class; below this => 'unreliable'
MIN_USAGE_SURVIVORS = 15  # per repo; below this USAGE is dropped wholesale
MIN_NAME_LEN = 4

# Ubiquitous method names: even if in-graph unique, the suffix-matching C
# resolver can over-link receiver calls (`cfg.get(...)`) on foreign types.
STOPLIST = {
    "get", "set", "new", "init", "main", "run", "call", "next", "close",
    "open", "read", "write", "update", "delete", "add", "remove", "clone",
    "copy", "parse", "format", "items", "keys", "values", "append", "push",
    "pop", "insert", "contains", "equals", "hashcode", "tostring", "iterator",
    "size", "length", "name", "value", "type", "args", "apply", "bind",
    "then", "catch", "map", "filter", "reduce", "foreach", "test", "check",
    "handle", "process", "create", "build", "make", "start", "stop", "reset",
    "clear", "load", "save", "send", "fetch", "visit", "accept", "emit",
    "fire", "trigger", "from", "into", "index", "flush", "peek", "wrap",
}

# ---- fix (f/a): per-language stoplist of common stdlib / prototype names.
# Even a repo-unique definition with one of these names is poisoned: textual
# and suffix-match evidence cannot distinguish it from java.lang.Math.*,
# Number/String/Array.prototype.*, Rust std method calls, or Python builtins
# (measured false positives: gson 'toIntExact'/'concat', zod 'toFixed'/
# 'normalize', serde 'as_str').  Compared case-insensitively.
LANG_STDLIB = {
    "java": {
        # java.lang.Math / Integer / Long
        "toIntExact", "floorDiv", "floorMod", "addExact", "subtractExact",
        "multiplyExact", "negateExact", "incrementExact", "decrementExact",
        "toDegrees", "toRadians", "signum", "hypot", "nextUp", "nextDown",
        "copySign", "getExponent", "parseInt", "parseLong", "parseFloat",
        "parseDouble", "valueOf", "decode", "bitCount", "rotateLeft",
        "rotateRight", "reverseBytes", "highestOneBit", "lowestOneBit",
        "numberOfLeadingZeros", "numberOfTrailingZeros",
        # java.lang.String / CharSequence / StringBuilder
        "charAt", "codePointAt", "substring", "subSequence", "indexOf",
        "lastIndexOf", "startsWith", "endsWith", "isEmpty", "isBlank",
        "trim", "strip", "stripLeading", "stripTrailing", "toLowerCase",
        "toUpperCase", "concat", "matches", "split", "replace", "replaceAll",
        "replaceFirst", "join", "repeat", "chars", "lines", "compareTo",
        "compareToIgnoreCase", "equalsIgnoreCase", "contentEquals", "intern",
        "getBytes", "toCharArray", "deleteCharAt", "setLength", "setCharAt",
        "ensureCapacity",
        # java.util List/Map/Set/Collections/Arrays/Objects/Optional/Stream
        "addAll", "removeAll", "retainAll", "containsAll", "toArray",
        "asList", "sort", "binarySearch", "fill", "copyOf", "copyOfRange",
        "entrySet", "keySet", "getOrDefault", "putAll", "putIfAbsent",
        "merge", "compute", "computeIfAbsent", "computeIfPresent",
        "removeIf", "forEach", "stream", "parallelStream", "collect",
        "flatMap", "distinct", "limit", "skip", "findFirst", "findAny",
        "anyMatch", "allMatch", "noneMatch", "count", "sorted", "boxed",
        "mapToInt", "mapToLong", "mapToObj", "unmodifiableList",
        "unmodifiableMap", "unmodifiableSet", "singletonList",
        "singletonMap", "emptyList", "emptyMap", "emptySet",
        "requireNonNull", "requireNonNullElse", "isNull", "nonNull",
        "hashCode", "deepEquals", "deepHashCode", "deepToString",
        "ifPresent", "ifPresentOrElse", "orElse", "orElseGet",
        "orElseThrow", "ofNullable", "isPresent",
        # Throwable / Object / misc
        "getClass", "notifyAll", "printStackTrace", "getMessage",
        "getCause", "initCause", "addSuppressed", "getStackTrace",
        # java.lang.Number accessors (fix (m) backstop): any repo @Override
        # of these is textually indistinguishable from the boxed JDK call
        # (gson LazilyParsedNumber.longValue vs Long.longValue)
        "longValue", "intValue", "doubleValue", "floatValue", "shortValue",
        "byteValue",
        # not java.lang but textually indistinguishable from Guava/JDK
        # wrappers seen in the wild (gson 'unwrap' File-caller case)
        "unwrap", "toString",
    },
    "ts": {
        # Number / Math / global
        "toFixed", "toPrecision", "toExponential", "toLocaleString",
        "parseInt", "parseFloat", "isNaN", "isFinite", "isInteger",
        "isSafeInteger", "trunc", "sign", "cbrt", "clamp", "fround",
        # String.prototype
        "charAt", "charCodeAt", "codePointAt", "fromCharCode",
        "fromCodePoint", "normalize", "includes", "startsWith", "endsWith",
        "padStart", "padEnd", "repeat", "trim", "trimStart", "trimEnd",
        "toLowerCase", "toUpperCase", "toLocaleLowerCase",
        "toLocaleUpperCase", "substring", "substr", "localeCompare",
        "matchAll", "search", "replaceAll",
        # Array.prototype / Object / Promise / JSON
        "slice", "splice", "concat", "join", "reverse", "sort", "indexOf",
        "lastIndexOf", "find", "findIndex", "findLast", "findLastIndex",
        "some", "every", "flat", "flatMap", "fill", "copyWithin",
        "entries", "shift", "unshift", "reduceRight", "hasOwnProperty",
        "isPrototypeOf", "propertyIsEnumerable", "assign", "freeze",
        "isFrozen", "seal", "getOwnPropertyNames", "defineProperty",
        "getPrototypeOf", "setPrototypeOf", "fromEntries", "groupBy",
        "isArray", "stringify", "toJSON", "valueOf", "toString", "then",
        "finally", "allSettled", "race", "resolve", "reject",
        # DOM / runtime globals
        "querySelector", "querySelectorAll", "addEventListener",
        "removeEventListener", "setTimeout", "setInterval", "clearTimeout",
        "clearInterval", "structuredClone", "requestAnimationFrame",
    },
    "rust": {
        "unwrap", "unwrap_or", "unwrap_or_else", "unwrap_or_default",
        "unwrap_err", "unwrap_unchecked", "expect", "expect_err",
        "as_str", "as_ref", "as_mut", "as_slice", "as_bytes", "as_deref",
        "as_ptr", "to_string", "to_owned", "to_vec", "to_le_bytes",
        "to_be_bytes", "to_str", "clone_from", "iter_mut", "into_iter",
        "collect", "extend", "take", "replace", "swap", "split_at",
        "split_off", "split_whitespace", "starts_with", "ends_with",
        "is_empty", "is_some", "is_none", "is_ok", "is_err",
        "is_char_boundary", "ok_or", "ok_or_else", "and_then", "or_else",
        "map_or", "map_or_else", "map_err", "borrow", "borrow_mut",
        "try_borrow", "deref", "deref_mut", "drop", "default", "from_str",
        "parse", "chars", "bytes", "char_indices", "lines",
        "to_lowercase", "to_uppercase", "trim", "trim_start", "trim_end",
        "push_str", "with_capacity", "capacity", "shrink_to_fit",
        "truncate", "retain", "dedup", "first", "last", "get_mut",
        "get_or_insert_with", "entry", "or_insert", "or_insert_with",
        "contains_key", "eq_ignore_ascii_case", "fmt", "hash",
        "partial_cmp", "cmp", "try_from", "try_into", "into", "from",
        "type_id", "get_unchecked", "concat", "zip", "enumerate", "rev",
        "skip", "step_by", "chain", "cycle", "fold", "position",
        "find_map", "filter_map", "flat_map", "any", "all", "sum",
        "product", "min_by", "max_by", "min_by_key", "max_by_key", "nth",
        "count", "join", "display",
    },
    "python": {
        # builtins
        "isinstance", "issubclass", "getattr", "setattr", "hasattr",
        "delattr", "callable", "staticmethod", "classmethod", "property",
        "super", "vars", "print", "repr", "sorted", "reversed",
        "enumerate", "zip", "range", "iter", "next", "divmod", "round",
        "format", "hash", "frozenset", "bytearray", "memoryview",
        "compile", "exec", "eval", "globals", "locals", "breakpoint",
        # str / bytes methods
        "startswith", "endswith", "strip", "lstrip", "rstrip",
        "splitlines", "split", "rsplit", "join", "encode", "decode",
        "casefold", "title", "capitalize", "upper", "lower", "swapcase",
        "center", "ljust", "rjust", "zfill", "find", "rfind", "rindex",
        "count", "expandtabs", "partition", "rpartition", "maketrans",
        "translate", "isdigit", "isalpha", "isalnum", "isspace",
        "isupper", "islower", "istitle", "isidentifier", "isnumeric",
        "isdecimal", "isprintable", "isascii",
        # dict / set / list methods, datetime, copy
        "fromkeys", "setdefault", "popitem", "update", "extend",
        "discard", "symmetric_difference", "intersection", "union",
        "difference", "issubset", "issuperset", "copy", "deepcopy",
        "total_seconds", "timestamp", "strftime", "strptime",
        "fromtimestamp", "utcnow", "today",
    },
}
LANG_STDLIB_LOWER = {lang: {n.lower() for n in names}
                     for lang, names in LANG_STDLIB.items()}

# ---- fix (e): dependency direction.  Primary src tree per repo (verified
# against the pinned checkouts) + aux-subtree detection.
PRIMARY_SRC_RE = {
    "serde": re.compile(
        r"^(serde|serde_core|serde_derive|serde_derive_internals)/src/"),
    "flask": re.compile(r"^src/flask/"),
    "gson": re.compile(r"^gson/src/main/"),
    "zod": re.compile(r"^packages/zod/src/"),
    # larger-repo expansion (2026-07-04): primary source, test/generated excluded
    "tokio": re.compile(r"^tokio[^/]*/src/"),
    "django": re.compile(r"^django/"),
    "hugo": re.compile(r"^(?!.*(?:_test\.go$|/testdata/))"),  # Go: keep non-test .go
    "elasticsearch": re.compile(r"/src/main/"),  # Maven modules: **/src/main/java
}
AUX_NAMES = {
    "test", "tests", "test_suite", "testing", "bench", "benches",
    "benchmark", "benchmarks", "docs", "doc", "examples", "example",
}


def _aux_root(rel: str):
    """If rel lies under a test/bench/docs/examples subtree, return the
    path prefix up to and including the first aux component, else None.
    Any DIRECTORY component named like a test dir counts, wherever it sits
    (zod keeps tests at packages/zod/src/v4/classic/tests/)."""
    parts = rel.split("/")
    for i, comp in enumerate(parts[:-1]):
        c = comp.lower()
        if c in AUX_NAMES or c.startswith("test"):
            return "/".join(parts[:i + 1])
    return None


def _dir_ok(repo: str, tfile: str, caller_files) -> bool:
    """Fix (e): a target under an aux subtree is only valid if ALL claimed
    callers live inside the SAME aux subtree; a target outside aux subtrees
    must be defined under the repo's primary src tree."""
    root = _aux_root(tfile)
    if root is not None:
        pre = root + "/"
        return all(f == tfile or f.startswith(pre) for f in caller_files)
    return bool(PRIMARY_SRC_RE[repo].match(tfile))


# ---- fix (l): cross-version attribution.  A repo that VENDORS multiple
# versions of its own API (zod ships v3 and v4 side by side, with v4 split
# into a shared core plus the sibling 'classic' and 'mini' surfaces) defeats
# name-based attribution across the subtrees: a test importing 'zod/v4'
# textually calls '.extract(' but dispatches to the v4 property closure, not
# the same-named v3 method the oracle linked.  A target therefore only
# accepts callers from its own version subtree; the v4 shared substrate
# (core, locales, …) additionally accepts any v4 caller because classic and
# mini are built ON it (the reverse direction stays rejected — core never
# imports classic/mini).

def _zod_version_key(rel: str):
    m = re.match(r"^packages/zod/src/(v3|v4)(?:/|$)", rel)
    if not m:
        return ("out", "")
    if m.group(1) == "v3":
        return ("v3", "")
    m2 = re.match(r"^packages/zod/src/v4/(classic|mini)(?:/|$)", rel)
    return ("v4", m2.group(1) if m2 else "shared")


def _zod_version_violations(tfile: str, caller_files):
    tmaj, tsub = _zod_version_key(tfile)
    bad = []
    for f in sorted(caller_files):
        cmaj, csub = _zod_version_key(f)
        if cmaj != tmaj:
            bad.append(f)
        elif tmaj == "v4" and tsub in ("classic", "mini") and csub != tsub:
            bad.append(f)
    return bad


VERSION_ATTRIBUTION = {"zod": _zod_version_violations}


def _version_violations(repo: str, tfile: str, caller_files):
    """Fix (l): list of caller files in a DIFFERENT version subtree than the
    target definition (empty when the repo has no vendored versions)."""
    fn = VERSION_ATTRIBUTION.get(repo)
    return fn(tfile, caller_files) if fn else []


def _line_at(text: str, lineno: int) -> str:
    lines = text.splitlines()
    return lines[lineno - 1] if 0 < lineno <= len(lines) else ""


# lines that define (not call) a symbol — used to exclude definition lines
# from missed-caller candidates and from caller-span textual evidence.
DEF_RE = {
    "rust": r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?(const\s+)?fn\s+{n}\b",
    "python": r"^\s*(async\s+)?def\s+{n}\b",
    "java": r"^\s*(public|protected|private|static|final|abstract|synchronized|native|default|\s)*[\w<>\[\],.\s]+\s+{n}\s*\(",
    "ts": r"^\s*(export\s+)?(default\s+)?(declare\s+)?(async\s+)?(function\s+{n}\b|(const|let|var)\s+{n}\b|{n}\s*[(:=]\s*(async\s*)?(function\b|\())",
}
COMMENT_PREFIX = ("//", "#", "*", "/*", "*/", "///", "//!", "#!")

# ---- fix (i): textual def-multiplicity cross-check.  The oracle's def dump
# misses parameters, local closures, getter/setter pairs and sometimes whole
# methods, so "unique_name" was only oracle-relative.  These per-language
# regexes re-count DEFINITIONS textually over the whole repo; '{n}' is
# substituted via str.replace (NOT str.format — the patterns contain literal
# braces).  Over-counting rejects a good target (collateral, acceptable);
# under-counting merely weakens the defense — so every pattern is written to
# never match plain call statements.
_JAVA_STMT_KW = (r"(?!(?:return|throw|new|else|case|assert|break|continue|"
                 r"if|while|for|do|switch|import|package|yield)\b)")
TEXT_DEF_RE = {
    "rust": [r"(?<![\w:])fn\s+{n}\s*[(<]"],
    "python": [r"^\s*(?:async\s+)?def\s+{n}\s*\("],
    "java": [
        # methods incl. annotation elements and overloads: optional
        # annotations + modifiers + optional generic + ONE return-type token
        # (optional generics/arrays), then 'name(' — statement keywords are
        # excluded so 'return name(...)' never counts, and the type token
        # cannot contain a bare comma so multi-line call continuations
        # ('raw, getBoundFields(…') never count either.
        r"^\s*(?:@[\w.]+(?:\([^)]*\))?\s+)*"
        r"(?:(?:public|protected|private|static|final|abstract|synchronized"
        r"|native|strictfp|default)\s+)*(?:<[^>]*>\s+)?" + _JAVA_STMT_KW +
        r"[\w$][\w$.]*(?:<[^<>]*(?:<[^<>]*>)?[^<>]*>)?(?:\[\])*\s+{n}\s*\(",
    ],
    "ts": [
        # named function declarations (incl. nested / exported / default)
        r"\bfunction\s*\*?\s*{n}\s*[(<]",
        # const/let/var closures: `const name = (…) =>` / `= function` /
        # `= async <T>(` / `= x =>`
        r"\b(?:const|let|var)\s+{n}\s*(?::[^=;]+)?=\s*(?:async\s+)?"
        r"(?:function\b|\(|<[A-Za-z_$]|[\w$]+\s*=>)",
        # get/set accessors (`override get message() {`)
        r"^\s*(?:(?:public|private|protected|static|abstract|override|async)"
        r"\s+)*(?:get|set)\s+{n}\s*\(",
        # modifier-prefixed class methods
        r"^\s*(?:(?:public|private|protected|static|abstract|override)\s+)+"
        r"(?:async\s+)?{n}\s*(?:<[^>]*>)?\s*\(",
        # bare shorthand methods (class body / object literal): the line
        # must END with '{' — call statements end with ';' / ')' instead
        r"^\s*(?:async\s+)?{n}\s*(?:<[^>]*>)?\s*\([^()]*\)\s*"
        r"(?::[^;={}]+)?\{\s*$",
        # class-field / assignment closures: `name = (…) =>`
        r"^\s*(?:(?:public|private|protected|static|readonly)\s+)*{n}\s*=\s*"
        r"(?:async\s+)?(?:\(|[\w$]+\s*=>|function\b)",
        # ---- fix (k): hidden def forms the round-3 attack used.
        # PROPERTY-ASSIGNED closures (`inst.extract = (values, params) =>`,
        # `obj.name = function`, `x.name = async <T>(…`)
        r"[\w$)\]]\s*\.\s*{n}\s*=(?!=)\s*(?:async\s+)?"
        r"(?:function\b|\(|<[A-Za-z_$]|[\w$]+\s*=>)",
        # PROPERTY ALIASES (`inst.removeDefault = inst.unwrap;`): assigning
        # any identifier path to a same-named property re-binds the name —
        # textual evidence can no longer be attributed to the target
        r"[\w$)\]]\s*\.\s*{n}\s*=(?!=)\s*[\w$][\w$.]*\s*;\s*$",
        # object-literal / interface CALLABLE PROPERTIES
        # (`name: (…) => …`, `name: function`, `name: async (…`)
        r"(?<![\w$.?]){n}\s*:\s*(?:async\s+)?"
        r"(?:function\b|\([^()]*\)\s*(?::[^;={}]*?)?=>|<[A-Za-z_$][^<>]*>\s*\()",
        # single-line interface / overload METHOD SIGNATURES
        # (`removeDefault(): T;`) — ends with ';', never a call statement
        r"^\s*(?:readonly\s+)?{n}\s*(?:<[^<>]*>)?\s*\([^()]*\)\s*:[^;={}]*;\s*$",
        # multiline GENERIC signature openers: `extract<const U …>(` at line
        # start, ending with the opening paren (v3 class methods and v4
        # interface signatures both use this shape; a generic CALL with
        # explicit type args at line start ending in '(' is vanishingly rare)
        r"^\s*(?:async\s+)?{n}\s*<[^;={}]*\(\s*$",
    ],
}

# fix (i) part 2: parameter / local-binding shadowing near an evidence line.
# A binding named like the target declared in scope makes textual evidence
# worthless ('return message(val)' called a refine() parameter, not the
# ZodError getter).  Kinds carry per-language disambiguation handled in
# _shadow_hit (python assignment vs. kwarg continuation line).
SHADOW_WINDOW = 120  # lines scanned above (and including) the evidence line
SHADOW_RE = {
    "python": [
        ("py_assign", r"^\s*{n}\s*(?::[^=]+)?=[^=]"),
        ("", r"^\s*(?:async\s+)?def\s+\w+\s*\(.*?\b{n}\s*[,=:)]"),
        ("", r"\bfor\s+[^:]*?\b{n}\b[^:]*?\bin\b"),
        ("", r"\bas\s+{n}\b"),
        ("", r"\blambda\b[^:]*\b{n}\b\s*[:,=]"),
    ],
    "rust": [
        ("", r"\blet\s+(?:mut\s+)?{n}\b"),
        ("", r"[(,]\s*(?:mut\s+)?{n}\s*:"),          # fn param `name: T`
        ("", r"\|[^|()]*\b{n}\s*[,:|]"),             # closure param |name|
        ("", r"\bfor\s+(?:mut\s+)?{n}\b"),
        ("", r"\bif\s+let\s+[^=]*\b{n}\b"),
    ],
    "java": [
        # `Type name =` / `Type name)` / `Type name,` / enhanced-for
        # `Type name :` — anchored to line start or '(' ',' so that plain
        # dotted/bare calls never match.
        ("", r"(?:^\s*|[(,]\s*)(?:final\s+)?" + _JAVA_STMT_KW +
         r"[\w$][\w$.<>\[\]]*\s+{n}\s*[=,;):]"),
    ],
    "ts": [
        ("", r"\b(?:const|let|var)\b[^=;()]*\b{n}\b"),
        ("", r"\(([^()]*\b{n}\b[^()]*)\)\s*(?::[^={}()]*)?=>"),  # arrow params
        ("", r"(?<![\w$.]){n}\s*=>"),                # single-param arrow
        ("", r"\bfunction\b[^()]*\(([^()]*\b{n}\b)"),  # function params
        ("", r"\bcatch\s*\(\s*{n}\b"),
        ("", r"^\s*{n}\?\s*:"),                      # optional param decl
    ],
}


def _count_textual_defs(hits, lang, name):
    """Fix (i): count textual definition lines of <name> among the rg hits
    (unique (file, line) pairs; comment lines never count)."""
    esc = re.escape(name)
    res = [re.compile(p.replace("{n}", esc)) for p in TEXT_DEF_RE[lang]]
    ts_overload_sig = re.compile(r"^\s*(?:export\s+)?(?:declare\s+)?"
                                 r"(?:default\s+)?function\b")
    seen = set()
    for f, ln, txt in hits:
        if (f, ln) in seen:
            continue
        st = txt.strip()
        if st.startswith(COMMENT_PREFIX):
            continue
        # TS overload SIGNATURES ('function f(…): T;' — single line, no
        # body) declare, not define; only the implementation is a def.
        if lang == "ts" and ts_overload_sig.match(txt) \
                and st.endswith(";") and "{" not in txt:
            continue
        if any(r.search(txt) for r in res):
            seen.add((f, ln))
    return len(seen)


def _shadow_hit(file_lines, ev_ln, lang, name, skip_lines=(),
                src_file=None, target_file=None):
    """Fix (i): scan the SHADOW_WINDOW lines up to (and including) the
    evidence line for a parameter/local binding named <name>.  Returns
    (lineno, text) or None.  skip_lines excludes the target's own
    definition line when the evidence is same-file.

    A python `from <mod> import name [as name]` that resolves to the
    TARGET'S OWN module binds the target itself (flask's `import abort as
    abort` re-export idiom) — that is a reference, not a shadow; a rename
    (`import other as name`) or a foreign-module import still shadows."""
    esc = re.escape(name)
    res = [(kind, re.compile(p.replace("{n}", esc)))
           for kind, p in SHADOW_RE[lang]]
    own_import_re = None
    if lang == "python" and src_file is not None and target_file is not None:
        own_import_re = re.compile(r"^\s*from\s+(\S+)\s+import\s+(.*)$")
        self_bind_re = re.compile(
            rf"(?<!as\s)\b{esc}(?:\s+as\s+{esc})?(?=\s*(?:[,#)]|$))")
    lo = max(1, ev_ln - SHADOW_WINDOW)
    for i in range(lo, min(ev_ln, len(file_lines)) + 1):
        if i in skip_lines:
            continue
        txt = file_lines[i - 1]
        st = txt.strip()
        if st.startswith(COMMENT_PREFIX):
            continue
        for kind, r in res:
            if r.search(txt):
                if kind == "py_assign" and st.endswith(","):
                    continue  # kwarg continuation line, not a binding
                if own_import_re is not None:
                    m = own_import_re.match(txt)
                    if m and self_bind_re.search(m.group(2)) and \
                            _py_module_matches(m.group(1), src_file,
                                               target_file):
                        continue  # import of the target itself, no shadow
                return (i, st[:120])
    return None


# ---- fix (m): Java external-override defense.  A unique-name Method that
# carries @Override necessarily overrides a NON-repo (JDK/external)
# super-method: a repo-side super definition (class method OR interface
# signature) would textually define the name a second time and already fail
# unique_name / def-multiplicity.  For such targets, dotted call evidence is
# polymorphic ('number.longValue()' statically binds to java.lang.Number,
# not the repo override), so EVERY receiver on every evidence line must be
# statically the owner class or a repo type related to it.

_JAVA_DECL_HEAD_RE = re.compile(
    r"\b(?:class|interface|enum|record)\s+([\w$]+)([^{;]*)\{", re.S)
_JAVA_TYPE_GRAPH_CACHE = {}


def _strip_generics(s: str) -> str:
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"<[^<>]*>", "", s)
    return s


def _java_type_graph(repo_dir: Path):
    """{repo type simple name -> set of super/interface simple names},
    from a textual scan of every .java file (tests included — repo
    subclasses inheriting an override live there too)."""
    key = str(repo_dir)
    if key not in _JAVA_TYPE_GRAPH_CACHE:
        graph = {}
        for p in repo_dir.rglob("*.java"):
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            # comments first: 'This class holds …' in a javadoc otherwise
            # feeds the decl regex and swallows the real declaration
            text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
            text = re.sub(r"//[^\n]*", "", text)
            for m in _JAVA_DECL_HEAD_RE.finditer(text):
                head = _strip_generics(m.group(2))
                supers = set()
                for mm in re.finditer(
                        r"\b(?:extends|implements)\b([^{]*?)"
                        r"(?=\bextends\b|\bimplements\b|\bpermits\b|$)",
                        head):
                    for tok in mm.group(1).split(","):
                        tok = tok.strip().split(".")[-1].strip()
                        if re.fullmatch(r"[\w$]+", tok):
                            supers.add(tok)
                graph.setdefault(m.group(1), set()).update(supers)
        _JAVA_TYPE_GRAPH_CACHE[key] = graph
    return _JAVA_TYPE_GRAPH_CACHE[key]


def _java_ancestors(t: str, graph) -> set:
    seen, stack = set(), [t]
    while stack:
        for s in graph.get(stack.pop(), ()):
            if s not in seen:
                seen.add(s)
                stack.append(s)
    return seen


def _java_receiver_related(t: str, owner: str, graph) -> bool:
    """Receiver type <t> is acceptable iff it IS the owner, a repo subclass
    of the owner (inherits the override), or a repo-DECLARED supertype of
    the owner (closed world + unique name: in-repo dispatch through a repo
    supertype can only reach this one override).  External supertypes
    (java.lang.Number) are never in the graph and stay rejected."""
    if t == owner:
        return True
    if owner in _java_ancestors(t, graph):
        return True
    return t in graph and t in _java_ancestors(owner, graph)


def _java_is_override(file_lines, tline: int) -> bool:
    """@Override on the definition line or in the annotation/comment block
    directly above it."""
    if not (0 < tline <= len(file_lines)):
        return False
    if re.search(r"@Override\b", file_lines[tline - 1]):
        return True
    for j in range(tline - 2, max(-1, tline - 12), -1):
        st = file_lines[j].strip()
        if not st or st.startswith(COMMENT_PREFIX):
            continue
        if st.startswith("@"):
            if st.startswith("@Override"):
                return True
            continue
        break
    return False


_JAVA_RECV_DECL_TMPL = (
    r"(?:^\s*|[(,;]\s*)(?:final\s+)?" + _JAVA_STMT_KW +
    r"([\w$][\w$.]*(?:<[^<>]*(?:<[^<>]*>)?[^<>]*>)?(?:\[\])*)\s+{r}\s*[=,;):]")


def _java_recv_decl_type(lines, ev_ln: int, recv: str):
    """Nearest declared type of <recv> in the SHADOW_WINDOW above (and
    including) the evidence line; simple name, generics/arrays stripped."""
    r = re.compile(_JAVA_RECV_DECL_TMPL.replace("{r}", re.escape(recv)))
    best = None
    for i in range(max(1, ev_ln - SHADOW_WINDOW),
                   min(ev_ln, len(lines)) + 1):
        txt = lines[i - 1]
        if txt.strip().startswith(COMMENT_PREFIX):
            continue
        m = r.search(txt)
        if m:
            best = m.group(1)
    if best is None:
        return None
    return _strip_generics(best).replace("[]", "").split(".")[-1] or None


def _java_override_bad_receiver(lines, ev_ln: int, sym: str, tfile: str,
                                evf: str, owner: str, graph):
    """Fix (m): returns a detail string when ANY receiver of '.{sym}(' on
    the evidence line cannot be statically pinned to the owner class or a
    repo type related to it; None when the line is clean.  Bare same-file
    calls (fix (j) forbids bare cross-file Method evidence) have the owner
    instance as implicit receiver and are clean."""
    txt = lines[ev_ln - 1] if 0 < ev_ln <= len(lines) else ""
    esc = re.escape(sym)
    for m in re.finditer(rf"(?:([\w$]+)|([)\]\"']))\s*\.\s*{esc}\s*\(", txt):
        if m.group(2):
            return (f"unresolvable chained/literal receiver: "
                    f"{txt.strip()[:100]}")
        recv = m.group(1)
        if recv in ("this", "super"):
            encl = Path(evf).stem  # java: public type matches the filename
            if evf == tfile or _java_receiver_related(encl, owner, graph):
                continue
            return f"'{recv}' receiver of {encl} unrelated to {owner}"
        typ = _java_recv_decl_type(lines, ev_ln, recv)
        if typ is None:
            if recv[0].isupper() and _java_receiver_related(recv, owner,
                                                            graph):
                continue  # static-style call on the owner / a repo relative
            return f"receiver '{recv}' has no resolvable repo type"
        if _java_receiver_related(typ, owner, graph):
            continue
        return (f"receiver '{recv}' statically typed {typ}, "
                f"not {owner}/repo-related")
    return None


# ---------------------------------------------------------------- utilities

def sh(cmd, cwd=None, env=None, check=True):
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={p.returncode}): {' '.join(map(str, cmd))}\n"
            f"stderr: {p.stderr[:800]}")
    return p.stdout


def oracle_env(store: Path) -> dict:
    env = dict(os.environ)
    env["CBM_CACHE_DIR"] = str(store)
    env["CBM_LOG_LEVEL"] = "error"
    return env


def project_name(repo_dir: Path) -> str:
    """The reference oracle derives the project id from repo_path, ignoring the
    'project' argument (verified empirically): strip leading '/', '/'->'-'."""
    return str(repo_dir).lstrip("/").replace("/", "-")


def oracle_query(repo_dir: Path, store: Path, query: str) -> dict:
    out = sh([str(CBM), "cli", "query_graph",
              json.dumps({"project": project_name(repo_dir), "query": query})],
             env=oracle_env(store))
    return json.loads(out.strip().splitlines()[-1])


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"note": "Repos cloned locally for benchmarking only; not "
                    "redistributed. Licenses recorded per repo.",
            "repos": {}}


def save_manifest(m: dict):
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


# ------------------------------------------------------------------- setup

def cmd_setup(repos):
    ROOT.mkdir(parents=True, exist_ok=True)
    gi = ROOT / ".gitignore"
    if not gi.exists():
        gi.write_text("*/\n!MANIFEST.json\n!candidates.json\n")
    manifest = load_manifest()
    for name in repos:
        spec = REPOS[name]
        dest = ROOT / name
        pin = spec["commit"]
        if dest.exists():
            head = sh(["git", "-C", str(dest), "rev-parse", "HEAD"],
                      check=False).strip()
            if head.startswith(pin) or pin.startswith(head[:len(pin)]):
                print(f"[setup] {name}: already at {head[:12]} — skip")
                full = head
            else:
                print(f"[setup] {name}: wrong commit {head[:12]} (want {pin})"
                      f" — re-pinning")
                shutil.rmtree(dest)
                full = _clone(name, spec, dest)
        else:
            full = _clone(name, spec, dest)
        files = sh(["git", "-C", str(dest), "ls-files"]).strip().splitlines()
        manifest["repos"].setdefault(name, {})
        manifest["repos"][name].update({
            "repo": name, "url": spec["url"], "commit": full,
            "lang": spec["lang"], "license": spec["license"],
            "files": len(files), "path": f"realcorpus/{name}",
        })
        print(f"[setup] {name}: {full[:12]} · {len(files)} files · "
              f"{spec['license']}")
    save_manifest(manifest)
    print(f"[setup] wrote {MANIFEST}")


def _clone(name, spec, dest: Path) -> str:
    seed = spec.get("local_seed")
    if seed and seed.exists():
        head = sh(["git", "-C", str(seed), "rev-parse", "HEAD"],
                  check=False).strip()
        if head.startswith(spec["commit"]):
            print(f"[setup] {name}: cloning from local seed {seed}")
            sh(["git", "clone", "--no-hardlinks", "-q", str(seed), str(dest)])
            sh(["git", "-C", str(dest), "checkout", "-q", head])
            return head
        print(f"[setup] {name}: local seed at {head[:12]} != pin "
              f"{spec['commit']} — falling back to network")
    full = spec["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", full):
        raise RuntimeError(f"{name}: benchmark commit must be a full SHA-1")
    print(f"[setup] {name}: shallow-fetching {full[:12]} from {spec['url']}")
    dest.mkdir(parents=True)
    sh(["git", "init", "-q", str(dest)])
    sh(["git", "-C", str(dest), "remote", "add", "origin", spec["url"]])
    sh(["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", full])
    sh(["git", "-C", str(dest), "checkout", "-q", "FETCH_HEAD"])
    return full


# ------------------------------------------------------------------ oracle

def _rows(d: dict):
    for r in d.get("rows", []):
        yield r


def _labels(cell: str):
    try:
        return json.loads(cell)
    except Exception:
        return [cell]


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# The reference oracle hard-fails a query whose result exceeds 100k rows
# ("result exceeded 100k rows — use narrower filters or add LIMIT"), which
# elasticsearch's CALLS graph does. Paginate every bulk dump with
# SKIP/LIMIT (verified supported by the pinned binary) — deterministic
# order is not guaranteed per page, but we only ever aggregate the full
# result set, so page boundaries do not matter as long as pages are
# disjoint, which SKIP/LIMIT gives us on the oracle's stable ordering.
_PAGE = 50_000


def _paged_rows(repo_dir: Path, store: Path, base_q: str):
    skip = 0
    while True:
        d = oracle_query(repo_dir, store, f"{base_q} SKIP {skip} LIMIT {_PAGE}")
        rows = list(_rows(d))
        yield from rows
        if len(rows) < _PAGE:
            return
        skip += _PAGE


def dump_edges(repo_dir: Path, store: Path, etype: str):
    """All <src>-[:etype]->(Function|Method) edges with positions."""
    q = (f"MATCH (a)-[:{etype}]->(b:Function|Method) "
         "RETURN b.name, b.file_path, b.start_line, b.end_line, "
         "labels(a), a.name, a.file_path, a.start_line, a.end_line")
    edges = []
    for r in _paged_rows(repo_dir, store, q):
        edges.append({
            "target": {"name": r[0], "file": r[1],
                       "line": _int(r[2]), "end_line": _int(r[3])},
            "source": {"labels": _labels(r[4]), "name": r[5], "file": r[6],
                       "line": _int(r[7]), "end_line": _int(r[8])},
        })
    return edges


def dump_defs(repo_dir: Path, store: Path):
    q = ("MATCH (f:Function|Method) RETURN f.name, f.file_path, "
         "f.start_line, f.end_line, labels(f)")
    defs = []
    for r in _paged_rows(repo_dir, store, q):
        defs.append({"name": r[0], "file": r[1], "line": _int(r[2]),
                     "end_line": _int(r[3]), "labels": _labels(r[4])})
    return defs


def rg_scan(repo_dir: Path, name: str, exts):
    """All \\bname\\b matches in the repo. Returns [(relfile, lineno, text)]."""
    cmd = ["rg", "-n", "--no-heading", "--no-messages"]
    for e in exts:
        cmd += ["-g", f"*{e}"]
    cmd += ["-e", rf"\b{re.escape(name)}\b", str(repo_dir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    hits = []
    for line in p.stdout.splitlines():
        try:
            f, ln, txt = line.split(":", 2)
        except ValueError:
            continue
        rel = os.path.relpath(f, repo_dir)
        hits.append((rel, int(ln), txt))
    return hits


def unique_names(defs, class_names, lang):
    counts = {}
    for d in defs:
        counts[d["name"]] = counts.get(d["name"], 0) + 1
    stdlib = LANG_STDLIB_LOWER.get(lang, set())
    ok = set()
    for n, c in counts.items():
        if c != 1:
            continue
        if len(n) < MIN_NAME_LEN or n.lower() in STOPLIST:
            continue
        if n.lower() in stdlib:
            continue  # fix (f): stdlib/prototype name — evidence is poisoned
        if n.startswith("__") or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n):
            continue
        if n in class_names:
            continue  # constructor/class collision: overload-ambiguous
        ok.add(n)
    return ok, counts


def dump_class_names(repo_dir: Path, store: Path):
    q = "MATCH (c:Class|Interface|Enum|Type) RETURN c.name"
    d = oracle_query(repo_dir, store, q)
    return {r[0] for r in _rows(d)}


def dump_method_owners(repo_dir: Path, store: Path):
    """(method name, file, line) -> owner class name, via DEFINES_METHOD."""
    q = ("MATCH (c)-[:DEFINES_METHOD]->(m:Method) "
         "RETURN c.name, m.name, m.file_path, m.start_line")
    d = oracle_query(repo_dir, store, q)
    return {(r[1], r[2], _int(r[3])): r[0] for r in _rows(d)}


EDGE_INTERSECT_RUNS = 3  # index N times, keep only edges present in ALL runs
STABILITY_BUILDS = 3  # fix (n): full independent candidate builds; a target
#                       not byte-identical across ALL of them is dropped
#                       under rejection key 'nondeterministic_oracle'


def build_candidates(name: str, repo_dir: Path, store: Path, lang: str, exts):
    """Run the oracle + filters (a)-(c) for one repo.

    The C indexer's parallel edge resolution FLICKERS run-to-run even for
    some unique-name targets (measured: serde CALLS 398 vs 395 edges, two
    targets appearing/disappearing).  We therefore index the repo
    EDGE_INTERSECT_RUNS times into separate stores and keep only the edges
    present in every run — floor semantics tolerate the under-count, and
    the surviving candidates stop depending on a resolution race."""
    t0 = time.time()
    edges_by_etype = {}
    defs = class_names = owners = None
    for run_i in range(EDGE_INTERSECT_RUNS):
        st = store if run_i == 0 else Path(f"{store}-verify{run_i}")
        # fresh isolated store, always re-index (fast; avoids stale drift)
        if st.exists():
            shutil.rmtree(st)
        st.mkdir(parents=True)
        sh([str(CBM), "cli", "index_repository",
            json.dumps({"project": name, "repo_path": str(repo_dir)})],
           env=oracle_env(st))
        for etype in ("CALLS", "USAGE", "IMPORTS"):
            keyed = {}
            for e in dump_edges(repo_dir, st, etype):
                k = (e["target"]["name"], e["target"]["file"],
                     e["target"]["line"], e["source"]["name"],
                     e["source"]["file"], e["source"]["line"])
                keyed.setdefault(k, e)
            if run_i == 0:
                edges_by_etype[etype] = keyed
            else:
                edges_by_etype[etype] = {
                    k: v for k, v in edges_by_etype[etype].items()
                    if k in keyed}
        if run_i == 0:
            defs = dump_defs(repo_dir, st)
            class_names = dump_class_names(repo_dir, st)
            owners = dump_method_owners(repo_dir, st)
        else:
            shutil.rmtree(st)
    uniq, def_counts = unique_names(defs, class_names, lang)
    def_labels = {(d["name"], d["file"], d["line"]): d["labels"]
                  for d in defs}
    file_cache = {}

    def file_text(rel):
        if rel not in file_cache:
            try:
                file_cache[rel] = (repo_dir / rel).read_text(errors="replace")
            except OSError:
                file_cache[rel] = ""
        return file_cache[rel]

    lines_cache = {}

    def file_lines(rel):
        if rel not in lines_cache:
            lines_cache[rel] = file_text(rel).splitlines()
        return lines_cache[rel]

    out = {"edge_types": {}, "stats": {
        "functions_methods": len(defs),
        "class_like_names": len(class_names),
        "unique_name_targets": len(uniq),
        "edge_intersection_runs": EDGE_INTERSECT_RUNS,
        "index_and_dump_seconds": None,
    }}

    for etype in ("CALLS", "USAGE", "IMPORTS"):
        # MEASURED: the C indexer's edge attribution FLICKERS run-to-run
        # (parallel-resolution race) — worst for ambiguous names, but a few
        # unique-name serde targets flicker too (round-3 audit: serde CALLS
        # 'ident' appeared/disappeared across regenerations).  'edges' is
        # already the INTERSECTION over EDGE_INTERSECT_RUNS independent
        # index runs; fix (n) additionally intersects whole TARGETS over
        # STABILITY_BUILDS independent builds in cmd_oracle.  The two edge
        # COUNTERS below can still flicker by ±1 (an intersected edge into
        # a target that never survives) — they are volatile diagnostics,
        # not ground truth.
        seen_e = set(edges_by_etype[etype])
        edges = list(edges_by_etype[etype].values())
        n_edges_unique = sum(1 for k in seen_e if k[0] in uniq)
        by_target = {}
        for e in edges:
            t = e["target"]
            if t["name"] not in uniq:
                continue
            key = (t["name"], t["file"], t["line"])
            by_target.setdefault(key, {"target": t, "sources": []})
            by_target[key]["sources"].append(e["source"])

        survivors, rejected = [], {"module_line0": 0, "rg_caller_unverified": 0,
                                   "rg_missed_beyond_tolerance": 0,
                                   "rg_no_hits": 0,
                                   "owner_class_unmentioned": 0,
                                   "direction_outside_primary_src": 0,
                                   "private_cross_file": 0,
                                   "foreign_static_import": 0,
                                   "shadowed_or_multidef": 0,
                                   "cross_version_attribution": 0,
                                   "external_override_receiver": 0,
                                   "nondeterministic_oracle": 0}
        for (tname, tfile, tline), grp in sorted(by_target.items()):
            srcs = _dedup_sources(grp["sources"])
            # fix (e): dependency direction — aux-subtree targets must have
            # all callers inside the same aux subtree; everything else must
            # be defined under the primary src tree.
            if not _dir_ok(name, tfile, {s["file"] for s in srcs}):
                rejected["direction_outside_primary_src"] += 1
                continue
            # fix (l): cross-version attribution — every caller must live
            # in the SAME version subtree as the target definition.
            if _version_violations(name, tfile, {s["file"] for s in srcs}):
                rejected["cross_version_attribution"] += 1
                continue
            # filter (b): Module nodes at line 0 => macro/module attribution
            if any("Module" in s["labels"] and s["line"] == 0 for s in srcs):
                rejected["module_line0"] += 1
                continue
            # filter (a2): suffix-match receiver over-link guard.  If the
            # target is a METHOD of class C, a real caller's file must
            # mention C somewhere; `this.factories.addAll(...)` (List.addAll)
            # got linked to JsonArray.addAll otherwise (probe finding, gson).
            owner = owners.get((tname, tfile, tline))
            if owner and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", owner or ""):
                owner_re = re.compile(rf"\b{re.escape(owner)}\b")
                if any(s["file"] != tfile
                       and not owner_re.search(file_text(s["file"]))
                       for s in srcs):
                    rejected["owner_class_unmentioned"] += 1
                    continue
            is_method = (owner is not None or "Method" in
                         def_labels.get((tname, tfile, tline), []))
            # fix (f): 'private' targets (java/ts) cannot have callers in
            # other files (gson toIntExact: private static, yet the oracle
            # claimed a caller in JavaTimeTypeAdapters.java).
            if lang in ("java", "ts") and re.search(
                    r"\bprivate\b", _line_at(file_text(tfile), tline)):
                if any(s["file"] != tfile for s in srcs):
                    rejected["private_cross_file"] += 1
                    continue
            # fix (f): Java — a claimed caller whose file static-imports
            # <name> from a DIFFERENT class is using that other symbol.
            if lang == "java":
                imp_re = re.compile(
                    rf"^\s*import\s+static\s+([\w.]+)\.{re.escape(tname)}\s*;",
                    re.M)
                foreign = False
                for s in srcs:
                    if s["file"] == tfile:
                        continue
                    m = imp_re.search(file_text(s["file"]))
                    if m and m.group(1).split(".")[-1] != (owner or ""):
                        foreign = True
                        break
                if foreign:
                    rejected["foreign_static_import"] += 1
                    continue
            hits = rg_scan(repo_dir, tname, exts)
            if not hits:
                rejected["rg_no_hits"] += 1
                continue
            # fix (i): textual def-multiplicity cross-check — the oracle's
            # def dump misses local closures, getter/setter pairs and even
            # whole methods; if the TEXT of the repo defines the name more
            # often than the oracle knows, "unique_name" is a lie.
            n_txt_defs = _count_textual_defs(hits, lang, tname)
            n_oracle_defs = def_counts.get(tname, 0)
            if n_txt_defs > n_oracle_defs:
                rejected["shadowed_or_multidef"] += 1
                continue
            pat = DEF_RE[lang].format(n=re.escape(tname))
            def_re = re.compile(pat)
            ver, evidence, why = _verify_sources(srcs, hits, tname,
                                                 grp["target"], def_re,
                                                 etype, is_method, lang,
                                                 file_lines)
            if ver is None:
                rejected[why] += 1
                continue
            # fix (m): @Override of a non-repo super-method — every
            # evidence receiver must statically be the owner class or a
            # repo type related to it, else the evidence is polymorphic
            # (gson 'number.longValue()' binds java.lang.Number).
            if lang == "java" and is_method and \
                    _java_is_override(file_lines(tfile), tline):
                graph = _java_type_graph(repo_dir)
                own = owner or Path(tfile).stem
                bad = None
                for evd in evidence:
                    evf, evl = evd["at"].rsplit(":", 1)
                    bad = _java_override_bad_receiver(
                        file_lines(evf), int(evl), tname, tfile, evf, own,
                        graph)
                    if bad:
                        break
                if bad:
                    rejected["external_override_receiver"] += 1
                    continue
            missed = 0
            tol = max(3, 2 * len(srcs))
            if etype == "CALLS":
                missed = _missed_call_sites(hits, srcs, grp["target"], tname,
                                            def_re, is_method)
                if missed > tol:
                    rejected["rg_missed_beyond_tolerance"] += 1
                    continue
            # fix (h): File/Module oracle sources are NOT function callers;
            # they go into 'file_evidence' (matched by file path only) and
            # never into expect_members.
            fn_callers = [v for v in ver
                          if not ({"File", "Module"} & set(v["labels"]))]
            file_srcs = [v for v in ver
                         if {"File", "Module"} & set(v["labels"])]
            survivors.append({
                "symbol": tname,
                "file": tfile, "line": tline,
                "location": f"{tfile}:{tline}",
                "edge_type": etype,
                "callers": ver,
                "min_count": len(ver),
                "expect_members": sorted({v["name"] for v in fn_callers}),
                "file_evidence": sorted({v["file"] for v in file_srcs}),
                "semantics": "floor",  # (d): min_count + subset, NEVER exact
                "filters": {
                    "unique_name": True,
                    "module_line0_sources": False,
                    "direction_ok": True,
                    "owner_class": owner,
                    "owner_class_mentioned_by_all_callers": bool(owner),
                    "is_method": is_method,
                    "rg_verified_callers": len(ver),
                    "rg_missed_call_sites": missed,
                    "rg_missed_tolerance": tol,
                    "textual_def_count": n_txt_defs,
                    "oracle_def_count": n_oracle_defs,
                },
                "rg_evidence": evidence,
            })
        out["edge_types"][etype] = {
            "oracle_edges": n_edges_unique,
            "oracle_edges_all_names_intersected": len(seen_e),
            "oracle_targets_unique_name": len(by_target),
            "surviving_targets": len(survivors),
            "rejected": rejected,
            "targets": survivors,
        }
        # NOTE: the fix (g) USAGE floor is applied in cmd_oracle AFTER the
        # fix (n) stability intersection, so it sees the final count.
    out["stats"]["index_and_dump_seconds"] = round(time.time() - t0, 2)
    return out


def _apply_usage_floor(block):
    """Fix (g): USAGE below the reliability floor is dropped wholesale."""
    n = block["surviving_targets"]
    if not block.get("dropped") and n < MIN_USAGE_SURVIVORS:
        block.update({
            "dropped": True,
            "drop_reason": (f"usage_survivors_{n}_below_"
                            f"{MIN_USAGE_SURVIVORS}"),
            "surviving_before_drop": n,
            "surviving_targets": 0,
            "targets": [],
        })


def _stability_intersect(builds):
    """Fix (n): keep only targets that are BYTE-IDENTICAL (full dict —
    every field is a deterministic function of the intersected edge set)
    across all STABILITY_BUILDS independent builds; everything else is the
    oracle's parallel-resolution race showing through (serde CALLS 'ident')
    and is dropped under 'nondeterministic_oracle'.  Mutates and returns
    builds[0]."""
    res = builds[0]
    for etype, block in res["edge_types"].items():
        canon = {(t["symbol"], t["file"], t["line"]):
                 json.dumps(t, sort_keys=True) for t in block["targets"]}
        stable = set(canon)
        for other in builds[1:]:
            om = {(t["symbol"], t["file"], t["line"]):
                  json.dumps(t, sort_keys=True)
                  for t in other["edge_types"][etype]["targets"]}
            stable = {k for k in stable if om.get(k) == canon[k]}
        kept = [t for t in block["targets"]
                if (t["symbol"], t["file"], t["line"]) in stable]
        block["rejected"]["nondeterministic_oracle"] = \
            len(block["targets"]) - len(kept)
        block["targets"] = kept
        block["surviving_targets"] = len(kept)
    return res


def _dedup_sources(sources):
    seen, out = set(), []
    for s in sources:
        k = (s["name"], s["file"], s["line"])
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    # the parallel C indexer emits edges in nondeterministic order; sort so
    # candidates.json is byte-stable run-to-run (verified: sets are identical)
    out.sort(key=lambda s: (s["file"], s["line"], s["name"]))
    return out


def _call_pattern(tname, is_method):
    """Fix (f)/2d: for a plain Function (non-Method) target, a dotted
    receiver call `x.name(` is evidence for SOME OTHER symbol (zod 'toFixed'
    linked Number.prototype.toFixed call sites); only `name(` NOT preceded
    by `.` or a word char counts.  Methods keep the permissive form."""
    esc = re.escape(tname)
    if is_method:
        return re.compile(rf"\b{esc}\s*\(")
    return re.compile(rf"(?<![.\w]){esc}\s*\(")


def _py_module_matches(mod: str, src_file: str, target_file: str) -> bool:
    """Python analogue of the Java foreign-static-import defense: does the
    <mod> in `from <mod> import name` plausibly resolve to the target's own
    module?  'from flask.globals import app_ctx' is NOT evidence for a
    same-named pytest fixture defined in tests/conftest.py."""
    t = target_file
    if t.endswith(".py"):
        t = t[:-3]
    if t.endswith("/__init__"):
        t = t[:-len("/__init__")]
    frag = mod.lstrip(".").replace(".", "/")
    if not frag:  # `from . import x` — same package only
        src_dir = os.path.dirname(src_file)
        return src_dir == os.path.dirname(target_file) or src_dir == t
    return t == frag or t.endswith("/" + frag)


def _make_evidence_check(tname, is_method, etype, lang, target_file):
    """Fix (g): reference-shaped evidence.  CALLS: call position only.
    USAGE/IMPORTS: call position, import/use position, or qualified
    reference.  Parameter names, struct field defs and macro token lists
    match none of these.  The import form must hit the BINDING clause, not
    a quoted module string (`import { type } from "arktype"` is not a
    reference to a local symbol named arktype); Python from-imports must
    point at the target's own module (fix (f) analogue of the Java
    foreign-static-import defense).

    Fix (j): for METHOD targets, cross-file call evidence must be dotted/
    qualified ('.name(' / '::name(' / 'super().name('); bare 'name(' only
    counts when the evidence file IS the target's own file (the same-file
    shadowing scan in _verify_sources must additionally be clean)."""
    esc = re.escape(tname)
    if is_method:
        dotted_re = re.compile(rf"(?:\.|::)\s*{esc}\s*\(")
        bare_re = re.compile(rf"\b{esc}\s*\(")

        def call_ok(txt, src_file):
            if dotted_re.search(txt):
                return True
            return src_file == target_file and bool(bare_re.search(txt))
    else:
        plain_re = _call_pattern(tname, False)

        def call_ok(txt, src_file):
            return bool(plain_re.search(txt))

    if etype == "CALLS":
        return call_ok
    use_re = re.compile(rf"^\s*(pub(\([^)]*\))?\s+)?use\b[^;]*\b{esc}\b")
    imp_re = re.compile(rf"^\s*import\b[^;\"']*\b{esc}\b")
    from_re = re.compile(rf"^\s*from\s+(\S+)\s+import\b.*\b{esc}\b")
    qual_res = [re.compile(rf"::\s*{esc}\b")]
    if is_method:
        # method reference without parens (`obj.method` passed as value)
        qual_res.append(re.compile(rf"\.\s*{esc}\b"))

    def check(txt, src_file):
        if call_ok(txt, src_file) or use_re.search(txt) or imp_re.search(txt):
            return True
        m = from_re.match(txt)
        if m:
            if lang != "python":
                return True
            return _py_module_matches(m.group(1), src_file, target_file)
        return any(p.search(txt) for p in qual_res)

    return check


def _verify_sources(srcs, hits, tname, target, def_re, etype, is_method,
                    lang, file_lines):
    """Filter (c) part 1: every claimed source needs a textual occurrence of
    the target name inside its line span (or anywhere in its file when the
    source is a File/Module without a span).  For CALLS the evidence line
    must contain an actual call pattern (fix 2d: non-dotted for plain
    functions; fix (j): dotted/qualified for methods unless same-file); for
    USAGE/IMPORTS it must be reference-shaped (fix (g)).  Comment lines and
    definition-shaped lines never count.  Fix (i): a parameter/local
    binding named <tname> declared near the accepted evidence line rejects
    the whole target.  Returns (verified, evidence, reason); verified is
    None => auto-REJECT the whole target under rejection key <reason>."""
    by_file = {}
    for f, ln, txt in hits:
        by_file.setdefault(f, []).append((ln, txt))
    accept = _make_evidence_check(tname, is_method, etype, lang,
                                  target["file"])
    tgt_span = (target["line"],
                target["end_line"] if target["end_line"] >= target["line"]
                else target["line"])
    verified, evidence = [], []
    for s in srcs:
        fhits = by_file.get(s["file"], [])
        span = None
        if s["line"] > 0 and not ({"File", "Module"} & set(s["labels"])):
            end = s["end_line"] if s["end_line"] >= s["line"] else s["line"]
            span = (s["line"], end)
        found = None
        for ln, txt in fhits:
            if span and not (span[0] <= ln <= span[1]):
                continue
            # the target's own definition span is never caller evidence
            # (TS shorthand methods `name() {` escape DEF_RE; recursion is
            # not an external caller — caught for zod 'arktype')
            if s["file"] == target["file"] \
                    and tgt_span[0] <= ln <= tgt_span[1]:
                continue
            if txt.strip().startswith(COMMENT_PREFIX):
                continue
            if not accept(txt, s["file"]):
                continue
            # a definition-shaped line is not evidence of a call/usage
            if def_re.match(txt):
                continue
            found = (ln, txt.strip())
            break
        if found is None:
            return None, {"unverified_caller": s}, "rg_caller_unverified"
        # fix (i): a parameter/local binding named <tname> in scope near
        # the evidence line makes the evidence worthless (zod v3 'message':
        # 'return message(val)' called a shadowing refine() parameter).
        skip = ({target["line"]} if s["file"] == target["file"] else ())
        sh = _shadow_hit(file_lines(s["file"]), found[0], lang, tname, skip,
                         src_file=s["file"], target_file=target["file"])
        if sh is not None:
            return None, {"shadowed_evidence": {
                "caller": s["name"], "at": f"{s['file']}:{found[0]}",
                "binding_at": f"{s['file']}:{sh[0]}", "binding": sh[1],
            }}, "shadowed_or_multidef"
        verified.append({"name": s["name"], "file": s["file"],
                         "line": s["line"], "labels": s["labels"]})
        evidence.append({"caller": s["name"],
                         "at": f"{s['file']}:{found[0]}",
                         "text": found[1][:160]})
    return verified, evidence, ""


def _missed_call_sites(hits, srcs, target, tname, def_re, is_method):
    """Filter (c) part 2: textual call sites that no oracle source span
    covers and that are not the definition/import lines.  Uses the same
    call shape as verification (fix 2d: dotted receiver calls are not call
    sites of a plain function)."""
    spans = {}
    for s in srcs:
        if s["line"] > 0 and not ({"File", "Module"} & set(s["labels"])):
            end = s["end_line"] if s["end_line"] >= s["line"] else s["line"]
            spans.setdefault(s["file"], []).append((s["line"], end))
        else:
            spans.setdefault(s["file"], []).append((1, 10**9))
    call_re = _call_pattern(tname, is_method)
    tgt_span = (target["line"],
                target["end_line"] if target["end_line"] >= target["line"]
                else target["line"])
    missed = 0
    for f, ln, txt in hits:
        st = txt.strip()
        if st.startswith(COMMENT_PREFIX):
            continue
        if not call_re.search(txt):
            continue
        if def_re.match(txt):
            continue
        if re.match(r"^\s*(use |import |from |pub use )", txt):
            continue
        if f == target["file"] and tgt_span[0] <= ln <= tgt_span[1]:
            continue  # recursion inside the definition doesn't lower a floor
        if any(a <= ln <= b for a, b in spans.get(f, [])):
            continue
        missed += 1
    return missed


def cmd_oracle(repos):
    if not (CBM.exists() and os.access(CBM, os.X_OK)):
        sys.exit(f"reference oracle binary missing/not executable: {CBM}")
    if shutil.which("rg") is None:
        sys.exit("ripgrep (rg) is required for filter (c)")
    manifest = load_manifest()
    result = {}
    if CANDIDATES.exists():
        result = json.loads(CANDIDATES.read_text()).get("repos", {})
    for name in repos:
        repo_dir = ROOT / name
        if not repo_dir.exists():
            sys.exit(f"[oracle] {name}: {repo_dir} missing — run setup first")
        spec = REPOS[name]
        store = ROOT / ".cbm-store" / name
        print(f"[oracle] {name}: indexing (isolated store {store}; "
              f"{STABILITY_BUILDS} independent builds × "
              f"{EDGE_INTERSECT_RUNS} index runs) …")
        builds = []
        for b in range(STABILITY_BUILDS):
            st = store if b == 0 else Path(f"{store}-stab{b}")
            builds.append(build_candidates(name, repo_dir, st, spec["lang"],
                                           spec["exts"]))
            if b:
                shutil.rmtree(st, ignore_errors=True)
        res = _stability_intersect(builds)  # fix (n): flicker-drop
        for et, dd in res["edge_types"].items():
            if et == "USAGE":
                _apply_usage_floor(dd)  # fix (g), after the flicker-drop
        result[name] = res
        manifest["repos"].setdefault(name, {})
        usage = res["edge_types"].get("USAGE", {})
        manifest["repos"][name]["usage_dropped"] = bool(usage.get("dropped"))
        if usage.get("dropped"):
            manifest["repos"][name]["usage_drop_reason"] = usage["drop_reason"]
        for et, d in res["edge_types"].items():
            extra = (f" — DROPPED ({d['drop_reason']})"
                     if d.get("dropped") else "")
            print(f"[oracle] {name} {et}: {d['oracle_edges']} oracle edges → "
                  f"{d['oracle_targets_unique_name']} unique-name targets → "
                  f"{d['surviving_targets']} survive filters{extra} "
                  f"(rejected: {d['rejected']})")
    CANDIDATES.write_text(json.dumps({
        "generated_unix": int(time.time()),
        "oracle": "reference oracle (pinned, f0c9be1)",
        "semantics": "floor: min_count + expect_members subset; an agent "
                     "finding MORE true callers must not fail",
        "never_use": ["HANDLES", "SEMANTICALLY_RELATED", "SIMILAR_TO"],
        # fix (n): everything OUTSIDE these fields — in particular every
        # 'targets' array — must be byte-identical across consecutive full
        # regenerations (verified by regenerating 3× and diffing).
        "volatile_diagnostics": [
            "generated_unix",
            "repos.*.stats.index_and_dump_seconds",
            "repos.*.edge_types.*.oracle_edges",
            "repos.*.edge_types.*.oracle_edges_all_names_intersected",
            "repos.*.edge_types.*.oracle_targets_unique_name",
            "repos.*.edge_types.*.rejected",
            # pre-drop counts of a WHOLESALE-DROPPED USAGE block: a flaky
            # target inside a dropped block never reaches any target set,
            # but its presence flickers these two fields (measured: serde
            # USAGE 5 vs 6 pre-drop survivors across regenerations)
            "repos.*.edge_types.USAGE.surviving_before_drop",
            "repos.*.edge_types.USAGE.drop_reason",
        ],
        "repos": result,
    }, indent=2) + "\n")
    print(f"[oracle] wrote {CANDIDATES}")
    save_manifest(manifest)


# ------------------------------------------------------------------- probe

def cmd_probe(repos=None):
    repos = repos or PROBE_REPOS
    if not CANDIDATES.exists():
        sys.exit("[probe] candidates.json missing — run oracle first")
    data = json.loads(CANDIDATES.read_text())
    manifest = load_manifest()
    rng = random.Random(42)
    for name in repos:
        if name not in data["repos"]:
            print(f"[probe] {name}: no oracle data — skipped")
            continue
        res = data["repos"][name]
        calls = res["edge_types"]["CALLS"]
        n_calls = calls["surviving_targets"]
        reliable = n_calls >= MIN_RELIABLE_TARGETS
        verdict = "reliable" if reliable else "unreliable"
        print(f"\n[probe] {name}: CALLS targets surviving = {n_calls} "
              f"(threshold {MIN_RELIABLE_TARGETS}) → {verdict}")
        for et in ("USAGE", "IMPORTS"):
            d = res["edge_types"][et]
            print(f"[probe] {name} {et}: surviving = {d['surviving_targets']}"
                  f" (rejected: {d['rejected']})")
        manifest["repos"].setdefault(name, {})
        manifest["repos"][name]["oracle_reliability"] = verdict
        manifest["repos"][name]["surviving_targets"] = {
            et: res["edge_types"][et]["surviving_targets"]
            for et in res["edge_types"]}
        # spot-check 5 targets: print definition + one verified caller snippet
        pool = calls["targets"]
        sample = rng.sample(pool, min(5, len(pool)))
        repo_dir = ROOT / name
        for t in sample:
            print(f"\n[probe] SPOT-CHECK {name} :: {t['symbol']} "
                  f"({t['location']}) callers={t['min_count']}")
            print(_snippet(repo_dir, t["file"], t["line"], "  def> "))
            ev = t["rg_evidence"][0] if t["rg_evidence"] else None
            if ev:
                f, ln = ev["at"].rsplit(":", 1)
                print(f"  caller {ev['caller']} @ {ev['at']}")
                print(_snippet(repo_dir, f, int(ln), "  use> "))
    save_manifest(manifest)
    print(f"\n[probe] reliability written to {MANIFEST}")


def _snippet(repo_dir: Path, rel: str, line: int, prefix: str, ctx=1):
    try:
        lines = (repo_dir / rel).read_text(errors="replace").splitlines()
    except OSError:
        return f"{prefix}<unreadable {rel}>"
    lo, hi = max(0, line - 1 - ctx), min(len(lines), line + ctx)
    return "\n".join(f"{prefix}{i+1:5d}| {lines[i]}" for i in range(lo, hi))


# ------------------------------------------------------------------- audit

def cmd_audit(repos=None) -> int:
    """Fix 5: automated adversarial gate.  Re-runs the systemic scans the
    manual audit used, over the WRITTEN candidates.json (not in-process
    state), and returns the number of hits.  'all' fails when this fails.
    Round-2 additions: scan 5 (textual def-multiplicity, fix (i)) and
    scan 6 (Method cross-file evidence shape + evidence-line shadowing,
    fixes (i)+(j)).
    Round-3 additions: scan 5 now counts the fix (k) hidden TS def forms
    (property-assigned closures/aliases, object-literal callables,
    interface signatures); scan 7 (cross-version attribution, fix (l));
    scan 8 (Java external-override evidence receivers, fix (m))."""
    if not CANDIDATES.exists():
        sys.exit("[audit] candidates.json missing — run oracle first")
    data = json.loads(CANDIDATES.read_text())
    repos = [r for r in (repos or list(REPOS)) if r in data["repos"]]
    hits = []

    def hit(repo, et, t, check, detail):
        hits.append((repo, et, t["symbol"], t["location"], check, detail))

    rg_cache, lines_cache = {}, {}

    def cached_rg(repo_dir, sym, exts):
        key = (str(repo_dir), sym)
        if key not in rg_cache:
            rg_cache[key] = rg_scan(repo_dir, sym, exts)
        return rg_cache[key]

    def cached_lines(repo_dir, rel):
        key = (str(repo_dir), rel)
        if key not in lines_cache:
            try:
                lines_cache[key] = (repo_dir / rel).read_text(
                    errors="replace").splitlines()
            except OSError:
                lines_cache[key] = []
        return lines_cache[key]

    for name in repos:
        lang = REPOS[name]["lang"]
        exts = REPOS[name]["exts"]
        repo_dir = ROOT / name
        stdlib = LANG_STDLIB_LOWER.get(lang, set())
        for et, dd in data["repos"][name]["edge_types"].items():
            for t in dd.get("targets", []):
                sym, tfile = t["symbol"], t["file"]
                caller_files = ({c["file"] for c in t["callers"]}
                                | set(t.get("file_evidence", [])))
                # scan 1: dependency direction
                if not _dir_ok(name, tfile, caller_files):
                    bad = sorted(f for f in caller_files
                                 if not f.startswith(
                                     (_aux_root(tfile) or "\0") + "/"))
                    hit(name, et, t, "direction",
                        f"target outside primary src or aux callers escape "
                        f"the subtree: {bad[:3]}")
                # scan 2: stdlib/builtin name
                if sym.lower() in stdlib or sym.lower() in STOPLIST:
                    hit(name, et, t, "stdlib_name",
                        f"'{sym}' is a {lang} stdlib/stoplist name")
                # scan 3: private visibility (java/ts)
                if lang in ("java", "ts"):
                    try:
                        text = (repo_dir / tfile).read_text(errors="replace")
                    except OSError:
                        text = ""
                    if re.search(r"\bprivate\b", _line_at(text, t["line"])) \
                            and any(f != tfile for f in caller_files):
                        hit(name, et, t, "private_visibility",
                            "private target with cross-file callers")
                # scan 4: File/Module pseudo-callers inside expect_members
                fm_names = {c["name"] for c in t["callers"]
                            if {"File", "Module"} & set(c["labels"])}
                for m in t.get("expect_members", []):
                    if m in fm_names or "/" in m or \
                            re.search(r"\.[A-Za-z]{1,4}$", m):
                        hit(name, et, t, "pseudo_caller_in_expect_members",
                            f"member '{m}' is a File/Module pseudo-caller")
                # scan 5 (fix (i)): textual def-multiplicity — the repo TEXT
                # must not define the name more often than the oracle knew
                # (parameters/local closures/getter-setter pairs/indexer-
                # missed methods defeat oracle-relative uniqueness).
                n_txt = _count_textual_defs(cached_rg(repo_dir, sym, exts),
                                            lang, sym)
                n_orc = t.get("filters", {}).get("oracle_def_count", 1)
                if n_txt > n_orc:
                    hit(name, et, t, "shadowed_or_multidef",
                        f"{n_txt} textual definitions of '{sym}' vs "
                        f"{n_orc} oracle definition(s)")
                # scan 6 (fixes (i)+(j)): per-evidence-line checks — cross-
                # file Method evidence must be dotted/qualified, and no
                # parameter/local binding named <sym> may be in scope near
                # any evidence line.
                is_method = t.get("filters", {}).get("is_method", False)
                esc = re.escape(sym)
                dotted_call = re.compile(rf"(?:\.|::)\s*{esc}\s*\(")
                dotted_ref = re.compile(rf"(?:\.|::)\s*{esc}\b")
                importish = re.compile(
                    r"^\s*(?:from\s+\S+\s+)?(?:import|use|pub\s+use)\b")
                for ev in t.get("rg_evidence", []):
                    evf, evl = ev["at"].rsplit(":", 1)
                    evl = int(evl)
                    lines = cached_lines(repo_dir, evf)
                    line_txt = lines[evl - 1] if evl <= len(lines) else ""
                    if is_method and evf != tfile:
                        ok = (dotted_call.search(line_txt) if et == "CALLS"
                              else (dotted_ref.search(line_txt)
                                    or importish.match(line_txt)))
                        if not ok:
                            hit(name, et, t, "method_bare_cross_file",
                                f"cross-file evidence at {ev['at']} is not "
                                f"dotted/qualified: {line_txt.strip()[:100]}")
                    skip = {t["line"]} if evf == tfile else ()
                    sh = _shadow_hit(lines, evl, lang, sym, skip,
                                     src_file=evf, target_file=tfile)
                    if sh is not None:
                        hit(name, et, t, "shadowed_or_multidef",
                            f"binding named '{sym}' at {evf}:{sh[0]} "
                            f"shadows evidence at {ev['at']}: {sh[1][:100]}")
                # scan 7 (fix (l)): cross-version attribution — no caller
                # may live in a different version subtree than the target.
                bad_ver = _version_violations(name, tfile, caller_files)
                if bad_ver:
                    hit(name, et, t, "cross_version_attribution",
                        f"caller files in a different version subtree than "
                        f"{tfile}: {bad_ver[:3]}")
                # scan 8 (fix (m)): Java @Override of an external super-
                # method — every evidence receiver must statically be the
                # owner class or a repo type related to it.
                if lang == "java" and is_method and \
                        _java_is_override(cached_lines(repo_dir, tfile),
                                          t["line"]):
                    graph = _java_type_graph(repo_dir)
                    own = (t.get("filters", {}).get("owner_class")
                           or Path(tfile).stem)
                    for ev in t.get("rg_evidence", []):
                        evf, evl = ev["at"].rsplit(":", 1)
                        bad = _java_override_bad_receiver(
                            cached_lines(repo_dir, evf), int(evl), sym,
                            tfile, evf, own, graph)
                        if bad:
                            hit(name, et, t, "external_override_receiver",
                                f"evidence at {ev['at']}: {bad}")
    if hits:
        print(f"\n[audit] FAIL — {len(hits)} adversarial hit(s):")
        for repo, et, sym, loc, check, detail in hits:
            print(f"[audit]   {repo} {et} {sym} @ {loc} :: {check}: {detail}")
    else:
        n = sum(len(dd.get("targets", []))
                for name in repos
                for dd in data["repos"][name]["edge_types"].values())
        print(f"\n[audit] PASS — 0 hits over {n} surviving targets "
              f"({', '.join(repos)})")
    return len(hits)


# --------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["setup", "oracle", "probe", "audit",
                                    "all"])
    ap.add_argument("--repos", nargs="*", default=list(REPOS),
                    help="subset of repos (default: all four)")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd in ("setup", "all"):
        cmd_setup(a.repos)
    if a.cmd in ("oracle", "all"):
        cmd_oracle(a.repos)
    if a.cmd in ("probe", "all"):
        cmd_probe([r for r in a.repos if r in PROBE_REPOS])
    if a.cmd in ("audit", "all"):
        n_hits = cmd_audit(a.repos)
        if n_hits:
            print(f"\n[done] {a.cmd} in {time.time() - t0:.1f}s — "
                  f"AUDIT GATE FAILED")
            sys.exit(1)
    print(f"\n[done] {a.cmd} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
