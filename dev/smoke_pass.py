"""The bench gate: every command AGENTS.md advertises, run against the binary.

For each advertised verb: one concrete invocation in the fixture repo, an
expected exit code, required output markers, and forbidden legacy markers
(old row shapes, ASCII bundles, German, removed verbs). A verb the binary
does not hold yet fails loudly — that IS the gap the release still has.

    python3 dev/smoke_pass.py <path-to-greppy-binary>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = "/Volumes/tmp/outputs-repo"
STORE = "/Volumes/tmp/wc-store2"

FORBIDDEN_EVERYWHERE = [
    "::Function::",          # pre-0.3.0 qualified rows
    "-- CALLERS",            # pre-0.3.0 brief bundle
    # (the pre-0.3.0 bundle header `== name ==` is checked as a line-anchored
    # regex below — a bare "== " substring is legitimate source code)
    "weitere",               # German
    "Zeilen",
    "suggestion:",           # banned advisory lines
    "try:",
]

# verb -> (argv-tail, expected-exits, required-regexes)
CASES = {
    "where-am-i": (["where-am-i"], {0}, [r"\d+ files", r"greppy expand [0-9a-f]+"]),
    "who-calls": (["who-calls", "parse_path"], {0}, [r"^\S+:\d+  \S+", ]),
    "who-calls-multi": (["who-calls", "parse_path", "data_set"], {0}, [r"^\S+:\d+  \S+"]),
    "who-calls-missing": (["who-calls", "xyzzy_frobnicate"], {1}, [r"no symbol"]),
    "callees": (["callees", "data_set"], {0}, [r"^\S+:\d+  \S+"]),
    "brief": (["brief", "parse_path"], {0}, [r"^\S+:\d+$|^\S+\.rs:\d+"]),
    # impact prints the callers AS the tree (the start symbol is the question,
    # not a row): require tree rows with hint sentences.
    "impact": (["impact", "parse_path"], {0}, [r"^\S+:\d+  \S+ — "]),
    "path": (["path", "--from", "data_set", "--to", "parse_path"], {0}, [r"data_set"]),
    "search": (["search", "restrict a value to a range"], {0, 1}, []),
    "search-symbol": (["search-symbol", "parse_path"], {0}, [r"parse_path"]),
    "search-pattern": (["search-pattern", "fn parse_path"], {0}, [r"parse_path"]),
    "read": (["read", "parse_path"], {0}, [r"fn parse_path"]),
    "read-smart": (["read-smart", "data_set"], {0}, []),
    "read-file": (["read-file", "edit-src/data.rs"], {0}, []),
    # refusals carry their own exit codes and true sentences:
    # 13 = OLD occurs 0/N times, 20 = empty stdin, 10 = nothing to undo
    "replace-text": (["replace-text", "edit-src/data.rs", "no-such-text-xyzzy", "NEW", "--dry-run"], {13}, [r"occurs 0 times"]),
    "patch": (["patch", "--dry-run"], {20}, [r"no DIFF"]),
    "undo": (["undo", "no-such-id"], {10}, [r"nothing to undo"]),
    "expand-bad-id": (["expand", "ffffffffffffffff"], {1}, [r"not found|expired"]),
    "grep-passthrough": (["-c", "parse_path", "edit-src/data.rs"], {0}, [r"^\d+$"]),
    # ON EVERY COMMAND promises --path, --json, --limit on every verb: sample
    # them across the families so a missing wire cannot hide again.
    "footer-path-search": (["search-symbol", "parse_path", "--path", "edit-src"], {0}, [r"parse_path"]),
    "footer-path-nav": (["who-calls", "parse_path", "--path", "edit-src"], {0}, [r"^\S+:\d+  \S+"]),
    "footer-json-search": (["search-pattern", "fn parse_path", "--json"], {0}, [r'"command": "search-pattern"']),
    "footer-limit-nav": (["who-calls", "parse_path", "--limit", "1"], {0}, [r"^\S+:\d+  \S+"]),
    "footer-path-read": (["read", "parse_path", "--path", "edit-src"], {0}, []),
}

REMOVED = ["find-usages", "references", "map", "outline", "changes", "verify",
           "search-symbols", "search-code"]  # orient was a section, never a verb

# Compiled out of 0.3.0 (feature `bash-smart`): not retired vocabulary, so it
# is not refused — it is simply not a greppy verb and falls through to grep.
NOT_IN_THIS_RELEASE = ["bash-smart"]


def run(binary: str, tail: list[str]) -> tuple[int, str]:
    env = dict(os.environ, GREPPY_STORE_DIR=STORE)
    proc = subprocess.run([binary, *tail], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300,
                          stdin=subprocess.DEVNULL)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> None:
    binary = os.path.abspath(sys.argv[1])
    passed = failed = 0
    for name, (tail, exits, required) in CASES.items():
        code, out = run(binary, tail)
        errors = []
        if code not in exits:
            errors.append(f"exit {code}, wanted {sorted(exits)}")
        json_case = "--json" in tail
        for marker in FORBIDDEN_EVERYWHERE:
            # a --json answer legitimately carries qualified names as data;
            # the format markers bind text output, the language ban binds all
            if json_case and marker in ("::Function::", "== "):
                continue
            if marker in out:
                errors.append(f"forbidden marker {marker!r}")
        if not json_case and re.search(r"^== \S.* ==$", out, re.M):
            errors.append("forbidden bundle header '== name =='")
        for pattern in required:
            if not re.search(pattern, out, re.M):
                errors.append(f"missing /{pattern}/")
        if errors:
            failed += 1
            print(f"FAIL {name}: " + "; ".join(errors))
            for line in out.splitlines()[:3]:
                print(f"     | {line[:110]}")
        else:
            passed += 1
            print(f"ok   {name}")
    for verb in REMOVED:
        code, out = run(binary, [verb, "parse_path"])
        # dead greppy vocabulary is REFUSED before grep passthrough
        # (unknown_verb_refusal): exit 64, the refusal names the verb, and it
        # neither answers with symbol rows nor greps the verb as a pattern.
        if code == 64 and "unrecognized" in out and not re.search(r"^\S+:\d+  \S+", out, re.M):
            passed += 1
            print(f"ok   removed:{verb} (refused)")
        else:
            failed += 1
            print(f"FAIL removed:{verb}: exit {code}, wanted the vocabulary refusal")
    for verb in NOT_IN_THIS_RELEASE:
        code, out = run(binary, [verb, "--", "sh", "-c", "echo ok"])
        if code != 64 and "greppy expand" not in out and "…" not in out:
            passed += 1
            print(f"ok   absent:{verb} (not a verb in this release)")
        else:
            failed += 1
            print(f"FAIL absent:{verb}: the binary still answers it (exit {code})")
    print(f"\nSMOKE: {passed} ok, {failed} fail")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
