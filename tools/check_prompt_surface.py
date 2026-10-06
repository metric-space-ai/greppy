#!/usr/bin/env python3
"""Fail when an advertised command or its flags are absent from binary help.

Checks command rows, inline code examples and flags in their continuation rows.
No indexing, inference, browser runtime or external service is started.
Unlike a prose scan, it does not mistake 'greppy holds' for a command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def mentions(text):
    rows = []
    previous = None
    for number, line in enumerate(text.splitlines(), 1):
        if re.match(r"^[A-Z][A-Z -]+:", line):
            previous = None
        fragments = []
        if re.match(r"^\s{2,}greppy\s", line):
            fragments.append(re.split(r"\s{2,}", line.strip(), maxsplit=1)[0])
        fragments.extend(match.group(1) for match in re.finditer(r"`(greppy\s[^`]+)`", line))
        for fragment in fragments:
            words = fragment.split()
            command = []
            # 'web session new' needs help at the nested command, not 'web'.
            for word in words[1:]:
                if not re.fullmatch(r"[a-z][a-z-]*|-p", word):
                    break
                command.append(word)
                if command[0] != "web":
                    break
                if len(command) == 2 and command[1] not in (
                        "session", "tab", "runtime", "trace", "endpoint", "script", "artifact", "result"):
                    break
                if len(command) >= 3:
                    break
            if not command:
                continue  # Literal grep PATTERN / -n compatibility examples.
            flags = re.findall(r"(?<![\w-])(?:--[a-z][a-z-]*|-[a-z]\b)", fragment)
            row = {"line": number, "command": command, "flags": flags}
            rows.append(row)
            previous = row
        if not fragments and previous is not None and line.startswith(" " * 20):
            previous["flags"].extend(re.findall(r"(?<![\w-])(?:--[a-z][a-z-]*|-[a-z]\b)", line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = Path(args.prompt).read_bytes()
    rows = mentions(data.decode())
    assert rows, "no advertised commands found"
    cache = {}
    failures = []

    def help_for(command):
        key = tuple(command)
        if key not in cache:
            result = subprocess.run([args.binary, *command, "--help"], capture_output=True,
                                    text=True, timeout=10)
            cache[key] = (result.returncode, result.stdout + result.stderr)
        return cache[key]

    for row in rows:
        code, text = help_for(row["command"])
        if code:
            failures.append({**row, "error": f"help returned {code}", "output": text})
            continue
        available = set(re.findall(r"(?<![\w-])(?:--[a-z][a-z-]*|-[a-z]\b)", text))
        for flag in row["flags"]:
            if flag not in available:
                failures.append({**row, "error": f"{flag} absent from this command's help"})
    report = {"binary": str(Path(args.binary).resolve()),
              "prompt_sha256": hashlib.sha256(data).hexdigest(),
              "checked_mentions": len(rows), "commands": rows,
              "help": [{"command": list(key), "exit": value[0], "text": value[1]}
                       for key, value in cache.items()],
              "failures": failures, "passed": not failures}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checked_mentions": len(rows), "failures": failures, "passed": not failures}))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
