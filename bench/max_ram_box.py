#!/usr/bin/env python3
"""Hard memory matrix for greppy on a headless Ubuntu box.

One JSON object per task on stdout. Prefers `systemd-run --scope` with
MemoryMax. If PID 1 is not systemd, creates a cgroup v2 under
/sys/fs/cgroup/greppy-max-ram (needs root, or passwordless sudo).

Agent runs talk to greppy's plain-HTTP gateway. They do not call
https://api.minimax.io. See docs/max-ram.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def parse_bytes(text: str) -> int:
    raw = text.strip().lower()
    if raw.isdigit():
        return int(raw)
    number = ""
    suffix = ""
    for char in raw:
        if char.isdigit():
            number += char
        else:
            suffix += char
    return int(number) * {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[suffix]


def role_of(cmdline: str) -> str:
    if "summarize-daemon" in cmdline:
        return "summarize-daemon"
    if "embed-daemon" in cmdline:
        return "embed-daemon"
    if "--internal-role content" in cmdline:
        return "browser-content"
    if "--internal-role controller" in cmdline:
        return "browser-controller"
    if "web-runtime" in cmdline and cmdline.split(" ", 1)[0].endswith("web-runtime"):
        return "web-runtime"
    if "greppy" in cmdline:
        return "greppy"
    return "other"


def read_status(pid: int) -> tuple[int, int, str, str]:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="replace")
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        cgroup = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii", errors="replace").strip()
    except OSError:
        return 0, 0, "", ""
    rss = hwm = 0
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            hwm = int(line.split()[1]) * 1024
    return rss, hwm, cmdline, cgroup


def cgroup_int(path: Path) -> int:
    try:
        raw = path.read_text().strip()
    except OSError:
        return 0
    if raw in {"", "max"}:
        return 0
    return int(raw)


def events(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in path.read_text().splitlines():
            key, value = line.split()
            out[key] = int(value)
    except OSError:
        pass
    return out


def systemd_available() -> bool:
    if not Path("/run/systemd/system").exists():
        return False
    probe = subprocess.run(
        ["systemd-run", "--scope", "-q", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


class Limiter:
    def __init__(self, memory: int, swap: int) -> None:
        self.memory = memory
        self.swap = swap
        self.kind = "systemd-run" if systemd_available() else "cgroup"
        self.cg = Path("/sys/fs/cgroup/greppy-max-ram")

    def prepare_cgroup(self) -> None:
        subprocess.run(["sudo", "mkdir", "-p", str(self.cg)], check=True)
        # A fresh directory is the only reliable peak reset on this 6.12 kernel.
        procs = []
        try:
            procs = [p for p in (self.cg / "cgroup.procs").read_text().split() if p.strip()]
        except OSError:
            procs = []
        for pid in procs:
            subprocess.run(["sudo", "kill", "-KILL", pid], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.05)
        subprocess.run(["sudo", "rmdir", str(self.cg)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["sudo", "mkdir", "-p", str(self.cg)], check=True)
        subprocess.run(["sudo", "tee", str(self.cg / "memory.max")], input=f"{self.memory}\n", text=True, stdout=subprocess.DEVNULL, check=True)
        swap_file = self.cg / "memory.swap.max"
        if swap_file.exists():
            subprocess.run(["sudo", "tee", str(swap_file)], input=f"{self.swap}\n", text=True, stdout=subprocess.DEVNULL, check=False)

    def start(self, cmd: list[str], cwd: str, env: dict[str, str], log: Path) -> subprocess.Popen[bytes]:
        if self.kind == "systemd-run":
            argv = [
                "systemd-run",
                "--scope",
                "-q",
                "-p",
                f"MemoryMax={self.memory}",
                "-p",
                f"MemorySwapMax={self.swap}",
                "--uid",
                str(os.getuid()),
                "--working-directory",
                cwd,
                "--",
                *cmd,
            ]
            return subprocess.Popen(argv, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.prepare_cgroup()
        env_path = Path(f"/tmp/greppy-max-ram-env-{os.getpid()}")
        lines = []
        for key, value in env.items():
            allowed = key in {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR"} or key.startswith("GREPPY_") or key == "MINIMAX_API_KEY"
            if not allowed or "\n" in value or not key.replace("_", "").isalnum():
                continue
            lines.append(f"{key}={shlex.quote(value)}")
        env_path.write_text("\n".join(lines) + "\n")
        wrapper = f"""
set +e
echo $$ > {self.cg}/cgroup.procs
cd {shlex.quote(cwd)}
set -a
. {env_path}
set +a
exec runuser --preserve-environment -u {os.environ.get("USER", "ubuntu")} -- "$@"
"""
        return subprocess.Popen(
            ["sudo", "bash", "-c", wrapper, "bash", *cmd],
            stdout=log.open("w"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )


def sample_tree(root_pid: int, peaks: dict[str, dict[str, int | str]]) -> None:
    seen: set[int] = set()
    stack = [root_pid]
    # Also pick up daemons whose executable is greppy, in case they left the group.
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = entry.joinpath("cmdline").read_bytes()
        except OSError:
            continue
        if raw.startswith(b"/") and (b"embed-daemon" in raw or b"summarize-daemon" in raw):
            first = raw.split(b"\x00", 1)[0]
            if first.endswith(b"/greppy"):
                stack.append(int(entry.name))
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        rss, hwm, cmdline, cgroup = read_status(pid)
        if cmdline:
            base = os.path.basename(cmdline.split(" ", 1)[0])
            if base not in {"sudo", "bash", "sh", "runuser", "systemd-run", "systemd"}:
                kind = role_of(cmdline)
                slot = peaks.setdefault(kind, {"rss_bytes": 0, "hwm_bytes": 0, "cmdline": cmdline, "cgroup": cgroup})
                slot["rss_bytes"] = max(int(slot["rss_bytes"]), rss)
                slot["hwm_bytes"] = max(int(slot["hwm_bytes"]), hwm)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            for child in children.read_text().split():
                if child.strip():
                    stack.append(int(child))
        except OSError:
            pass


def run_limited(limiter: Limiter, label: str, cmd: list[str], cwd: str, env: dict[str, str], timeout: float) -> dict:
    log = Path(f"/tmp/greppy-max-ram-{label.replace('/', '_')}.log")
    started = time.perf_counter()
    proc = limiter.start(cmd, cwd, env, log)
    peaks: dict[str, dict[str, int | str]] = {}
    cg_peak = 0
    cg_path: Path | None = None
    before_oom = 0
    while proc.poll() is None:
        if time.perf_counter() - started > timeout:
            subprocess.run(["sudo", "kill", "-KILL", f"-{proc.pid}"], check=False)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        sample_tree(proc.pid, peaks)
        if cg_path is None:
            for slot in peaks.values():
                text = str(slot.get("cgroup") or "")
                for part in text.split(":"):
                    if part.startswith("/") and part not in {"/", ""}:
                        candidate = Path("/sys/fs/cgroup") / part.lstrip("/")
                        if (candidate / "memory.current").exists():
                            cg_path = candidate
                            before_oom = events(candidate / "memory.events").get("oom_kill", 0)
        if cg_path is not None:
            cg_peak = max(cg_peak, cgroup_int(cg_path / "memory.current"), cgroup_int(cg_path / "memory.peak"))
        elif limiter.kind == "cgroup":
            cg_peak = max(cg_peak, cgroup_int(limiter.cg / "memory.current"), cgroup_int(limiter.cg / "memory.peak"))
            cg_path = limiter.cg
            before_oom = events(limiter.cg / "memory.events").get("oom_kill", 0)
        time.sleep(0.05)
    wall = time.perf_counter() - started
    if limiter.kind == "cgroup":
        cg_path = limiter.cg
    after = events((cg_path or limiter.cg) / "memory.events") if cg_path or limiter.kind == "cgroup" else {}
    kernel_peak = cgroup_int((cg_path or limiter.cg) / "memory.peak") if (cg_path or limiter.kind == "cgroup") else 0
    tail = ""
    try:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-20:])
    except OSError:
        pass
    code = proc.returncode
    return {
        "label": label,
        "limiter": limiter.kind,
        "memory_max": limiter.memory,
        "swap_max": limiter.swap,
        "argv": cmd,
        "exit_code": code,
        "killed": code in (137, -9) or (code is not None and code < 0 and -code == signal.SIGKILL),
        "oom_kill_delta": after.get("oom_kill", 0) - before_oom,
        "wall_s": round(wall, 3),
        "cgroup_peak_bytes": max(cg_peak, kernel_peak),
        "processes": peaks,
        "log_tail": tail,
    }


def greppy(binary: str, memory: str, repo: str, *args: str) -> list[str]:
    return [binary, "--max-ram", memory, "--device", "cpu", "--root", repo, *args]


def tasks(binary: str, memory: str, repo: str, agent: bool) -> list[tuple[str, list[str], float]]:
    rows: list[tuple[str, list[str], float]] = [
        ("index", greppy(binary, memory, repo, "index", repo), 900),
        ("search-symbol", greppy(binary, memory, repo, "search-symbol", "RamBudget", "--limit", "5"), 180),
        (
            "search-pattern",
            greppy(binary, memory, repo, "search-pattern", "--path", "crates/web-client/src/max_ram.rs", "OS_RESERVE_BYTES"),
            180,
        ),
        ("search", greppy(binary, memory, repo, "search", "reserve host memory for the browser"), 180),
        ("brief", greppy(binary, memory, repo, "brief", "crates/web-client/src/max_ram.rs::RamBudget"), 180),
        (
            "edit",
            greppy(
                binary,
                memory,
                repo,
                "replace-text",
                "crates/web-client/src/max_ram.rs",
                "Host RAM budget for headless machines.",
                "Host RAM budget for headless machines.",
            ),
            60,
        ),
        ("web-doctor", [binary, "--max-ram", memory, "--device", "cpu", "web", "doctor"], 60),
    ]
    if agent:
        rows.append(
            (
                "agent",
                [
                    binary,
                    "--max-ram",
                    memory,
                    "--device",
                    "cpu",
                    "-p",
                    "In crates/web-client/src/max_ram.rs, do not edit anything. Reply with the name of the RamBudget struct and stop.",
                    "--model",
                    os.environ.get("GREPPY_MODEL", "MiniMax-M3.1-Flash"),
                    "--endpoint",
                    os.environ.get("GREPPY_ENDPOINT", "http://127.0.0.1:8317"),
                    "--context-tokens",
                    os.environ.get("GREPPY_CONTEXT_TOKENS", "1000000"),
                    "--max-turns",
                    "3",
                    "--mode",
                    "performance",
                    "--in-place",
                    "--json",
                ],
                300,
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run greppy under 1/2/3/4 GiB memory caps.")
    parser.add_argument("--bin", default="greppy")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--sizes", default="1g,2g,3g,4g")
    parser.add_argument("--swap", default="0")
    parser.add_argument("--out", default="")
    parser.add_argument("--skip-agent", action="store_true")
    args = parser.parse_args()
    repo = str(Path(args.repo).resolve())
    env = os.environ.copy()
    if not env.get("GREPPY_API_KEY") and env.get("MINIMAX_API_KEY"):
        env["GREPPY_API_KEY"] = env["MINIMAX_API_KEY"]
    env.setdefault("GREPPY_MODEL", "MiniMax-M3.1-Flash")
    env.setdefault("GREPPY_ENDPOINT", "http://127.0.0.1:8317")
    agent = not args.skip_agent
    if agent:
        from urllib.parse import urlparse

        parsed = urlparse(env["GREPPY_ENDPOINT"])
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme == "https":
            agent = False
            print(json.dumps({
                "label": "agent-skipped",
                "reason": "GREPPY_ENDPOINT is https. greppy has no TLS. Point it at a local HTTP gateway that calls MiniMax.",
            }))
        else:
            try:
                import socket
                sock = socket.create_connection((host, port), 2)
                sock.close()
            except OSError as exc:
                agent = False
                print(json.dumps({
                    "label": "agent-skipped",
                    "reason": f"no gateway at {env['GREPPY_ENDPOINT']}: {exc}",
                    "model": env["GREPPY_MODEL"],
                    "minimax_key_present": bool(env.get("MINIMAX_API_KEY")),
                }))
    handle = open(args.out, "a") if args.out else None
    for memory in [part.strip() for part in args.sizes.split(",") if part.strip()]:
        limiter = Limiter(parse_bytes(memory), parse_bytes(args.swap))
        store = Path(f"/tmp/greppy-max-ram-store-{memory}-{args.swap}")
        store.mkdir(parents=True, exist_ok=True)
        env["GREPPY_STORE_DIR"] = str(store)
        for name, cmd, timeout in tasks(args.bin, memory, repo, agent):
            if name == "edit":
                subprocess.run(["git", "-C", repo, "checkout", "--", "crates/web-client/src/max_ram.rs"], check=False)
            record = run_limited(limiter, f"{memory}-{name}", cmd, repo, env, timeout)
            line = json.dumps(record)
            print(line, flush=True)
            if handle:
                handle.write(line + "\n")
                handle.flush()
    if handle:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
