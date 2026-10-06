#!/usr/bin/env python3
"""Capture the real -p and PTY requests without contacting a model provider.

Run through the shared heavy-job gate: normal startup/self-check is retained.
The supplied expected files are exact effective system prompts, not substrings.
This verifies transport and tool schemas, not model adoption or task quality.
All disposable data and output must be on the designated development volume.
"""
import argparse
import fcntl
import hashlib
import http.server
import json
import os
from pathlib import Path
import pty
import select
import signal
import struct
import subprocess
import termios
import threading
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def system_text(body):
    system = body.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        assert all(part.get("type") == "text" for part in system), system
        return "".join(part["text"] for part in system)
    raise AssertionError("request has no text system prompt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-one-shot", required=True)
    parser.add_argument("--expected-interactive", required=True)
    parser.add_argument("--require-distinct", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    assert str(output).startswith(("/Volumes/tmp/", "/mnt/nvme1/")), output
    output.mkdir(parents=True, exist_ok=False)
    binary = Path(args.binary).resolve(strict=True)
    report = {"binary": str(binary), "binary_sha256": digest(binary.read_bytes()),
              "pid": os.getpid(), "terminal": False, "passed": False,
              "scope": "real startup and outbound requests to a local stub; no model quality claim",
              "runs": []}
    requests = []
    active_mode = None
    received = threading.Event()

    def save():
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")

    class Gateway(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.endswith("/models"):
                self.reply(200, b'{"data":[{"id":"prompt-capture"}]}')
            else:
                self.reply(404, b'{}')

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4 * 1024 * 1024:
                self.reply(400, b'{}')
                return
            body = json.loads(self.rfile.read(length))
            # Never save authorization headers, even if the caller has a key.
            requests.append({"mode": active_mode, "path": self.path, "body": body})
            (output / f"request-{active_mode}.json").write_text(json.dumps(requests[-1], indent=2) + "\n")
            received.set()
            self.reply(500, b'{"error":{"message":"capture complete; no provider contacted"}}')

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    server.daemon_threads = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    inherited = tuple(fd for fd in range(3, 256) if inheritable(fd))
    save()
    try:
        for mode, expected_path in (("one-shot", args.expected_one_shot),
                                    ("interactive", args.expected_interactive)):
            active_mode = mode
            received.clear()
            root = output / mode
            repo = root / "repo"
            repo.mkdir(parents=True)
            (repo / "lib.rs").write_text("pub fn capture_target() -> i32 { 7 }\n")
            for command in (("init", "-q"), ("add", "lib.rs"),
                            ("-c", "user.name=Prompt capture", "-c", "user.email=capture@localhost",
                             "commit", "-qm", "fixture")):
                subprocess.run(["git", *command], cwd=repo, check=True, capture_output=True)
            env = dict(os.environ, GREPPY_STORE_DIR=str(root / "store"),
                       GREPPY_CONFIG_DIR=str(root / "config"),
                       GREPPY_WORKSPACE_DIR=str(root / "workspaces"),
                       TERM="xterm", GREPPY_ASCII="1")
            for key in list(env):
                if key.startswith("GREPPY_TEST_") or key in ("CI", "GREPPY_MODEL", "GREPPY_API_KEY",
                        "GREPPY_ENDPOINT", "GREPPY_SKIP_SELFCHECK", "GREPPY_NO_SANDBOX"):
                    env.pop(key, None)
            command = [str(binary), "agent" if mode == "interactive" else "-p",
                       "Say capture-ready. Do not invoke tools or change files.",
                       "--model", "prompt-capture", "--endpoint", endpoint,
                       "--max-turns", "1", "--deadline-secs", "90"]
            master = slave = None
            log = (root / "terminal.log").open("wb")
            child = None
            try:
                if mode == "interactive":
                    master, slave = pty.openpty()
                    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 110, 0, 0))
                    child = subprocess.Popen(command, cwd=repo, env=env, stdin=slave, stdout=slave,
                                             stderr=slave, pass_fds=inherited, start_new_session=True)
                    os.close(slave)
                    slave = None
                else:
                    child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=log, pass_fds=inherited,
                                             start_new_session=True)
                report["runs"].append({"mode": mode, "pid": child.pid, "command": command,
                                       "output": str(root), "stop": "capture or 90 seconds"})
                save()
                until = time.monotonic() + 90
                tail = b""
                while not received.is_set() and child.poll() is None and time.monotonic() < until:
                    if master is not None and select.select([master], [], [], .1)[0]:
                        try:
                            data = os.read(master, 65536)
                        except OSError:
                            break
                        log.write(data)
                        scan = tail + data
                        for query, answer in ((b"\x1b[6n", b"\x1b[1;1R"),
                                              (b"\x1b[c", b"\x1b[?1;2c"),
                                              (b"\x1b[>c", b"\x1b[>0;0;0c")):
                            if query in scan:
                                os.write(master, answer)
                        tail = scan[-3:]
                    else:
                        received.wait(.1)
                assert received.is_set(), f"{mode}: no request captured; see {root / 'terminal.log'}"
                request = next(item for item in requests if item["mode"] == mode)
                text = system_text(request["body"])
                expected = Path(expected_path).read_bytes()
                actual = text.encode()
                (root / "effective-prompt.md").write_bytes(actual)
                report["runs"][-1].update(prompt_sha256=digest(actual), prompt_bytes=len(actual),
                                           expected_sha256=digest(expected))
                assert actual == expected, f"{mode}: outbound prompt differs from expected exact bytes"
                tools = request["body"].get("tools", [])
                assert len(tools) == 1 and tools[0]["name"] == "greppy", tools
                schema = tools[0]["input_schema"]
                assert schema["properties"]["args"]["type"] == "array", schema
                assert schema["properties"]["args"]["items"]["type"] == "string", schema
                report["runs"][-1]["tool_schema"] = schema
            finally:
                if child is not None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=10)
                    report["runs"][-1]["child_reaped"] = True
                for fd in (master, slave):
                    if fd is not None:
                        os.close(fd)
                log.close()
                save()
        if args.require_distinct:
            assert report["runs"][0]["prompt_sha256"] != report["runs"][1]["prompt_sha256"], "TUI still has one-shot identity"
        assert report["runs"][0]["tool_schema"] == report["runs"][1]["tool_schema"]
        report["passed"] = True
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        report["terminal"] = True
        save()
    print(json.dumps({"passed": True, "receipt": str(output / "receipt.json")}))


def inheritable(fd):
    try:
        return os.get_inheritable(fd)
    except OSError:
        return False


if __name__ == "__main__":
    main()
