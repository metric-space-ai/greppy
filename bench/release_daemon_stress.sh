#!/usr/bin/env bash
# End-to-end daemon process/stress acceptance for an unpacked Unix release
# artifact.
#
# Ports the in-tree inference-daemon suite (crates/cli/src/inference_daemon.rs
# tests, crates/cli/src/embed_daemon.rs, crates/cli/src/summarize_daemon.rs)
# to the PACKAGED binary: every scenario drives real CLI processes and the
# daemons' public local endpoint from OUTSIDE, and asserts only observable
# behavior — exit codes, JSON responses, process lifecycle (pgrep), and
# socket files. No test-harness internals, no injected fakes.
#
# Scenario map (in-tree test -> section below):
# * live_server_serializes_clients_evicts_and_reloads_one_model
#     -> "concurrent clients" (one owner pid across a 32+ connection burst)
#        + "idle eviction" (state ready -> evicted -> reload on demand)
# * thirty_two_spawn_contenders_have_one_owner
#     -> "concurrent clients" (burst against a cold endpoint spawns one owner)
# * saturated_queue_rejects_work_and_expires_queued_deadlines
#     -> "queue limits" (flood while the model is busy; classified capacity
#        responses: error_kind=capacity, retryable=true)
# * slow_client_does_not_block_inference_or_prematurely_end_server
#     -> "slow client" (stalled partial frame; concurrent requests unaffected;
#        stalled connection is rejected after the 5s connection read timeout)
# * frame_reader_rejects_oversize_and_slow_clients
#     -> "oversize request" + "slow client"
# * killed_daemon_is_replaced_and_stale_endpoint_is_repaired
#     -> "kill -9 mid-load" (fresh owner pid, correct answers afterwards)
# * daemon_owner_repairs_stale_endpoint (+ client-side ENOTSOCK
#   classification, stale_non_socket_endpoint_classifies_as_no_daemon)
#     -> "stale endpoint recovery" (regular file at the socket path)
# * protocol/versioning validation (client_request_limit, validate tests)
#     -> "protocol sanity" (ping, status, protocol mismatch, malformed frame)
#
# Deliberately NOT ported (documented deviations):
# * The 60s request deadline and the 75s hung-worker hard timeout have no
#   external configuration surface (crates/cli/src/embed_daemon.rs and
#   summarize_daemon.rs hardcode CLIENT_READ_TIMEOUT/HARD_REQUEST_TIMEOUT),
#   so observing them would cost >60s per assertion against a fixed <5 min
#   budget. The externally observable request timeout that IS covered here is
#   the 5s connection read timeout (slow-client section). The long deadlines
#   stay covered in-tree with injected policies.
# * The in-tree 32-client test uses a 15ms fake model. Real CPU inference
#   serializes at seconds per request, so 32 concurrent REAL inferences would
#   exceed the fixed 60s client deadline by arithmetic on any host. This port
#   keeps 32+ concurrent connections against one owner but bounds the real
#   inference depth (5 CLI clients) and drives the remaining connections
#   through the full accept -> reader -> queue -> respond pipeline.
# * Heavy concurrency runs against the embedding daemon only: one Qwen
#   summary takes tens of seconds on CPU, and both daemons share the same
#   serve loop (crates/cli/src/inference_daemon.rs). The summarize daemon is
#   covered with a real `brief`, a queue-limit flood during its busy window,
#   oversize rejection, and model-TTL eviction + idle exit.
#
# The script is copied verbatim into the Unix release tarball (see
# .github/workflows/release.yml) and must stay self-contained: fixtures are
# generated inline, and only tooling that exists on the ubuntu and macos
# runners is used (bash 3.2+, jq, pgrep, python3 — python3 is already a
# dependency of the same verify-packages job via release_artifacts.py).

set -euo pipefail

BIN="${1:?usage: release_daemon_stress.sh /path/to/greppy [work-dir]}"
case "$BIN" in /*) ;; *) BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")" ;; esac
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 64; }
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/greppy-daemon-stress-XXXXXX")}"
mkdir -p "$WORK/store" "$WORK/out"

section() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

command -v jq >/dev/null 2>&1 || fail "jq is required"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v pgrep >/dev/null 2>&1 || fail "pgrep is required"

# Contain daemon sockets (crates/cli/src/inference_daemon.rs
# unix_runtime_dir(): $XDG_RUNTIME_DIR/greppy when the joined path fits a
# unix-socket-safe 32 chars) in a directory this script owns, so process and
# socket assertions cannot see daemons from other sessions.
RUNTIME_BASE="/tmp/gds-$$"
mkdir -p -m 700 "$RUNTIME_BASE"
export XDG_RUNTIME_DIR="$RUNTIME_BASE"

cleanup() {
  pkill -f -- "-daemon --socket $RUNTIME_BASE/" 2>/dev/null || true
}
trap cleanup EXIT

export GREPPY_STORE_DIR="$WORK/store"
# One device identity for every code path (spawn, prewarm, doctor, queries):
# the endpoint hash includes the device (Endpoint::for_identity), so a mixed
# cpu/auto session would talk to two different daemons.
export GREPPY_DEVICE=cpu
# Long embedding TTLs while the stress sections run (no surprise exits
# between sections); the dedicated eviction section respawns with short TTLs.
export GREPPY_EMBED_DAEMON_MODEL_TTL_S=600
export GREPPY_EMBED_DAEMON_EXIT_TTL_S=600
# Short summarize TTLs from the start: the brief section is the only Qwen
# workload, and its daemon then demonstrates model eviction + idle exit
# without a second (expensive) model load.
export GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S=5
export GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S=15

# Raw endpoint client for the daemons' public newline-framed JSON transport.
# Everything it sends/reads is the same observable surface any local client
# of the packaged binary uses.
CLIENT="$WORK/daemon_client.py"
cat >"$CLIENT" <<'PY'
import json, socket, sys, threading, time

def connect(path, timeout):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    return s

def read_line(s):
    buf = b""
    while b"\n" not in buf:
        chunk = s.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0].decode("utf-8", "replace")

def round_trip(path, payload, timeout=10.0, raw=None):
    s = connect(path, timeout)
    try:
        s.sendall(raw if raw is not None else (json.dumps(payload).encode() + b"\n"))
        return read_line(s)
    finally:
        s.close()

def status(path, timeout=5.0):
    return json.loads(round_trip(path, {"protocol": 2, "op": "status"}, timeout))

def main():
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "req":
        print(round_trip(path, json.loads(sys.argv[3])))
    elif mode == "status":
        print(json.dumps(status(path)))
    elif mode == "oversize":
        n = int(sys.argv[3])
        started = time.time()
        line = round_trip(path, None, timeout=30.0, raw=b"x" * n)
        print(json.dumps({"response": json.loads(line), "elapsed_s": time.time() - started}))
    elif mode == "slow":
        # Stalled client: one byte of a frame, then silence. The daemon must
        # reject the connection after its read timeout instead of hanging.
        s = connect(path, 30.0)
        try:
            s.sendall(b"{")
            started = time.time()
            line = read_line(s)
            print(json.dumps({"response": json.loads(line), "elapsed_s": time.time() - started}))
        finally:
            s.close()
    elif mode == "wait-active":
        # Wait until the daemon reports an in-flight inference job (job ids
        # are set while the model loads AND while it infers), i.e. the main
        # loop is provably busy.
        deadline = time.time() + float(sys.argv[3])
        while time.time() < deadline:
            try:
                if status(path, 2.0).get("active_request_id"):
                    print("active")
                    return
            except OSError:
                pass
            time.sleep(0.02)
        sys.exit(2)
    elif mode == "wait-state":
        want, deadline = sys.argv[3], time.time() + float(sys.argv[4])
        last = None
        while time.time() < deadline:
            try:
                last = status(path, 2.0).get("state")
                if last == want:
                    print(want)
                    return
            except OSError as error:
                last = f"unreachable: {error}"
            time.sleep(0.1)
        print(f"timed out waiting for state {want!r}; last: {last}", file=sys.stderr)
        sys.exit(2)
    elif mode in ("burst", "flood"):
        # N concurrent framed requests. "burst" expects every connection to
        # receive a response echoing its request_id (idle-daemon pipeline
        # correctness); "flood" tolerates client-side timeouts for requests
        # parked behind real inference and reports how many were rejected
        # with a classified capacity response (queue-limit behavior).
        n = int(sys.argv[3])
        timeout = float(sys.argv[4])
        results = [None] * n
        def worker(i):
            request_id = f"{mode}-{i}"
            # The product client retries transient socket-level failures
            # (inference_daemon::request + retry_delays); a non-retrying raw
            # client would turn rare kernel-level connect/reset races under
            # churn into false negatives. Server RESPONSES are never retried
            # here — load shedding must stay observable.
            for attempt in range(3):
                try:
                    line = round_trip(
                        path,
                        {"protocol": 2, "op": "infer", "request_id": request_id},
                        timeout=timeout,
                    )
                    if not line:
                        raise OSError("connection closed without a response")
                    response = json.loads(line)
                    response["_echo_ok"] = response.get("request_id") == request_id
                    results[i] = response
                    return
                except OSError as error:
                    results[i] = {"_client_error": str(error)}
                    time.sleep(0.1 * (attempt + 1))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        responded = [r for r in results if r and "_client_error" not in r]
        def capacity(r):
            return r.get("error_kind") == "capacity" and r.get("retryable") is True
        print(json.dumps({
            "sent": n,
            "responded": len(responded),
            "echo_ok": sum(1 for r in responded if r.get("_echo_ok")),
            "capacity": sum(1 for r in responded if capacity(r)),
            # Accept-stage capacity rejections are written before the frame
            # is read, so they cannot echo a request_id; every OTHER response
            # must. This is the per-connection contract a non-retrying local
            # client observes.
            "echo_or_capacity": sum(
                1 for r in responded if r.get("_echo_ok") or capacity(r)
            ),
            "client_errors": sum(1 for r in results if r and "_client_error" in r),
            "client_error_detail": next(
                (r["_client_error"] for r in results if r and "_client_error" in r),
                None,
            ),
        }))
    else:
        sys.exit(64)

main()
PY

# Deadline-polling helpers (no bare sleeps as synchronization).
poll() { # poll <deadline-seconds> <description> <command...>
  local deadline
  deadline=$(( $(date +%s) + $1 )); shift
  local what="$1"; shift
  until "$@" >/dev/null 2>&1; do
    [ "$(date +%s)" -lt "$deadline" ] || fail "timed out waiting for: $what"
    sleep 0.2
  done
}

daemon_count() { pgrep -f -- "-daemon --socket $1" 2>/dev/null | wc -l | tr -d ' '; }
daemon_count_is() { [ "$(daemon_count "$1")" -eq "$2" ]; }
daemon_pid() { python3 "$CLIENT" status "$1" | jq -r '.daemon_pid'; }

# Every query below must be unique: the CLI caches query embeddings in the
# store (embed_query_cached), and only cache MISSES reach the daemon.
QUERY_SEQ=0
unique_query() {
  QUERY_SEQ=$((QUERY_SEQ + 1))
  printf 'stress probe %s %s %s' "$$" "$QUERY_SEQ" "$1"
}

# --- fixtures: one repo per daemon workload ----------------------------------
# repo-embed holds only struct definitions: semantic-search embeds the query
# via the embedding daemon but never requests Qwen purpose summaries (those
# are generated for function-like hits only), keeping the heavy concurrency
# sections off the multi-second summarize path.
# repo-brief holds functions so `brief` exercises the summarize daemon.
section "fixtures: index embed-only and brief repos"

REPO_EMBED="$WORK/repo-embed"
REPO_BRIEF="$WORK/repo-brief"
mkdir -p "$REPO_EMBED/src" "$REPO_EMBED/.git" "$REPO_BRIEF/src" "$REPO_BRIEF/.git"
cat >"$REPO_EMBED/src/lib.rs" <<'RS'
/// Configuration limits for score processing.
pub struct ScoreLimits {
    pub minimum: i32,
    pub maximum: i32,
}

/// A recorded score sample with its source label.
pub struct ScoreSample {
    pub label: String,
    pub value: i32,
}

/// Aggregated score statistics over a session window.
pub struct ScoreStats {
    pub mean: f64,
    pub count: usize,
}

/// Rolling window of recent samples for trend detection.
pub struct TrendWindow {
    pub samples: Vec<i32>,
    pub capacity: usize,
}
RS
cat >"$REPO_BRIEF/src/lib.rs" <<'RS'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
RS

"$BIN" --root "$REPO_EMBED" index "$REPO_EMBED" >"$WORK/out/index-embed.txt"
grep -Eq 'embedded [1-9][0-9]* code spans' "$WORK/out/index-embed.txt" \
  || fail "index left repo-embed without embedded spans: $(tail -1 "$WORK/out/index-embed.txt")"
"$BIN" --root "$REPO_BRIEF" index "$REPO_BRIEF" >"$WORK/out/index-brief.txt"
grep -Eq 'embedded [1-9][0-9]* code spans' "$WORK/out/index-brief.txt" \
  || fail "index left repo-brief without embedded spans: $(tail -1 "$WORK/out/index-brief.txt")"

# Daemon endpoints are published by doctor for exactly this kind of local
# diagnosis; they are the same addresses every CLI client derives.
"$BIN" --root "$REPO_EMBED" doctor --json >"$WORK/out/doctor.json" || test $? -eq 1
EMBED_SOCK="$(jq -re '.inference.daemons.embedding.endpoint' "$WORK/out/doctor.json")" \
  || fail "doctor --json did not report an embedding daemon endpoint"
SUMMARY_SOCK="$(jq -re '.inference.daemons.summary.endpoint' "$WORK/out/doctor.json")" \
  || fail "doctor --json did not report a summary daemon endpoint"
case "$EMBED_SOCK" in "$RUNTIME_BASE"/*) ;; *) fail "embedding endpoint escaped the contained runtime dir: $EMBED_SOCK" ;; esac
echo "embedding endpoint: $EMBED_SOCK"
echo "summary endpoint:   $SUMMARY_SOCK"

run_semantic() { # run_semantic <label> <query>
  "$BIN" --root "$REPO_EMBED" semantic-search "$2" --json >"$WORK/out/$1.json" \
    || fail "semantic-search '$2' exited $?"
  jq -e '.status == "ok" and (.hits | length) >= 1' "$WORK/out/$1.json" >/dev/null \
    || fail "semantic-search '$2' returned no ok hits"
}

# --- warm daemon + protocol sanity -------------------------------------------
section "warm daemon: spawn on demand, ping/status, protocol + malformed rejection"

run_semantic warmup "$(unique_query warmup)"
poll 15 "one embedding daemon owner" daemon_count_is "$EMBED_SOCK" 1
EMBED_PID="$(daemon_pid "$EMBED_SOCK")"
[ "$EMBED_PID" -gt 0 ] || fail "embedding daemon status did not report a pid"
pgrep -f -- "-daemon --socket $EMBED_SOCK" | grep -qx "$EMBED_PID" \
  || fail "status daemon_pid $EMBED_PID does not match the daemon process list"

python3 "$CLIENT" req "$EMBED_SOCK" '{"protocol":2,"op":"ping","request_id":"sanity-ping"}' >"$WORK/out/ping.json"
jq -e '.ok == true and .request_id == "sanity-ping"' "$WORK/out/ping.json" >/dev/null \
  || fail "ping did not return ok with the echoed request id: $(cat "$WORK/out/ping.json")"
python3 "$CLIENT" status "$EMBED_SOCK" >"$WORK/out/status.json"
jq -e '.protocol == 2 and .state == "ready" and .queue_capacity >= 1' "$WORK/out/status.json" >/dev/null \
  || fail "warm daemon status is not ready: $(cat "$WORK/out/status.json")"
python3 "$CLIENT" req "$EMBED_SOCK" '{"protocol":1,"op":"ping"}' >"$WORK/out/proto.json"
jq -e '.error == "protocol-version mismatch"' "$WORK/out/proto.json" >/dev/null \
  || fail "stale protocol version was not rejected: $(cat "$WORK/out/proto.json")"
python3 - "$EMBED_SOCK" <<'PY' >"$WORK/out/malformed.json"
import socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.settimeout(10)
s.connect(sys.argv[1]); s.sendall(b"not json\n")
buf = b""
while b"\n" not in buf:
    chunk = s.recv(65536)
    if not chunk:
        break
    buf += chunk
print(buf.split(b"\n")[0].decode())
PY
jq -e '.error == "malformed request"' "$WORK/out/malformed.json" >/dev/null \
  || fail "malformed frame was not rejected: $(cat "$WORK/out/malformed.json")"

# --- oversize request rejection ----------------------------------------------
# Embedding daemon request cap is 1 MiB (crates/cli/src/embed_daemon.rs
# MAX_REQUEST_BYTES). A frame past the cap must be rejected as soon as the
# limit is crossed (well before the 5s read timeout) and must not kill the
# daemon.
section "oversize request: rejected at the frame limit, daemon survives"

python3 "$CLIENT" oversize "$EMBED_SOCK" $((1048576 + 4096)) >"$WORK/out/oversize.json"
jq -e '.response.error == "request too large or incomplete" and .elapsed_s < 4' "$WORK/out/oversize.json" >/dev/null \
  || fail "oversize frame was not size-rejected: $(cat "$WORK/out/oversize.json")"
[ "$(daemon_pid "$EMBED_SOCK")" = "$EMBED_PID" ] || fail "oversize request killed or replaced the daemon"

# --- slow client + connection read timeout -----------------------------------
# A stalled partial frame must not block other clients, and the stalled
# connection itself must be rejected once the daemon's 5s connection read
# timeout expires (the externally observable request-timeout surface; the 60s
# request deadline and 75s hard watchdog are not externally tunable — see
# header).
section "slow client: does not block inference; stalled frame times out at ~5s"

python3 "$CLIENT" slow "$EMBED_SOCK" >"$WORK/out/slow.json" &
SLOW_PID=$!
run_semantic slow-bypass "$(unique_query slow-bypass)"
wait "$SLOW_PID" || fail "slow client did not receive a rejection response"
jq -e '.response.error == "request too large or incomplete" and .elapsed_s >= 4 and .elapsed_s < 15' \
  "$WORK/out/slow.json" >/dev/null \
  || fail "stalled frame was not timeout-rejected: $(cat "$WORK/out/slow.json")"
[ "$(daemon_pid "$EMBED_SOCK")" = "$EMBED_PID" ] || fail "slow client killed or replaced the daemon"

# --- concurrent clients, one owner, queue limits ------------------------------
# 5 real CLI clients (real model inference, all must succeed; the count is
# bounded so serialized real inference stays well inside the fixed 60s client
# deadline on slow CI hosts) + a 48-way framed flood inside the busy window
# (classified capacity rejections) + a 26-way echo burst (every connection
# answered with its own request_id), all against ONE daemon owner pid.
section "concurrency: 32+ connections, single owner, classified queue-limit rejections"

BEFORE_COMPLETED="$(python3 "$CLIENT" status "$EMBED_SOCK" | jq -r '.completed_requests')"
CLI_PIDS=""
for i in 1 2 3 4 5; do
  "$BIN" --root "$REPO_EMBED" semantic-search "$(unique_query "burst-$i")" --json \
    >"$WORK/out/burst-$i.json" 2>"$WORK/out/burst-$i.err" &
  CLI_PIDS="$CLI_PIDS $!"
done

python3 "$CLIENT" wait-active "$EMBED_SOCK" 60 >/dev/null \
  || fail "no in-flight inference observed while 5 CLI clients were queued"
python3 "$CLIENT" flood "$EMBED_SOCK" 48 20 >"$WORK/out/flood.json"
jq -e '.capacity >= 1' "$WORK/out/flood.json" >/dev/null \
  || fail "flood during a busy inference produced no classified capacity rejection: $(cat "$WORK/out/flood.json")"
jq -e '.responded >= .capacity and .responded >= 1' "$WORK/out/flood.json" >/dev/null \
  || fail "flood responses are inconsistent: $(cat "$WORK/out/flood.json")"
echo "flood: $(cat "$WORK/out/flood.json")"

for pid in $CLI_PIDS; do
  wait "$pid" || fail "a concurrent CLI client failed (see $WORK/out/burst-*.err)"
done
for i in 1 2 3 4 5; do
  jq -e '.status == "ok" and (.hits | length) >= 1' "$WORK/out/burst-$i.json" >/dev/null \
    || fail "concurrent CLI client $i returned no ok hits"
done

# Idle-pipeline burst: every one of 26 simultaneous connections must be
# answered promptly, either echoing its own request_id or shedding load with
# a classified retryable capacity response (a simultaneous burst can outrun
# the fixed reader pool even on an idle daemon; real clients retry those).
python3 "$CLIENT" burst "$EMBED_SOCK" 26 30 >"$WORK/out/burst.json"
jq -e '.responded == 26 and .echo_or_capacity == 26 and .client_errors == 0' "$WORK/out/burst.json" >/dev/null \
  || fail "echo burst lost connections or misclassified responses: $(cat "$WORK/out/burst.json")"
echo "burst: $(cat "$WORK/out/burst.json")"

poll 15 "one embedding daemon owner after the burst" daemon_count_is "$EMBED_SOCK" 1
[ "$(daemon_pid "$EMBED_SOCK")" = "$EMBED_PID" ] \
  || fail "daemon owner changed during the concurrency burst"
AFTER_COMPLETED="$(python3 "$CLIENT" status "$EMBED_SOCK" | jq -r '.completed_requests')"
[ "$((AFTER_COMPLETED - BEFORE_COMPLETED))" -ge 5 ] \
  || fail "daemon served fewer than the 5 real embeddings ($BEFORE_COMPLETED -> $AFTER_COMPLETED)"
REJECTED="$(python3 "$CLIENT" status "$EMBED_SOCK" | jq -r '.rejected_requests')"
[ "$REJECTED" -ge 1 ] || fail "daemon status shows no rejected requests after the flood"

# --- kill -9 mid-load: clean respawn, correct answers -------------------------
# SIGKILL the owner while requests are in flight. The endpoint file survives
# the kill (no Drop runs), so the next client must classify the dead endpoint,
# spawn a fresh owner, repair the socket, and serve correct answers.
section "kill -9 mid-load: stale socket repaired, fresh owner, correct answers"

for i in 1 2 3; do
  "$BIN" --root "$REPO_EMBED" semantic-search "$(unique_query "kill-$i")" --json \
    >"$WORK/out/kill-$i.json" 2>&1 &
done
python3 "$CLIENT" wait-active "$EMBED_SOCK" 60 >/dev/null \
  || fail "no in-flight inference observed before the kill"
kill -9 "$EMBED_PID"
poll 15 "SIGKILLed daemon to disappear" bash -c "! kill -0 $EMBED_PID 2>/dev/null"
# In-flight clients may fail (their daemon died mid-request; the client
# contract only spawns on a PROVEN-absent daemon) — that is accepted, not
# asserted. Reap them without failing the section.
wait >/dev/null 2>&1 || true
[ -S "$EMBED_SOCK" ] || fail "SIGKILL should leave the stale socket file behind (endpoint teardown is a clean-exit path)"

run_semantic kill-recovery "$(unique_query kill-recovery)"
poll 15 "one respawned embedding daemon owner" daemon_count_is "$EMBED_SOCK" 1
NEW_EMBED_PID="$(daemon_pid "$EMBED_SOCK")"
[ "$NEW_EMBED_PID" != "$EMBED_PID" ] || fail "daemon pid did not change across kill -9"
jq -e '.state == "ready"' <(python3 "$CLIENT" status "$EMBED_SOCK") >/dev/null \
  || fail "respawned daemon is not ready"
EMBED_PID="$NEW_EMBED_PID"

# --- stale endpoint recovery: regular file at the socket path -----------------
# Harder variant of stale-endpoint repair: the endpoint path holds a plain
# file, not a dead socket (macOS reports ENOTSOCK on connect there). The next
# client must still treat the endpoint as daemon-less, spawn an owner, and the
# owner must replace the junk file with a live socket.
section "stale endpoint: junk regular file is repaired by the next spawn"

kill -9 "$EMBED_PID"
poll 15 "killed daemon to disappear" bash -c "! kill -0 $EMBED_PID 2>/dev/null"
rm -f "$EMBED_SOCK"
printf 'stale' >"$EMBED_SOCK"
run_semantic stale-recovery "$(unique_query stale-recovery)"
[ -S "$EMBED_SOCK" ] || fail "endpoint path was not repaired back into a socket"
poll 15 "one embedding daemon owner after repair" daemon_count_is "$EMBED_SOCK" 1
EMBED_PID="$(daemon_pid "$EMBED_SOCK")"

# --- summarize daemon: real brief, queue limits, oversize ----------------------
# One real Qwen summary (multi-second on CPU) provides the busy window for
# the summarize daemon's queue-limit flood, then proves the daemon (not the
# deterministic fallback) served the brief: completed_requests advanced and
# no error was recorded.
section "summarize daemon: brief served by the daemon, queue limits, oversize"

"$BIN" --root "$REPO_BRIEF" brief apply_limit --json >"$WORK/out/brief.json" 2>"$WORK/out/brief.err" &
BRIEF_PID=$!
python3 "$CLIENT" wait-active "$SUMMARY_SOCK" 120 >/dev/null \
  || fail "summarize daemon never reported the brief request in flight"
# Short client timeout: capacity rejections are written immediately by the
# accept/reader stages; only requests parked behind the real summary would
# wait longer, and abandoning them must not harm the daemon (asserted below).
python3 "$CLIENT" flood "$SUMMARY_SOCK" 48 5 >"$WORK/out/summary-flood.json"
jq -e '.capacity >= 1' "$WORK/out/summary-flood.json" >/dev/null \
  || fail "summarize flood produced no classified capacity rejection: $(cat "$WORK/out/summary-flood.json")"
echo "summary flood: $(cat "$WORK/out/summary-flood.json")"

wait "$BRIEF_PID" || fail "brief failed under flood: $(cat "$WORK/out/brief.err")"
jq -e '
  .schema_version == "greppy.brief.v1" and
  .status == "ok" and
  (.definitions | length) >= 1 and
  (.definitions[0].summary | length) >= 1
' "$WORK/out/brief.json" >/dev/null || fail "brief JSON contract failed under daemon stress"
python3 "$CLIENT" status "$SUMMARY_SOCK" >"$WORK/out/summary-status.json"
jq -e '.completed_requests >= 1 and .last_error == null and .daemon_pid > 0' \
  "$WORK/out/summary-status.json" >/dev/null \
  || fail "summarize daemon did not serve the brief cleanly: $(cat "$WORK/out/summary-status.json")"
poll 15 "one summarize daemon owner" daemon_count_is "$SUMMARY_SOCK" 1

python3 "$CLIENT" oversize "$SUMMARY_SOCK" $((262144 + 4096)) >"$WORK/out/summary-oversize.json"
jq -e '.response.error == "request too large or incomplete"' "$WORK/out/summary-oversize.json" >/dev/null \
  || fail "summarize daemon accepted an oversize frame: $(cat "$WORK/out/summary-oversize.json")"

# --- idle eviction: model TTL, then idle exit with socket teardown ------------
# The product's external TTL knobs (GREPPY_*_DAEMON_MODEL_TTL_S/_EXIT_TTL_S,
# read by the daemon at spawn) shrink both idle horizons to seconds. Status
# polling refreshes the EXIT clock but not the MODEL clock, so the eviction is
# observed via status and the exit strictly via process/socket observation.
section "idle eviction: model evicted on TTL, daemon exits and removes its socket"

kill -9 "$EMBED_PID"
poll 15 "long-TTL daemon to disappear" bash -c "! kill -0 $EMBED_PID 2>/dev/null"
(
  export GREPPY_EMBED_DAEMON_MODEL_TTL_S=2
  export GREPPY_EMBED_DAEMON_EXIT_TTL_S=8
  run_semantic evict-warmup "$(unique_query evict-warmup)"
)
poll 15 "short-TTL embedding daemon owner" daemon_count_is "$EMBED_SOCK" 1
EMBED_PID="$(daemon_pid "$EMBED_SOCK")"
python3 "$CLIENT" wait-state "$EMBED_SOCK" evicted 30 >/dev/null \
  || fail "embedding model was not evicted after its 2s idle TTL"
# A request after eviction must transparently reload one model instance.
run_semantic evict-reload "$(unique_query evict-reload)"
jq -e '.state == "ready" or .state == "evicted"' <(python3 "$CLIENT" status "$EMBED_SOCK") >/dev/null \
  || fail "daemon did not return to a healthy state after the post-eviction reload"
# Hands off from here: any status request would refresh the exit clock.
poll 60 "short-TTL daemon idle exit" bash -c "! kill -0 $EMBED_PID 2>/dev/null"
poll 15 "endpoint socket teardown on clean exit" bash -c "! test -e '$EMBED_SOCK'"

# The summarize daemon (short TTLs since spawn) must have drained on its own
# idle clock as well — its last activity was the brief section above.
poll 90 "summarize daemon idle exit" daemon_count_is "$SUMMARY_SOCK" 0
poll 15 "summary endpoint socket teardown" bash -c "! test -e '$SUMMARY_SOCK'"

printf '\nrelease daemon stress passed: %s\n' "$BIN"
