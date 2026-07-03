#!/usr/bin/env bash
# Shared helpers for the battle-proof validation harness (Track C).
#
# These scripts are a BLACK-BOX suite: they drive the already-built
# binaries and assert production invariants. They never touch crate
# source or Cargo files.
#
# Conventions every battle script follows:
#   * Each check prints a single line "PASS <desc>" or "FAIL <desc>".
#   * Each script ends with a machine-parseable summary line:
#         BATTLE_SUMMARY <name> pass=<n> fail=<n>
#   * Exit status is non-zero iff any check FAILed.
#
# The aggregator (run_battle.sh) sources nothing from here directly; it
# parses the BATTLE_SUMMARY line. Individual scripts `source lib.sh`.

set -uo pipefail

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
BATTLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$BATTLE_DIR/../.." && pwd)}"

GREPPLUS_BIN="${GREPPLUS_BIN:-$WORKSPACE_ROOT/target/debug/grepplus}"
GREPPLUS_GREP_BIN="${GREPPLUS_GREP_BIN:-$WORKSPACE_ROOT/target/debug/grepplus-grep}"

# Real grep used as the byte-exact oracle for the drop-in contract.
REAL_GREP="${REAL_GREP:-/usr/bin/grep}"
if [[ ! -x "$REAL_GREP" ]]; then
    REAL_GREP="$(command -v grep || echo /usr/bin/grep)"
fi

# ---------------------------------------------------------------------------
# Per-script counters + reporting
# ---------------------------------------------------------------------------
declare -i BATTLE_PASS=0
declare -i BATTLE_FAIL=0

# pass <desc>
pass() {
    BATTLE_PASS+=1
    printf 'PASS %s\n' "$*"
}

# fail <desc>
fail() {
    BATTLE_FAIL+=1
    printf 'FAIL %s\n' "$*"
}

# check <cond-rc> <desc>: pass iff $1 == 0.
check() {
    local rc="$1"; shift
    if [[ "$rc" -eq 0 ]]; then
        pass "$*"
    else
        fail "$*"
    fi
}

# assert_eq <expected> <actual> <desc>
assert_eq() {
    local expected="$1" actual="$2"; shift 2
    if [[ "$expected" == "$actual" ]]; then
        pass "$* (=$actual)"
    else
        fail "$* (expected=$expected actual=$actual)"
    fi
}

# assert_ge <value> <min> <desc>: pass iff value >= min (integers).
assert_ge() {
    local value="$1" min="$2"; shift 2
    if [[ "$value" -ge "$min" ]]; then
        pass "$* ($value >= $min)"
    else
        fail "$* ($value < $min)"
    fi
}

# emit_summary <name>: print the machine-parseable summary line and
# return non-zero if any check failed.
emit_summary() {
    local name="$1"
    echo ""
    echo "BATTLE_SUMMARY $name pass=$BATTLE_PASS fail=$BATTLE_FAIL"
    [[ "$BATTLE_FAIL" -eq 0 ]]
}

# ---------------------------------------------------------------------------
# Build guard
# ---------------------------------------------------------------------------
require_bins() {
    local missing=0
    for b in "$@"; do
        if [[ ! -x "$b" ]]; then
            echo "MISSING BINARY: $b" >&2
            missing=1
        fi
    done
    if [[ "$missing" -ne 0 ]]; then
        echo "Build first: (cd '$WORKSPACE_ROOT' && cargo build --bins)" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# SQLite helper — graph.db lives at $store/<ws-hash>/graph.db
# ---------------------------------------------------------------------------
graph_db_path() {
    # $1 = store dir
    find "$1" -name graph.db -type f 2>/dev/null | head -n1
}

sqlite_q() {
    # $1 = db, rest = query
    local db="$1"; shift
    sqlite3 "$db" "$*"
}

# git_init_corpus <dir>: make <dir> a committed git repo so the
# discovery layer treats it as a real workspace.
git_init_corpus() {
    local dir="$1"
    git -C "$dir" init -q .
    git -C "$dir" add -A
    git -C "$dir" -c user.email=battle@grepplus.test \
                  -c user.name='battle' \
                  -c commit.gpgsign=false \
                  commit -q -m 'battle corpus' >/dev/null 2>&1 || true
}

# rss_kb <pid>: resident set size in KB (macOS + Linux).
rss_kb() {
    ps -o rss= -p "$1" 2>/dev/null | tr -d ' '
}

# rss_kb_tree <pid>: peak-friendly RSS in KB for <pid> AND all of its
# descendants, summed. This is the robust sampler: even if a wrapper
# shell forks the real worker as a child (so the backgrounded PID is the
# wrapper, not the worker), the worker's RSS is still counted.
#
# Why this exists: the old samplers backgrounded `( cd dir && bin ... ) &`
# and sampled `$!`. On Linux bash does NOT exec-optimise a compound
# `cd && bin`, so `$!` is a tiny ~1.5 MB subshell and the indexer's real
# RSS was never measured (it always reported ~1.5 MB). Summing the whole
# process tree fixes that on every platform regardless of exec-optimisation.
rss_kb_tree() {
    local root="$1"
    [[ -n "$root" ]] || { echo 0; return; }
    # ONE `ps` snapshot, ONE awk pass: compute the transitive closure of
    # <root> over the (pid,ppid) graph and sum RSS. Iterating to a fixed
    # point inside awk keeps this to two subprocesses per sample (cheap
    # enough for a 50 ms loop) regardless of tree depth.
    # `-A`/`ax` is essential: a bare `ps -o …` only lists the controlling
    # terminal's processes, which (inside a `$(…)` subshell) can omit the
    # very worker we are sampling. We need the FULL process table to walk
    # the tree.
    ps -A -o pid=,ppid=,rss= 2>/dev/null | awk -v root="$root" '
        { pid[NR]=$1; ppid[NR]=$2; rss[NR]=$3 }
        END {
            in_tree[root]=1
            changed=1
            while (changed) {
                changed=0
                for (i=1; i<=NR; i++) {
                    if (!in_tree[pid[i]] && in_tree[ppid[i]]) {
                        in_tree[pid[i]]=1
                        changed=1
                    }
                }
            }
            total=0
            for (i=1; i<=NR; i++) if (in_tree[pid[i]]) total+=rss[i]
            print total
        }'
}

# index_bg_peak_rss <log> <pidvar> <rssvar> -- <cmd...>: run <cmd...> in
# the BACKGROUND with stdout/stderr to <log>, store the worker PID in the
# variable named <pidvar>, and sample peak whole-tree RSS (KB) into the
# variable named <rssvar> while it runs. Returns 0 (the command's own exit
# status is obtained by the caller via `wait "$pid"`).
#
# Callers that need a specific working directory should pass the command
# as `env -C <dir> ...` or rely on a `cd` done by the caller; this helper
# intentionally does NOT wrap in a `cd` subshell, so the backgrounded PID
# IS the worker process (the bug the old harness had).
#
# Usage:
#   index_bg_peak_rss "$log" idx_pid peak_rss -- \
#       env GREPPLUS_STORE_DIR="$store" "$GREPPLUS_BIN" index "$corpus"
#   wait "$idx_pid"; rc=$?
index_bg_peak_rss() {
    local log="$1" pidvar="$2" rssvar="$3"
    shift 3
    [[ "$1" == "--" ]] && shift
    "$@" >"$log" 2>&1 &
    local pid=$!
    printf -v "$pidvar" '%s' "$pid"
    local peak=0 cur
    while kill -0 "$pid" 2>/dev/null; do
        cur="$(rss_kb_tree "$pid")"
        [[ -n "$cur" && "$cur" -gt "$peak" ]] && peak="$cur"
        sleep 0.05
    done
    printf -v "$rssvar" '%s' "$peak"
    return 0
}
