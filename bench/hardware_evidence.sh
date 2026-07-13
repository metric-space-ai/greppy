#!/usr/bin/env bash
# Hardware-evidence harness: one reproducible inference-evidence run of a
# greppy binary on real hardware, emitted as a JSON artifact conforming to
# bench/hardware-evidence.schema.json (committed artifacts live under
# bench/hardware-evidence/, see the README there).
#
# The script is self-contained like bench/release_package_smoke.sh: the
# fixture is generated inline, and only tooling that exists on stock Linux
# and macOS is used (bash 3.2+, jq, perl, cmp, find, shasum or sha256sum;
# nvidia-smi and ldd for the CUDA leg). Target runtime: well under 10
# minutes per platform.
#
# Privacy contract: the artifact is public release evidence. It records
# hardware make/model, driver and OS versions, and product/model digests —
# and NOTHING host-identifying: no hostnames, no usernames, no serial
# numbers, no absolute paths. A final scrub gate greps the artifact for the
# local hostname/username and refuses to emit on a hit.
#
# Usage:
#   hardware_evidence.sh /path/to/greppy --backend cpu|metal|cuda \
#     [--out FILE] [--source-sha SHA] [--calls N] [--require-baseline-x86]
#
# CUDA leg: run on the intended GPU only, e.g.
#   CUDA_VISIBLE_DEVICES=0 hardware_evidence.sh ./greppy --backend cuda ...
# The harness strips every directory containing `nvcc` from PATH and unsets
# LD_LIBRARY_PATH before invoking the product, then asserts the bundled
# backend library needs no CUDA toolkit at runtime (driver only): the
# artifact proves the SHIPPED binary works on a box without a toolkit.

set -euo pipefail

BIN="${1:?usage: hardware_evidence.sh /path/to/greppy --backend cpu|metal|cuda [--out FILE] [--source-sha SHA] [--calls N] [--require-baseline-x86]}"
shift
case "$BIN" in /*) ;; *) BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")" ;; esac
[ -x "$BIN" ] || { echo "not executable: $BIN" >&2; exit 64; }

BACKEND=""
OUT=""
SOURCE_SHA="${GREPPY_EVIDENCE_SOURCE_SHA:-unknown}"
CALLS=20
REQUIRE_BASELINE_X86=0
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) BACKEND="${2:?--backend needs cpu|metal|cuda}"; shift 2 ;;
    --out) OUT="${2:?--out needs a file path}"; shift 2 ;;
    --source-sha) SOURCE_SHA="${2:?--source-sha needs a git sha}"; shift 2 ;;
    --calls) CALLS="${2:?--calls needs a positive integer}"; shift 2 ;;
    --require-baseline-x86) REQUIRE_BASELINE_X86=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
case "$BACKEND" in cpu|metal|cuda) ;; *) echo "--backend must be cpu, metal, or cuda" >&2; exit 64 ;; esac
case "$CALLS" in ''|*[!0-9]*|0) echo "--calls must be a positive integer" >&2; exit 64 ;; esac

command -v jq >/dev/null 2>&1 || { echo "hardware_evidence: jq is required" >&2; exit 69; }
command -v perl >/dev/null 2>&1 || { echo "hardware_evidence: perl is required (millisecond timing)" >&2; exit 69; }

section() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

if command -v sha256sum >/dev/null 2>&1; then
  HASH_CMD="sha256sum"
else
  HASH_CMD="shasum -a 256"
fi

now_ms() { perl -MTime::HiRes=time -e 'printf "%d\n", time()*1000'; }

# Canonical content digest of a directory tree (same construction as
# bench/release_package_smoke.sh dir_digest): platform-independent for an
# identical fixture, so artifacts from different machines are comparable.
dir_digest() {
  (
    cd "$1" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do
      $HASH_CMD "$f"
    done
  ) | $HASH_CMD | awk '{print $1}'
}

# --- environment sanitization -------------------------------------------------
# CUDA leg contract: the shipped binary must run WITHOUT a CUDA toolkit —
# it bundles an nvcc-prebuilt backend library (linked --cudart=static) and
# needs only the driver's libcuda.so.1. Prove it by hiding every toolkit
# directory from the product.
CUDA_TOOLKIT_VISIBLE="null"
if [ "$BACKEND" = "cuda" ]; then
  command -v nvidia-smi >/dev/null 2>&1 || fail "cuda backend requested but nvidia-smi is unavailable"
  SANITIZED_PATH=""
  OLD_IFS="$IFS"; IFS=':'
  for dir in $PATH; do
    [ -n "$dir" ] || continue
    [ -x "$dir/nvcc" ] && continue
    case "$dir" in */cuda*/bin|*/cuda/bin) continue ;; esac
    SANITIZED_PATH="${SANITIZED_PATH:+$SANITIZED_PATH:}$dir"
  done
  IFS="$OLD_IFS"
  PATH="$SANITIZED_PATH"
  export PATH
  unset LD_LIBRARY_PATH CUDA_HOME CUDA_PATH || true
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_TOOLKIT_VISIBLE="true"
  else
    CUDA_TOOLKIT_VISIBLE="false"
  fi
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/greppy-hw-evidence-XXXXXX")"
VRAM_POLL_PID=""
cleanup() {
  [ -n "$VRAM_POLL_PID" ] && kill "$VRAM_POLL_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT
mkdir -p "$WORK/repo/src" "$WORK/repo/.git" "$WORK/store"

# --- deterministic fixture ----------------------------------------------------
# Identical to the bench/release_package_smoke.sh fixture so cross-platform
# artifacts measure the same workload (tree_sha256 in the artifact pins it).
section "fixture"

cat >"$WORK/repo/src/lib.rs" <<'RS'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
pub fn validate_score(value: i32) -> bool { value <= 100 }
pub fn default_score() -> i32 { 50 }
pub fn minimum_score() -> i32 { 0 }
pub fn maximum_score() -> i32 { 100 }
RS

cat >"$WORK/repo/src/case.rs" <<'RS'
#[derive(Copy, Clone)]
pub enum RenameRule {
    LowerCase,
    UpperCase,
    SnakeCase,
}

impl RenameRule {
    /// Apply a rename case rule to a struct field name.
    pub fn apply_to_field(self, field: &str) -> String {
        match self {
            RenameRule::LowerCase => field.to_lowercase(),
            RenameRule::UpperCase => field.to_uppercase(),
            RenameRule::SnakeCase => field.to_string(),
        }
    }
}

pub struct RenameAllRules {
    pub serialize: RenameRule,
    pub deserialize: RenameRule,
}

pub struct Name {
    pub serialize: String,
    pub deserialize: String,
}

impl Name {
    /// Rename the serialize and deserialize names by the container rules.
    pub fn rename_by_rules(&mut self, rules: &RenameAllRules) {
        self.serialize = rules.serialize.apply_to_field(&self.serialize);
        self.deserialize = rules.deserialize.apply_to_field(&self.deserialize);
    }

    /// Return the field name used when serializing.
    pub fn serialize_name(&self) -> &str {
        &self.serialize
    }
}
RS

FIXTURE_FILES="$(find "$WORK/repo" -type f | grep -c '' || true)"
FIXTURE_DIGEST="$(dir_digest "$WORK/repo")"
echo "fixture: $FIXTURE_FILES files, tree $FIXTURE_DIGEST"

export GREPPY_STORE_DIR="$WORK/store"
# Keep daemons warm across the latency series, but let them drain quickly
# once the run ends (the CUDA leg must release the GPU promptly).
export GREPPY_EMBED_DAEMON_MODEL_TTL_S=120
export GREPPY_EMBED_DAEMON_EXIT_TTL_S=20
export GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S=120
export GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S=20
# Contain daemon sockets in a directory this script owns. Prefer the real
# XDG_RUNTIME_DIR (a tmpfs on Linux — /tmp may live on a full disk), fall
# back to /tmp; both keep the joined socket path unix-socket-safe short
# (crates/cli/src/inference_daemon.rs unix_runtime_dir()).
RUNTIME_BASE="$(mktemp -d "${XDG_RUNTIME_DIR:-/tmp}/ghe-XXXXXX")"
chmod 700 "$RUNTIME_BASE"
export XDG_RUNTIME_DIR="$RUNTIME_BASE"

drain_daemons() {
  local deadline=$(( $(date +%s) + 180 ))
  while pgrep -f '(embed|summarize)-daemon --socket' >/dev/null 2>&1; do
    [ "$(date +%s)" -lt "$deadline" ] || fail "inference daemons did not exit within 180s"
    sleep 1
  done
}

# --- host fingerprint (scrubbed) ----------------------------------------------
section "host fingerprint"

OS_RAW="$(uname -s)"
case "$OS_RAW" in
  Linux) OS="linux"; OS_VERSION="$(uname -r)" ;;
  Darwin) OS="macos"; OS_VERSION="$(sw_vers -productVersion)" ;;
  *) fail "unsupported platform: $OS_RAW" ;;
esac
ARCH="$(uname -m)"
if [ "$OS" = "linux" ]; then
  CPU_MODEL="$(awk -F': ' '/^model name/{print $2; exit}' /proc/cpuinfo)"
  CPU_CORES="$(nproc)"
  RAM_GB="$(awk '/^MemTotal:/{printf "%.0f", $2/1048576}' /proc/meminfo)"
else
  CPU_MODEL="$(sysctl -n machdep.cpu.brand_string)"
  CPU_CORES="$(sysctl -n hw.ncpu)"
  RAM_GB="$(( $(sysctl -n hw.memsize) / 1073741824 ))"
fi
APPLE_GEN=""
if [ "$OS" = "macos" ]; then
  APPLE_GEN="$(printf '%s\n' "$CPU_MODEL" | sed -n 's/^Apple \(M[0-9][0-9]*\).*/\1/p')"
fi

GPU_NAME=""; GPU_DRIVER=""; GPU_VRAM_MIB=""; GPU_INDEX="${GREPPY_EVIDENCE_GPU_INDEX:-0}"
if [ "$BACKEND" = "cuda" ]; then
  GPU_LINE="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)" \
    || fail "nvidia-smi could not query GPU index $GPU_INDEX"
  GPU_NAME="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $1}')"
  GPU_DRIVER="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $2}')"
  GPU_VRAM_MIB="$(printf '%s' "$GPU_LINE" | awk -F', ' '{print $3}')"
fi
echo "host: $OS $OS_VERSION $ARCH, $CPU_MODEL ($CPU_CORES cores, ${RAM_GB}GB)"
[ -n "$GPU_NAME" ] && echo "gpu:  $GPU_NAME (driver $GPU_DRIVER, ${GPU_VRAM_MIB}MiB)"

# --- product fingerprint -------------------------------------------------------
section "product fingerprint"

VERSION="$("$BIN" --version | head -1)"
BINARY_SHA="$($HASH_CMD "$BIN" | awk '{print $1}')"
echo "$VERSION (binary sha256 $BINARY_SHA, source $SOURCE_SHA)"

# --- checks accumulator --------------------------------------------------------
CHECKS="$WORK/checks.tsv"
: >"$CHECKS"
OVERALL=pass
check() {
  # check NAME PASS(0|1) [DETAIL]
  local name="$1" ok="$2" detail="${3:-}"
  local verdict=true
  if [ "$ok" -ne 0 ]; then verdict=false; OVERALL=fail; fi
  printf '%s\t%s\t%s\n' "$name" "$verdict" "$detail" >>"$CHECKS"
  if [ "$verdict" = true ]; then
    printf 'ok   %s\n' "$name"
  else
    printf 'FAIL %s%s\n' "$name" "${detail:+ ($detail)}" >&2
  fi
}

# --- backend selection (doctor, pre-index) -------------------------------------
section "backend selection: doctor --json"

DEVICE_FLAG="$BACKEND"
"$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" doctor --json >"$WORK/doctor.json" || test $? -eq 1
jq -e '.command == "doctor"' "$WORK/doctor.json" >/dev/null || fail "doctor --json did not produce a doctor report"

SELECTED_BACKEND="$(jq -r '.inference.registry.selected_backend // "none"' "$WORK/doctor.json")"
SELECTED_DEVICE="$(jq -r '.inference.registry.selected_device_id // empty' "$WORK/doctor.json")"
BACKEND_COMPILED="$(jq -r --arg b "$BACKEND" '[.inference.registry.probes[] | select(.backend == $b)][0].compiled // false' "$WORK/doctor.json")"
BACKEND_BUILD_INFO="$(jq -r --arg b "$BACKEND" '[.inference.registry.probes[] | select(.backend == $b)][0].build_info // ""' "$WORK/doctor.json")"
CPU_CAPS_JSON="$(jq -c '[.inference.registry.probes[] | select(.backend == "cpu") | .devices[].capabilities[]] | unique' "$WORK/doctor.json")"
METAL_FAMILY="$(jq -r '[.inference.registry.probes[] | select(.backend == "metal") | .devices[] | select(.rejection_reason == null) | .metal_family][0] // empty' "$WORK/doctor.json")"
METAL_TENSOR="$(jq -r '[.inference.registry.probes[] | select(.backend == "metal") | .devices[] | select(.rejection_reason == null) | .capabilities[]] | any(. == "tensor-ops")' "$WORK/doctor.json")"
CUDA_COMPUTE_CAP="$(jq -r '[.inference.registry.probes[] | select(.backend == "cuda") | .devices[] | select(.rejection_reason == null) | .compute_capability][0] // empty' "$WORK/doctor.json")"

ok=0; [ "$SELECTED_BACKEND" = "$BACKEND" ] || ok=1
check "doctor selects requested backend ($BACKEND)" $ok "selected=$SELECTED_BACKEND device=${SELECTED_DEVICE:-none}"
ok=0; [ "$BACKEND_COMPILED" = "true" ] || ok=1
check "requested backend compiled into binary" $ok "$BACKEND_BUILD_INFO"

MODELS_JSON="$(jq -c '{
  embedding: {
    model_id: .inference.models.embedding.model_id,
    model_sha256: .inference.models.embedding.model_sha256,
    tokenizer_sha256: .inference.models.embedding.tokenizer_sha256
  },
  summary: {
    model_id: .inference.models.summary.model_id,
    model_sha256: .inference.models.summary.model_sha256,
    tokenizer_sha256: .inference.models.summary.tokenizer_sha256
  }
}' "$WORK/doctor.json")"
ok=0
jq -e '(.embedding.model_sha256 | type == "string" and length == 64) and
       (.summary.model_sha256 | type == "string" and length == 64)' \
  >/dev/null <<<"$MODELS_JSON" || ok=1
check "doctor reports embedded model digests" $ok

if [ "$REQUIRE_BASELINE_X86" -eq 1 ]; then
  ok=0; [ "$ARCH" = "x86_64" ] || ok=1
  check "baseline-x86: architecture is x86_64" $ok "arch=$ARCH"
  ok=0
  jq -e 'any(.[]; . == "avx-vnni" or . == "avx512f") | not' >/dev/null <<<"$CPU_CAPS_JSON" || ok=1
  check "baseline-x86: CPU lacks AVX-VNNI/AVX-512 (SIMD fallback paths in use)" $ok "capabilities=$CPU_CAPS_JSON"
fi

# --- VRAM polling (cuda) --------------------------------------------------------
VRAM_BASELINE=""; VRAM_SAMPLES="$WORK/vram.samples"; VRAM_POLL_INTERVAL="0.5"
if [ "$BACKEND" = "cuda" ]; then
  VRAM_BASELINE="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  (
    while :; do
      nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1
      sleep "$VRAM_POLL_INTERVAL"
    done
  ) >"$VRAM_SAMPLES" &
  VRAM_POLL_PID=$!
fi

# --- measurements ----------------------------------------------------------------
section "measure: cold index"

T0="$(now_ms)"
"$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" index "$WORK/repo" >"$WORK/index.txt"
INDEX_MS=$(( $(now_ms) - T0 ))
check "index completes on fixture" 0 "${INDEX_MS}ms"
echo "index: ${INDEX_MS}ms"

QUERY="restrict a numeric value to an allowed range"
SYMBOL="apply_limit"

measure_series() {
  # measure_series LABEL OUTFILE -- CMD ARGS...
  local label="$1" outfile="$2"; shift 3
  : >"$outfile"
  "$@" >/dev/null   # warm-up call, unmeasured
  local i=1 t0 t1
  while [ "$i" -le "$CALLS" ]; do
    t0="$(now_ms)"
    "$@" >/dev/null
    t1="$(now_ms)"
    echo $(( t1 - t0 )) >>"$outfile"
    i=$(( i + 1 ))
  done
}

# Nearest-rank percentiles plus min/max over a sample file, as a JSON object.
series_json() {
  LC_ALL=C sort -n "$1" | awk -v calls="$CALLS" '
    { v[NR] = $1 }
    END {
      p50 = v[int((NR * 50 + 99) / 100)];
      p95 = v[int((NR * 95 + 99) / 100)];
      printf "{\"calls\":%d,\"p50_ms\":%d,\"p95_ms\":%d,\"min_ms\":%d,\"max_ms\":%d}", calls, p50, p95, v[1], v[NR]
    }'
}

section "measure: semantic-search latency ($CALLS calls)"
measure_series semantic-search "$WORK/search.samples" -- \
  "$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" semantic-search "$QUERY" --json
SEARCH_JSON="$(series_json "$WORK/search.samples")"
echo "semantic-search: $SEARCH_JSON"

section "measure: brief latency ($CALLS calls)"
measure_series brief "$WORK/brief.samples" -- \
  "$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" brief "$SYMBOL" --json
BRIEF_JSON="$(series_json "$WORK/brief.samples")"
echo "brief: $BRIEF_JSON"

# --- functional contract checks ---------------------------------------------------
section "contract checks"

"$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" brief "$SYMBOL" --json >"$WORK/brief.json"
ok=0
jq -e '
  .schema_version == "greppy.brief.v1" and
  .status == "ok" and
  (.definitions | length) >= 1 and
  (.definitions[0].summary | length) >= 1 and
  (.expand_id | type == "string" and length > 0)
' "$WORK/brief.json" >/dev/null || ok=1
check "brief returns definition with summary" $ok

"$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" semantic-search "$QUERY" --json >"$WORK/semantic.json"
ok=0
jq -e '
  .schema_version == "greppy.semantic-search.v1" and
  .status == "ok" and
  (.hits | length) >= 1 and
  ([.hits[].score] | . == (sort | reverse))
' "$WORK/semantic.json" >/dev/null || ok=1
check "semantic-search returns ranked hits" $ok

EXPAND_ID="$(jq -r '.expand_id' "$WORK/semantic.json")"
ok=0
{ "$BIN" --root "$WORK/repo" expand "$EXPAND_ID" --json >"$WORK/expand.json" 2>/dev/null \
    && jq -e --arg id "$EXPAND_ID" '.id == $id and (.payload_text | length > 0)' "$WORK/expand.json" >/dev/null; } || ok=1
check "expand resolves search evidence" $ok

"$BIN" --device "$DEVICE_FLAG" --root "$WORK/repo" semantic-search "$QUERY" --json >"$WORK/semantic2.json"
ok=0
[ "$(jq -c '[.hits[].qualified_name]' "$WORK/semantic.json")" = "$(jq -c '[.hits[].qualified_name]' "$WORK/semantic2.json")" ] || ok=1
check "semantic-search ordering is deterministic across reruns" $ok

# grep passthrough: byte-exact vs system grep (same tier-2 discovery as
# bench/release_package_smoke.sh — never `command -v grep`).
REAL_GREP=""
for candidate in /usr/bin/grep /bin/grep; do
  if [ -x "$candidate" ]; then REAL_GREP="$candidate"; break; fi
done
if [ -n "$REAL_GREP" ]; then
  export GREPPY_REAL_GREP="$REAL_GREP"
  rc=0; ( cd "$WORK/repo" && "$BIN" -n apply_limit src/lib.rs ) >"$WORK/g1.out" 2>"$WORK/g1.err" || rc=$?
  erc=0; ( cd "$WORK/repo" && "$REAL_GREP" -n apply_limit src/lib.rs ) >"$WORK/g2.out" 2>"$WORK/g2.err" || erc=$?
  ok=0
  { [ "$rc" -eq "$erc" ] && cmp -s "$WORK/g1.out" "$WORK/g2.out" && cmp -s "$WORK/g1.err" "$WORK/g2.err"; } || ok=1
  check "grep passthrough byte-exact vs system grep" $ok
  unset GREPPY_REAL_GREP
fi

# CUDA runtime-only contract: the backend library the binary materialized
# must not need any CUDA toolkit library — only the driver's libcuda.so.1.
CUDA_NEEDED_JSON="null"
if [ "$BACKEND" = "cuda" ]; then
  ok=0; [ "$CUDA_TOOLKIT_VISIBLE" = "false" ] || ok=1
  check "no CUDA toolkit visible to the product (runtime-only environment)" $ok "nvcc stripped from PATH, LD_LIBRARY_PATH unset"
  BACKEND_SO="$(find "$GREPPY_STORE_DIR/runtime/cuda" -name 'greppy-cuda-backend.so' 2>/dev/null | head -1)"
  ok=0; [ -n "$BACKEND_SO" ] || ok=1
  check "bundled CUDA backend materialized from the binary" $ok
  if [ -n "$BACKEND_SO" ] && command -v ldd >/dev/null 2>&1; then
    NEEDED="$(ldd "$BACKEND_SO" | awk '{print $1}' | grep -v '^/' | LC_ALL=C sort)"
    CUDA_NEEDED_JSON="$(printf '%s\n' "$NEEDED" | jq -R . | jq -cs .)"
    ok=0
    if printf '%s\n' "$NEEDED" | grep -Eq 'cudart|cublas|nvrtc'; then ok=1; fi
    check "CUDA backend needs no toolkit libraries (driver's libcuda.so.1 only)" $ok "needed: $(printf '%s' "$NEEDED" | tr '\n' ' ')"
  fi
fi

# --- drain daemons, stop VRAM polling ---------------------------------------------
section "drain"
drain_daemons
VRAM_JSON="null"
if [ "$BACKEND" = "cuda" ]; then
  kill "$VRAM_POLL_PID" 2>/dev/null || true
  wait "$VRAM_POLL_PID" 2>/dev/null || true
  VRAM_POLL_PID=""
  VRAM_PEAK="$(LC_ALL=C sort -n "$VRAM_SAMPLES" | tail -1 | tr -d ' ')"
  [ -n "$VRAM_PEAK" ] || fail "VRAM poller collected no samples"
  VRAM_JSON="$(jq -cn --argjson base "$VRAM_BASELINE" --argjson peak "$VRAM_PEAK" --argjson iv "$VRAM_POLL_INTERVAL" \
    '{baseline_mib: $base, peak_mib: $peak, poll_interval_s: $iv}')"
  echo "vram: baseline ${VRAM_BASELINE}MiB, peak ${VRAM_PEAK}MiB"
fi
rm -rf "$RUNTIME_BASE"

# --- notes ------------------------------------------------------------------------
NOTES="$WORK/notes.txt"
: >"$NOTES"
if [ "$BACKEND" = "cuda" ]; then
  echo "GPU may be shared with other tenants; VRAM peak-baseline bounds the product footprint." >>"$NOTES"
fi
if [ "$BACKEND" = "metal" ]; then
  if [ "$METAL_TENSOR" = "true" ]; then
    echo "Metal tensor-ops (Metal 4) matmul path exercised on this GPU generation (family: ${METAL_FAMILY:-unknown})." >>"$NOTES"
  else
    echo "Metal tensor-ops (Metal 4 / M5-class GPU) matmul path NOT exercised on this machine (family: ${METAL_FAMILY:-unknown}); simdgroup fallback measured instead, tensor path pending newer hardware." >>"$NOTES"
  fi
fi
if [ "$REQUIRE_BASELINE_X86" -eq 1 ]; then
  echo "Run asserts SIMD baseline fallbacks: CPU verified to lack AVX-VNNI/AVX-512." >>"$NOTES"
fi
NOTES_JSON="$(jq -R . <"$NOTES" | jq -cs .)"

# --- artifact ----------------------------------------------------------------------
section "artifact"

GPU_JSON="null"
if [ "$BACKEND" = "cuda" ]; then
  GPU_JSON="$(jq -cn --arg name "$GPU_NAME" --arg driver "$GPU_DRIVER" --argjson vram "$GPU_VRAM_MIB" \
    --arg cc "$CUDA_COMPUTE_CAP" \
    '{name: $name, driver_version: $driver, vram_total_mib: $vram,
      compute_capability: (if $cc == "" then null else $cc end), metal_family: null}')"
elif [ "$BACKEND" = "metal" ]; then
  GPU_JSON="$(jq -cn --arg name "$CPU_MODEL" --arg os "$OS_VERSION" --arg family "$METAL_FAMILY" \
    '{name: ($name + " GPU"), driver_version: ("macOS " + $os), vram_total_mib: 0,
      compute_capability: null, metal_family: (if $family == "" then null else $family end)}')"
fi

CHECKS_JSON="$(jq -Rn '[inputs | split("\t") |
  {name: .[0], pass: (.[1] == "true")} + (if .[2] != "" then {detail: .[2]} else {} end)]' <"$CHECKS")"

OUT="${OUT:-hardware-evidence-$OS-$ARCH-$BACKEND.json}"
mkdir -p "$(dirname "$OUT")"
jq -n \
  --arg created "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg os "$OS" --arg os_version "$OS_VERSION" --arg arch "$ARCH" \
  --arg cpu_model "$CPU_MODEL" --argjson cpu_cores "$CPU_CORES" --argjson ram_gb "$RAM_GB" \
  --arg apple_gen "$APPLE_GEN" \
  --argjson gpu "$GPU_JSON" \
  --arg version "$VERSION" --arg source_sha "$SOURCE_SHA" --arg binary_sha "$BINARY_SHA" \
  --argjson backend_compiled "$BACKEND_COMPILED" --arg backend_build_info "$BACKEND_BUILD_INFO" \
  --arg requested "$BACKEND" --arg selected "$SELECTED_BACKEND" --arg selected_device "$SELECTED_DEVICE" \
  --argjson cpu_caps "$CPU_CAPS_JSON" \
  --argjson baseline_x86 "$([ "$REQUIRE_BASELINE_X86" -eq 1 ] && echo true || echo false)" \
  --argjson toolkit_visible "$CUDA_TOOLKIT_VISIBLE" \
  --argjson cuda_needed "$CUDA_NEEDED_JSON" \
  --argjson models "$MODELS_JSON" \
  --argjson fixture_files "$FIXTURE_FILES" --arg fixture_digest "$FIXTURE_DIGEST" \
  --argjson index_ms "$INDEX_MS" \
  --argjson search "$SEARCH_JSON" --argjson brief "$BRIEF_JSON" \
  --argjson vram "$VRAM_JSON" --argjson notes "$NOTES_JSON" \
  --argjson checks "$CHECKS_JSON" --arg status "$OVERALL" \
  '{
    schema_version: "greppy.hardware-evidence.v1",
    created_utc: $created,
    platform: ({
      os: $os, os_version: $os_version, arch: $arch,
      cpu_model: $cpu_model, cpu_cores: $cpu_cores, ram_gb: $ram_gb
    } + (if $apple_gen != "" then {apple_silicon_generation: $apple_gen} else {} end)
      + {gpu: $gpu}),
    product: {
      version: $version, source_sha: $source_sha, binary_sha256: $binary_sha,
      backend_compiled: $backend_compiled, backend_build_info: $backend_build_info
    },
    backend: {
      requested: $requested, selected: $selected,
      selected_device: (if $selected_device == "" then null else $selected_device end),
      cpu_capabilities: $cpu_caps,
      baseline_x86_required: $baseline_x86,
      cuda_toolkit_visible: $toolkit_visible,
      cuda_backend_needed_libs: $cuda_needed
    },
    models: $models,
    measurements: {
      fixture: {files: $fixture_files, tree_sha256: $fixture_digest},
      index_cold_ms: $index_ms,
      semantic_search_ms: $search,
      brief_ms: $brief,
      vram: $vram,
      notes: $notes
    },
    checks: $checks,
    status: $status
  }' >"$OUT"

# --- scrub gate ---------------------------------------------------------------------
# The artifact is public: refuse to emit anything containing the local
# hostname or username. (Values recorded above cannot contain them by
# construction; this is the belt-and-braces gate.)
for secret in "$(hostname 2>/dev/null | cut -d. -f1)" "$(id -un)"; do
  [ -n "$secret" ] && [ ${#secret} -ge 3 ] || continue
  if grep -Fiq -- "$secret" "$OUT"; then
    rm -f "$OUT"
    fail "artifact contained host-identifying string; refused to emit"
  fi
done

echo "evidence written: $OUT"
[ "$OVERALL" = "pass" ] || fail "one or more contract checks failed (status: fail recorded in $OUT)"
printf '\nhardware evidence run passed: %s (%s)\n' "$BIN" "$BACKEND"
