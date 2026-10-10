# Security Review: Greppy 0.4.0

Reviewed on 2026-09-03 at commit `8931ade3` (`codex/release-0.4.0-int-build`).
The workspace package version is `0.4.0` (`Cargo.toml:25`). The review covered
the Rust workspaces, the embedded web runtime, its vendored Servo networking
code, dependency advisories, agent sandboxing, local IPC/capabilities, path
policies, edit integrity, and the HTTP/HTTPS policy proxy.

## Executive summary

No critical or high-severity issue was found. Three medium-severity issues were
confirmed:

1. the web runtime contains `rsa 0.10.0-rc.18`, affected by the Marvin
   key-recovery timing side channel (RUSTSEC-2023-0071);
2. the vendored Servo HSTS preload list is globally disabled, permitting a
   first-visit HTTP downgrade for every preloaded domain; and
3. the policy proxy creates unbounded OS threads and has no active-connection
   limit, allowing hostile page content to exhaust local process resources.

The main workspace lockfile has no known RustSec vulnerability. The separate
web-runtime lockfile has the RSA vulnerability plus seven unmaintained crates
and one yanked crate.

## Remediation verification

Follow-up verification on 2026-09-03 at release commit `6d69a6ca` (including
the security fixes from `7072f1a7`) confirmed that all three medium findings
are closed or fail-closed mitigated:

- **SEC-001 mitigated:**
  `crates/web-runtime/runtime/src/content_worker.rs:4059` disables
  page-accessible `SubtleCrypto`, while the regression at line 4405 requires
  that preference to remain false. `rsa 0.10.0-rc.18` remains locked, so
  RUSTSEC-2023-0071 is a temporary capability-disabled CI exception rather
  than a dependency upgrade; the removal condition is documented in
  `SECURITY.md:127-136`.
- **SEC-002 fixed:** the unconditional false return was removed from
  `crates/web-runtime/vendor/servo-net/hsts.rs`. The regular workspace test at
  `crates/web-runtime/runtime/tests/hsts-preload.rs:5-16` now verifies exact
  domains and include-subdomain semantics.
- **SEC-003 fixed:**
  `crates/web-runtime/runtime/src/policy_proxy.rs:13-15,23-39` enforces 32
  active connections per proxy and 128 process-wide with RAII accounting;
  `accept_loop` at lines 177-205 rejects excess connections with HTTP 503
  before allocating an OS thread.

Independent targeted verification passed: HSTS preload 1/1, bounded proxy
connections 1/1, engine security preferences 1/1, and both Cargo lockfile
audits exited zero under the documented exceptions. No critical, high, or
open medium finding remains at `6d69a6ca`; dependency-maintenance warnings
below remain informational.

## Medium severity

### SEC-001: WebCrypto RSA decryption uses a crate affected by the Marvin timing attack

**Impact:** An attacker who can submit many chosen RSA ciphertexts and observe
decryption timing may recover an RSA private key used by WebCrypto in the
embedded browser.

Evidence:

- `crates/web-runtime/Cargo.lock:5360-5363` pins `rsa 0.10.0-rc.18`.
- `cargo tree --manifest-path crates/web-runtime/Cargo.toml -i rsa` proves the
  production chain `rsa -> servo-script -> servo -> web-runtime`.
- The compiled `servo-script 0.5.0` implementation exposes RSA-OAEP decryption:
  `dom/webcrypto/subtlecrypto/rsa_oaep_operation.rs:92-149` constructs
  `DecryptingKey` and calls `decrypt(ciphertext)` for supported hashes;
  `subtlecrypto.rs:4957-4961` dispatches WebCrypto RSA-OAEP to that function.
- `cargo audit --no-fetch --file crates/web-runtime/Cargo.lock` reports
  RUSTSEC-2023-0071, CVSS 5.9 (medium), with no fixed release. The official
  advisory states that non-constant-time behavior can leak enough timing
  information over a network to recover a private key:
  <https://rustsec.org/advisories/RUSTSEC-2023-0071>.

Preconditions and limits:

- Exploitation requires an RSA private-key operation on attacker-controlled
  ciphertexts plus a timing oracle. Purely local use on an uncompromised host
  is outside the advisory's exploitable scenario.
- The issue is reachable through the browser's WebCrypto RSA-OAEP path; it is
  not merely an unused transitive package.

Recommendation:

- Do not ship RSA private-key decryption through this crate while no
  constant-time fix exists. Disable RSA-OAEP WebCrypto in the runtime, or patch
  Servo to use a reviewed constant-time backend.
- Add an advisory gate for both lockfiles in release CI. Treat the web-runtime
  `cargo audit` failure as release-blocking unless an explicit, documented
  reachability exception is approved.

### SEC-002: HSTS preload enforcement is disabled for every domain

**Impact:** On the first visit to a preloaded HSTS domain, an on-path attacker
can keep an explicit `http://` navigation on plaintext HTTP and alter or read
traffic that should have been upgraded locally before any request left the
machine.

Evidence:

- `crates/web-runtime/vendor/servo-net/hsts.rs:123-150` makes
  `HstsPreloadList::is_host_secure` unconditionally return `false`; the real
  domain and include-subdomain checks are unreachable.
- `crates/web-runtime/vendor/servo-net/hsts.rs:155-168` consults that preload
  list before falling back to dynamically learned HSTS entries.
- `crates/web-runtime/vendor/servo-net/hsts.rs:208-241` upgrades `http`/`ws`
  only when the host is considered secure.
- `crates/web-runtime/runtime/src/content_worker.rs:4047-4054` explicitly sets
  `network_enforce_tls_enabled = false`, so there is no global HTTPS-only mode
  masking the disabled preload check.
- The vendored regression at
  `crates/web-runtime/vendor/servo-net/tests/hsts.rs:297-308` expects parsed
  preload entries to return `true`, but `servo-net` is excluded from the web
  workspace and this test is not part of the 128-test runtime unit suite.
- A temporary isolated library-test copy of that exact preload assertion was
  run against the vendored crate. It failed immediately at
  `preload.is_host_secure("example.com")`; the same build warned that the
  lookup after `return false` is unreachable. The direct vendored integration
  suite could not serve as a gate because unrelated stale test APIs fail to
  compile. All temporary test/workspace/lockfile changes were removed after
  reproduction.

The code comment disables all preloads to accommodate one host
(`neverssl.com`). That trades a single compatibility exception for a global
security bypass. Chromium documents that preloading closes the fresh-profile
window by shipping HSTS state with the browser:
<https://chromium.googlesource.com/playground/chromium-org-site/+/refs/heads/main/hsts/index.md>.
RFC 6797 also recognizes preloaded Known HSTS Host lists as a user-agent
mechanism: <https://www.rfc-editor.org/rfc/rfc6797>.

Recommendation:

- Remove the unconditional return and restore the normal preload lookup.
- If a compatibility exception is genuinely required, scope it narrowly to a
  reviewed hostname or correct the preload dataset; do not disable every
  entry.
- Run the vendored HSTS tests in CI or copy a focused preload-upgrade
  regression into the web-runtime workspace. Include a first-navigation test
  that proves no plaintext request is emitted for a known preloaded domain.

### SEC-003: Policy proxy has unbounded connection and thread creation

**Impact:** A hostile page can open many HTTP(S)/WebSocket connections and
exhaust file descriptors, thread stacks, memory, or scheduler capacity in the
content worker, causing denial of service to Greppy's web runtime.

Evidence:

- `crates/web-runtime/runtime/src/content_worker.rs:884-886` creates one
  `PolicyProxy` and configures the browser engine to use it.
- `crates/web-runtime/runtime/src/content_worker.rs:4047-4053` forces both HTTP
  and HTTPS through that proxy with no no-proxy bypass.
- `crates/web-runtime/runtime/src/policy_proxy.rs:116-129` spawns a new OS
  thread for every accepted TCP connection and has no semaphore, connection
  counter, queue bound, or rejection threshold.
- `crates/web-runtime/runtime/src/policy_proxy.rs:383-392` creates another OS
  thread for one direction of every CONNECT tunnel; long-lived tunnels have
  no lifetime or idle bound in this layer.
- `crates/web-runtime/runtime/src/limits.rs:8-22,88-116,191-215` defines byte,
  request, wall-time, and RSS limits but no active-connection/thread limit.
- `crates/web-runtime/runtime/src/daemon.rs:4403-4411` charges requests only for
  top-level navigation operations, not page-created subresources or sockets.
  The proxy byte counter is sampled after operations; it does not stop a
  connection while bytes or threads accumulate.
- A bounded macOS unit-test PoC opened 48 loopback TCP connections to a local
  `PolicyProxy` and intentionally sent no request bytes. After 500 ms, the same
  test process had grown from 3 to 51 OS threads: exactly one additional
  `greppy-policy-proxy-conn` thread per idle connection. The PoC passed in
  0.53 seconds and was removed after measurement; no product source change
  remains.

This is source- and runtime-confirmed without driving the machine toward
resource exhaustion. The normal threat path is remote content loaded by the
browser, so no local shell access is required after the user/agent navigates
to hostile content.

Recommendation:

- Replace thread-per-connection with bounded async I/O or a fixed worker pool.
- Enforce a small global and per-session active-connection limit before
  spawning work; reject excess connections without allocating a thread.
- Tie tunnel lifetime, idle timeout, cancellation, and byte accounting to the
  owning session. Enforce byte limits during relay, not only after a browser
  operation returns.
- Add a regression that opens more than the limit, proves bounded thread/FD
  growth, and verifies recovery after connections close.

## Informational dependency hygiene

`cargo audit --no-fetch --file crates/web-runtime/Cargo.lock` also reports:

- unmaintained: `bincode 1.3.3` (RUSTSEC-2025-0141), `paste 1.0.15`
  (RUSTSEC-2024-0436), and five `unic-* 0.9.0` crates
  (RUSTSEC-2025-0075/0080/0081/0098/0100);
- yanked: `chacha20 0.10.1`.

The main lockfile reports only the unmaintained `paste 1.0.15` warning and no
known vulnerability. These warnings are not treated as vulnerabilities by
themselves, but the web runtime's 886-package dependency surface needs an
explicit update/exception policy.

## Checks performed

- `cargo audit` main `Cargo.lock`: 408 dependencies, zero vulnerabilities,
  one allowed unmaintained warning.
- `cargo audit` web-runtime `Cargo.lock`: 886 dependencies, one vulnerability,
  eight warnings.
- Agent sandbox tests: 28/28 passed, including symlink swaps, outside-root
  writes, temporary-directory sibling escapes, Seatbelt, and Landlock paths.
- Web policy-proxy tests: 9/9 passed, including metadata IP/hostname blocking,
  redirects, keep-alive DNS rebinding, and HTTPS CONNECT rebinding.
- Full web-runtime unit tests: 128/128 passed, including capability FD leakage,
  worker handshake, image identity, and artifact digest fail-closed checks.
- Edit engine unit tests: 84/84 passed.
- Workspace path-policy tests: 2/2 passed.
- CLI hardening suite on the reviewed commit: 39/40 passed; the sole failure
  was a non-security first-use latency contract (10.2-13.2 seconds vs. <10).
  It was reported separately and is fixed on `origin/codex/release-0.4.0` in
  commit `946a5f23f28e81545c4fb9be96f51be31ad25559`.

## Positive controls observed

- Agent execution defaults to enforced sandboxing at the CLI boundary.
- Unix runtime directories validate owner/type and are forced to mode `0700`;
  IPC sockets are mode `0600` and requests require constant-time capability
  matching.
- Attach capabilities are passed through inherited close-on-exec file
  descriptors rather than argv, with leakage regression coverage.
- The network policy re-resolves at connect time and filters literal and
  resolved IP addresses, including cloud metadata and non-public destinations
  in the research profile.
- Install/upgrade scripts verify payload SHA-256 manifests and contain explicit
  symlink/destination guards.

## Review limitations

This was a source, dependency, and targeted regression review, not a full
memory-safety audit of Servo/V8/SpiderMonkey or a fuzzing campaign. The active
Greppy semantic index was refreshing during the review; security-critical
source evidence was therefore read from live files with `greppy read-file` or
`greppy rg`, and stale-index false negatives were reported separately to the
requested Codex task.
