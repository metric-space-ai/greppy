# Docker network isolation for productive V3 agents

gpu3 uses Docker for the agent boundary. `bwrap` and `socat` are not assumed.
The productive topology has two user-defined networks:

```text
agent container ── agent-internal (--internal, no DNS) ── CONNECT proxy
                                                            │
                                             proxy-egress ───┘
```

The agent joins only `agent-internal`. The proxy is the only dual-homed
container. Its image contains an immutable CONNECT policy allowing exactly the
host and port found in the frozen MiniMax provider source. Every other CONNECT,
including GitHub, is denied. The internal network has no direct external route,
and agents run with `--dns 127.0.0.1`; the proxy address is injected as an IP,
so the agent neither needs nor receives general DNS.

## Preregistration

Copy [`network_policy.example.json`](network_policy.example.json), pin both
container manifest digests and local image IDs, and change `status` to `sealed`.
The policy binds:

- Docker server minimum 27.2;
- exact internal/egress network and proxy container names;
- proxy and agent-audit image digests/IDs;
- the SHA-256 and extracted HTTPS hosts of `minimax-provider.js`;
- canonical CONNECT allowlist SHA-256;
- positive provider and negative GitHub/DNS/direct-socket probes;
- forbidden NAS and builder roots.

The proxy policy must be baked into its pinned image at the preregistered
`proxy_policy_path` as canonical compact JSON with no trailing newline (for the
example: `{"allow_connect":[{"host":"api.minimax.io","port":443}]}`). The
running proxy carries these labels and has no host mounts:

```text
dev.greppy.v3.role=allowlist-connect-proxy
dev.greppy.v3.proxy-policy-sha256=<canonical allowlist SHA-256>
```

Create the agent network with Docker's `--internal` flag and create a separate
ordinary bridge for proxy egress. Start the pinned proxy on both and attach no
other container to the egress network. A productive agent is launched on the
internal network only, with all capabilities dropped, `no-new-privileges`, a
read-only container root, and DNS set to `127.0.0.1`. `HTTPS_PROXY` points to
the proxy's internal IP; `NO_PROXY` is empty.

Only the task worktree, its per-arm Greppy store, and explicit read-only
dependency caches may be mounted into a productive agent. Never mount:

- `GREPPY_BENCH_NAS_ROOT` or any child;
- the NVMe mirror/builder clone tree;
- sealed tests, gold, manifests, or evidence staging;
- the Docker socket.

The MiniMax key is injected only into the productive agent process. It is not
passed to the proxy, audit container, Docker labels, argv, or reports. The
network audit performs an unauthenticated TLS/HTTP reachability probe; a frozen
400/401/403/404/405 response proves the route without logging or transmitting a
credential.

## Mandatory audit

Run this after the proxy and networks exist, before the three-trajectory smoke,
and again before every resumed full-run shard:

```bash
python3 bench/agent_coding/v3/audit_network_isolation.py \
  --config /sealed/config/network_policy.json \
  --report "$GREPPY_BENCH_NVME_ROOT/network-audit.json"
```

The audit launches a disposable, mount-free container from the pinned audit
image on the internal network and proves all of the following together:

1. MiniMax is reachable only through the CONNECT proxy.
2. `https://github.com/` is denied by that proxy.
3. `github.com` cannot be resolved in the agent namespace.
4. direct TCP sockets to preregistered public targets fail.
5. the proxy is dual-homed on exactly the two registered networks, while the
   probe container has only the internal network.
6. proxy image, labels, policy hash and zero-mount state match preregistration.

Success exits `0`. Every policy, Docker, topology or probe failure exits `2`
and still emits JSON. The report contains no environment or headers.
`audit_evidence.proof_sha256` binds the canonical evidence object without that
field. Its schema is exactly `greppy.provider-only-egress.v1`, the attestation
consumed by `runner.py`; there is no second summary schema. Store the resulting
report hash in the smoke/full-run manifest. The runner recomputes this proof and
re-inspects the live internal/egress network IDs plus proxy container/image/IP
identity immediately before execution. Replacing a network or proxy after the
audit therefore fails closed.

The attestation is valid only for the same pinned agent image, internal network,
proxy endpoint and DNS flags used by the productive runner. Do not run a probe
container with stronger flags than the real agent and reuse its report. Until
those identities match and the actual positive/negative probes pass, the
example remains `template-not-sealed` and cannot produce release evidence.
