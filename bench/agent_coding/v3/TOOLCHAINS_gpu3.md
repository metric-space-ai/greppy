# gpu3 toolchains for the v3 adapters

Phase 2 requires all 24 adapters to build and test for real. Nine toolchain
profiles are declared in `repository_registry.json`; six of them did not exist
on gpu3 when the plan was written, which would have shown up as "Java, C++ and
Ruby quietly dropped" — the exact failure the plan's Phase 2 acceptance
forbids.

Measured 2026-07-31, then installed. Everything is USER-LEVEL (no sudo), under
`~/opt` or the tool's own manager, so it can be reproduced and pinned into the
digest-pinned images.

| Profile | Repos | State before | Installed as |
|---|---:|---|---|
| rust-cargo | 3 | present | `~/.cargo/bin` (rustup; `/usr/bin/cargo` is too old — "lock file version 4") |
| python-pip | 3 | present | system python3 3.10 |
| cpp-cmake | 3 | present | system cmake + ninja + g++ |
| go-test | 3 | **missing** | `~/opt/go` (go1.23.4) |
| ts-pnpm | 2 | **missing** | pnpm 11.18 via npm under node 22 |
| javascript-node | 4 | **missing** | nvm node v22.23.1 (system node is v12 — too old for modern tooling) |
| java-maven | 1 | **missing** | `~/opt/maven` (3.9.11, from archive.apache.org — dlcdn only serves current) |
| java-gradle | 2 | **missing** | `~/opt/gradle` (8.10.2) |
| ruby-bundler | 3 | **missing** | rbenv + ruby-build, ruby 3.3.6, bundler |

One PATH line that satisfies every profile:

```bash
export PATH=$HOME/.nvm/versions/node/v22.23.1/bin:$HOME/opt/go/bin:$HOME/opt/maven/bin:$HOME/opt/gradle/bin:$HOME/.rbenv/shims:$HOME/.rbenv/bin:$HOME/.cargo/bin:$PATH
```

Traps that cost time here, so they do not cost it again:

- `nvm install` refuses while `~/.npmrc` carries a `prefix=` line; delete that
  line first (the old value was `~/opt/npm-global`).
- The `pi` shim resolves `node` from PATH, so with the system node v12 first it
  dies on optional chaining — node 22 must precede `/usr/bin`.
- Maven is not on `dlcdn.apache.org` once a release ages out; use
  `archive.apache.org/dist/maven/maven-3/<version>/binaries/`.
- Java itself is present (openjdk 11). Repositories needing 17/21 will need a
  JDK per adapter — check before declaring an adapter `ready`.

**These installs live in a home directory, not in an image.** Phase 2 requires
them pinned into digest-pinned images with offline dependency caches; this file
is the inventory that build must reproduce, not a substitute for it.
