# grepplus bench fixture

This fixture is a hand-crafted Rust project that the Phase 7
benchmarks index and grep against. It contains:

- `src/lib.rs` — `Greeter`, `ProcessOrder`, `UserService`, `hello`
- `src/greeter.rs` — secondary module
- `src/orders.rs` — order-handling helpers
- `src/script.py` — a Python file that exercises the
  `Language::Unsupported` path (the file is discovered but not parsed)

The fixture's git state is intentionally committed at the start of
the bench run; each scenario mutates the working tree and is reset
between scenarios by `bench/freshness_bench.sh::reset_repo` via
`git clean -fdx` + `git checkout -- .`.
