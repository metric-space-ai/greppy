# SQLite compatibility patches

The pinned Servo 0.5.0 dependency API is retained. These two registry crates
are copied verbatim except that their normalized Cargo.toml rusqlite
requirement is raised from 0.38 to 0.41 (also reflected in the SQL adapter's
Cargo.toml.orig). No Rust source is changed. Their standalone upstream
Cargo.lock files are omitted; the resolved graph is in crates/web-runtime/Cargo.lock.

- servo-storage 0.5.0: archive SHA-256
  96803839259126fc8bd3f7c9e18b25c681c1783c8521178914c380f6753f3d0a;
  Servo source 77fccacc1f1fdce10498d50173aafaa09d02879e, components/storage.
  MPL-2.0; LICENSE and the source notices are retained.
- sea-query-rusqlite 0.8.0: archive SHA-256
  1ec6038023c8517c623e5bf9606b3c54d40bc8296bb6b2986040428dd84deddd;
  SeaQL/sea-query source f4f75ea470ce25572f3d3c856d6ebf1b10d6f62f,
  sea-query-rusqlite. MIT OR Apache-2.0; LICENSE-MIT is retained.

The regular bundled feature is preserved. Both workspace lockfiles resolve
libsqlite3-sys 0.39.1, which bundles SQLite 3.54.1 and fixes
[RUSTSEC-2026-0459](https://github.com/RustSec/advisory-db/blob/main/crates/libsqlite3-sys/RUSTSEC-2026-0459.md).
This does not claim to patch SQLCipher; bundled-sqlcipher is not enabled.

Remove these path patches when the compatible published Servo storage and SQL
adapter dependencies both accept the patched rusqlite/libsqlite3-sys line.
