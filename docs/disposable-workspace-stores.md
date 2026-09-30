# Disposable workspace stores on macOS

With no `GREPPY_STORE_DIR` override, a workspace whose path or canonical identity
is under `/Volumes/tmp` puts a **new** workspace store under
`/Volumes/tmp/dev-artifacts/greppy/workspace-stores/v2/<workspace-hash>`.
Graph data, workspace sidecars and edit journals use the same store directory.
Canonical durable repositories, explicit store overrides, shared model caches,
immutable Base caches and existing workspace lock identities retain their roots.

The tmp volume must exist as a mounted filesystem. Store creation rejects a
missing or unmounted volume, a workspace that resolves outside that volume,
and symlinked disposable cache namespace components. It does not create a
replacement `/Volumes/tmp` directory on the system disk.

An existing versioned or legacy workspace cache remains authoritative in its
original directory. This includes incomplete caches and all sidecars; selection
does not move, delete, snapshot or reindex data while another client may use it.
An explicit `GREPPY_STORE_DIR` continues to select its existing namespace.

**Remaining migration gap:** retained system-disk stores have not been relocated,
and the storage policy is not satisfied for those entries. This change deliberately
adds no automatic migration command: old clients can select the original path
before acquiring a lifecycle lock, so a lock held only during a directory move
cannot establish cross-version safety. A later coordinated migration must account
for all old-client ownership, database sidecars and CoW references, and verify
query reuse and recovery before releasing the original cache. Ordinary queries
continue to reuse retained stores while that migration remains pending.
