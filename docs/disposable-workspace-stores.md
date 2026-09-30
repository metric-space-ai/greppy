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
original directory when no disposable store has been established. Once a disposable
store has been created, later durable directories cannot redirect its selection.
Existing disposable stores require a matching ownership manifest; nonempty unowned
stores are rejected without adopting or deleting their bytes. Retained original
caches keep their complete sidecars, including incomplete caches. Store selection
does not relocate or rebuild data while another client may use it.
An explicit `GREPPY_STORE_DIR` continues to select its existing namespace.

Initializing graph, pack, cache and journal writers consume the authoritative
directory returned by store creation before locking or opening a path. A read-only
locator lookup is only a candidate: an old client can create a retained durable
store between that lookup and initialization.

Cache inventory, status, clear and GC include both validated store namespaces
when no explicit store override is supplied. Disposable inventory requires the tmp
volume to be mounted and every namespace component to be a directory without
symlinks. Only matching manifests whose source identity is under the disposable
volume are managed there. GC respects both lifecycle and writer leases, and moves
disposable entries to trash on that same volume; invalid entries remain unmanaged.

**Remaining migration gap:** retained system-disk stores have not been relocated,
and the storage policy is not satisfied for those entries. This change deliberately
adds no automatic migration command: old clients can select the original path
before acquiring a lifecycle lock, so a lock held only during a directory move
cannot establish cross-version safety. A later coordinated migration must account
for all old-client ownership, database sidecars and CoW references, and verify
query reuse and recovery before releasing the original cache. Ordinary queries
continue to reuse retained stores while that migration remains pending.
