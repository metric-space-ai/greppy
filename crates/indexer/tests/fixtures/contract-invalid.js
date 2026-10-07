// Synthetic residual provider-contract failure for JS/TS usage repair.
// microsoft/typescript-go (Apache-2.0) at dbd8d7f28c4ab9783d10f5ff3e09f9084ad48086
// is the known repository repro. Its bench failure was "dropped != 0" on the
// pre-degrade repair; after validate_or_degrade, real extracts are cured
// (invalid spans and blank identities are dropped, and file identity /
// confidence are rewritten). This file is the stand-in the repair test skips
// via the test-only residual-contract arm for this filename.
function skipped() {
    return 1;
}
