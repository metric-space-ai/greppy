//! The edit transaction core: snapshot, overlap rejection, in-memory apply,
//! reparse, changed-range accounting.
//!
//! All byte ranges are computed against one immutable snapshot; multiple
//! operations on a file are applied from the highest to the lowest offset so
//! earlier applications never shift later targets. Overlapping ranges are
//! rejected before anything is applied.

use std::path::{Path, PathBuf};

use crate::hash::sha256_hex;
use greppy_core::{Error, Result};
use greppy_parser::Language;

/// One planned mutation of a byte range within a snapshot.
#[derive(Debug, Clone)]
pub struct PlannedOp {
    pub id: String,
    pub range: (usize, usize),
    pub replacement: Vec<u8>,
}

/// An immutable view of one file at plan time.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub path: PathBuf,
    pub content: Vec<u8>,
    pub file_sha256: String,
}

impl Snapshot {
    pub fn read(path: &Path) -> Result<Self> {
        let meta = std::fs::symlink_metadata(path).map_err(|source| Error::Io {
            context: format!("stat {}", path.display()),
            source,
        })?;
        if meta.file_type().is_symlink() {
            return Err(Error::Workspace(format!(
                "refusing to edit through symlink: {}",
                path.display()
            )));
        }
        let content = std::fs::read(path).map_err(|source| Error::Io {
            context: format!("read {}", path.display()),
            source,
        })?;
        let file_sha256 = sha256_hex(&content);
        Ok(Self {
            path: path.to_path_buf(),
            content,
            file_sha256,
        })
    }
}

/// Result of applying planned operations in memory.
#[derive(Debug)]
pub struct Applied {
    pub content: Vec<u8>,
    pub file_sha256: String,
    /// Ranges (in ORIGINAL coordinates) that were replaced.
    pub changed_ranges: Vec<(usize, usize)>,
}

/// Reject overlaps, then apply high→low against the snapshot.
pub fn apply_in_memory(snapshot: &Snapshot, ops: &[PlannedOp]) -> Result<Applied> {
    for op in ops {
        let (start, end) = op.range;
        if start > end || end > snapshot.content.len() {
            return Err(Error::Invalid(format!(
                "operation {}: range {start}..{end} outside file of {} bytes",
                op.id,
                snapshot.content.len()
            )));
        }
    }
    let mut sorted: Vec<&PlannedOp> = ops.iter().collect();
    sorted.sort_by_key(|op| op.range.0);
    for pair in sorted.windows(2) {
        // insertions (empty ranges) at the same position also conflict: their
        // relative order against the same snapshot would be ambiguous
        if pair[1].range.0 < pair[0].range.1
            || (pair[0].range == pair[1].range && pair[0].range.0 == pair[0].range.1)
        {
            return Err(Error::Invalid(format!(
                "operations {} and {} overlap ({}..{} vs {}..{}); nothing was changed",
                pair[0].id,
                pair[1].id,
                pair[0].range.0,
                pair[0].range.1,
                pair[1].range.0,
                pair[1].range.1
            )));
        }
    }
    let mut content = snapshot.content.clone();
    for op in sorted.iter().rev() {
        content.splice(op.range.0..op.range.1, op.replacement.iter().copied());
    }
    let changed_ranges = sorted.iter().map(|op| op.range).collect();
    let file_sha256 = sha256_hex(&content);
    Ok(Applied {
        content,
        file_sha256,
        changed_ranges,
    })
}

/// Verify that every byte outside the declared original ranges maps
/// unchanged into the result (accounting for length deltas of the edits).
pub fn outside_ranges_unchanged(before: &[u8], after: &[u8], ops: &[PlannedOp]) -> bool {
    let mut sorted: Vec<&PlannedOp> = ops.iter().collect();
    sorted.sort_by_key(|op| op.range.0);
    let mut b = 0usize; // cursor in before
    let mut a = 0usize; // cursor in after
    for op in &sorted {
        let (start, end) = op.range;
        if before.get(b..start) != after.get(a..a + (start - b)) {
            return false;
        }
        a += start - b + op.replacement.len();
        b = end;
    }
    before.get(b..) == after.get(a..)
}

/// ERROR/MISSING counts of a parse tree, for syntax postconditions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyntaxCounts {
    pub errors: usize,
    pub missing: usize,
}

/// The kinds of the ancestor chain (parent -> root, leaf excluded) of the
/// smallest node covering `range`. This is the structural CONTEXT the edited
/// bytes live in.
///
/// Counting ERROR/MISSING nodes alone is not a sufficient syntax gate:
/// tree-sitter's error recovery silently reinterprets many malformations
/// without emitting ERROR nodes (proven 2026-07-17: replacing a Go method
/// body with a whole file's text — copyright header, package decl, imports —
/// yielded new_errors=0 while gofmt rejected the file, so the certificate
/// falsely reported `syntax: proved`). A structural edit must not change the
/// context its target sits in: a body stays inside its function, a function
/// stays a top-level declaration. When the surrounding context's kind chain
/// changes, the edit broke the grammar in a way tree-sitter recovered past.
fn context_kinds(language: Language, content: &[u8], range: (usize, usize)) -> Option<Vec<String>> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let leaf = tree
        .root_node()
        .descendant_for_byte_range(range.0, range.1.saturating_sub(1).max(range.0))?;
    let mut kinds = Vec::new();
    let mut node = leaf.parent();
    while let Some(cur) = node {
        kinds.push(cur.kind().to_string());
        node = cur.parent();
    }
    Some(kinds)
}

/// Does the edited region still sit in the same structural context after the
/// edit? `before_range` is the target in the pre-edit content; `after_range`
/// is the changed span in the post-edit content. Returns true (permissive)
/// when either side cannot be parsed — that path is covered by the
/// ERROR/MISSING count, which reports not-applicable.
pub fn structural_context_preserved(
    language: Language,
    before: &[u8],
    before_range: (usize, usize),
    after: &[u8],
    after_range: (usize, usize),
) -> bool {
    match (
        context_kinds(language, before, before_range),
        context_kinds(language, after, after_range),
    ) {
        (Some(b), Some(a)) => a == b,
        _ => true,
    }
}

/// Parse `content` and count ERROR and MISSING nodes. `None` when the
/// language is not tree-sitter-supported (postcondition then reports
/// not-applicable rather than silently passing).
pub fn syntax_counts(language: Language, content: &[u8]) -> Option<SyntaxCounts> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let mut errors = 0usize;
    let mut missing = 0usize;
    let mut cursor = tree.walk();
    let mut reached_root = false;
    while !reached_root {
        let node = cursor.node();
        if node.is_error() {
            errors += 1;
        }
        if node.is_missing() {
            missing += 1;
        }
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                reached_root = true;
                break;
            }
        }
    }
    Some(SyntaxCounts { errors, missing })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(content: &[u8]) -> Snapshot {
        Snapshot {
            path: PathBuf::from("mem"),
            content: content.to_vec(),
            file_sha256: sha256_hex(content),
        }
    }

    fn op(id: &str, range: (usize, usize), replacement: &[u8]) -> PlannedOp {
        PlannedOp {
            id: id.into(),
            range,
            replacement: replacement.to_vec(),
        }
    }

    #[test]
    fn applies_high_to_low_without_shifting() {
        let s = snap(b"aaa bbb ccc");
        let applied =
            apply_in_memory(&s, &[op("1", (0, 3), b"XXXXX"), op("2", (8, 11), b"Y")]).unwrap();
        assert_eq!(applied.content, b"XXXXX bbb Y");
        assert!(outside_ranges_unchanged(
            &s.content,
            &applied.content,
            &[op("1", (0, 3), b"XXXXX"), op("2", (8, 11), b"Y")]
        ));
    }

    #[test]
    fn rejects_overlap_without_changing_anything() {
        let s = snap(b"0123456789");
        let err = apply_in_memory(&s, &[op("a", (0, 5), b""), op("b", (3, 7), b"")]);
        assert!(err.is_err());
    }

    #[test]
    fn outside_check_detects_clobber() {
        let before = b"aaa bbb ccc";
        // simulate a buggy apply that also mutated untouched bytes
        let after = b"XXX bbb cZc";
        assert!(!outside_ranges_unchanged(
            before,
            after,
            &[op("1", (0, 3), b"XXX")]
        ));
    }

    #[test]
    fn syntax_counts_flag_broken_rust() {
        let ok = syntax_counts(Language::Rust, b"fn main() {}\n").unwrap();
        assert_eq!(
            ok,
            SyntaxCounts {
                errors: 0,
                missing: 0
            }
        );
        let broken = syntax_counts(Language::Rust, b"fn main( {}\n").unwrap();
        assert!(broken.errors + broken.missing > 0);
    }

    #[test]
    fn property_random_mutation_never_corrupts() {
        // deterministic pseudo-random walk: any post-snapshot mutation must be
        // caught by the hash check before publish (verified here via sha
        // comparison, the same check publish performs)
        let s = snap(b"the quick brown fox jumps over the lazy dog");
        let mut seed = 0x9e3779b97f4a7c15u64;
        for _ in 0..500 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (seed >> 33) as usize % s.content.len();
            let mut live = s.content.clone();
            live[idx] ^= 0x20;
            assert_ne!(
                sha256_hex(&live),
                s.file_sha256,
                "mutation must change hash"
            );
        }
    }
}
