//! Language registry — the **parallel-safe** language onboarding surface.
//!
//! Adding a new language used to require editing six shared locations (the
//! `Language` enum, `name()`, `grammar()`, `cached_query_set()`, the extension
//! table, and `spec_for()`), so two agents adding two languages always
//! conflicted. This module removes that: a language is now **one
//! self-contained file** under `src/langs/<lang>.rs` that declares a
//! [`LangDef`] and self-registers via `inventory::submit!`. Nothing shared is
//! edited except the single `tree-sitter-<lang>` line in `Cargo.toml` (which
//! git auto-merges, since each is its own line).
//!
//! A `LangDef` bundles EVERYTHING the extraction engine needs for a language:
//! its display name, the file extensions / filenames that select it, a
//! grammar constructor, the declarative [`LangSpec`], and the three
//! tree-sitter query sources (definitions / calls / imports). Compiled queries
//! are cached per-language in a process-global side table keyed by the
//! `LangDef`'s address.
//!
//! At runtime [`Language::Registered`] wraps a `&'static LangDef`, so a
//! registry language behaves exactly like a hand-wired enum variant — same
//! `name()`, `grammar()`, extraction, and query caching.

use std::sync::OnceLock;

use crate::query::{CompiledQuery, QueryKind};
use crate::spec::LangSpec;

/// A fully self-contained language definition. One `inventory::submit!` of a
/// `LangDef` in a `src/langs/<lang>.rs` file wires a whole language.
pub struct LangDef {
    /// Display name (e.g. `"elixir"`) used in qualified names + errors.
    pub name: &'static str,
    /// File extensions (without the dot) that select this language.
    pub extensions: &'static [&'static str],
    /// Exact filenames (e.g. `"Dockerfile"`) that select this language.
    pub filenames: &'static [&'static str],
    /// Grammar constructor.
    pub grammar: fn() -> tree_sitter::Language,
    /// Declarative extraction spec.
    pub spec: &'static LangSpec,
    /// tree-sitter query source for definitions.
    pub def_query: &'static str,
    /// tree-sitter query source for calls.
    pub call_query: &'static str,
    /// tree-sitter query source for imports (may be empty).
    pub import_query: &'static str,
}

/// Per-language compiled-query cache, keyed by the `LangDef`'s address (each is
/// a unique `static`). Kept OUT of `LangDef` because `inventory::submit!`
/// const-promotes the submitted value, and interior mutability (`OnceLock`)
/// cannot be const-promoted. Compiled queries are leaked (they live for the
/// whole process anyway), so the returned `&'static [CompiledQuery]` is sound.
#[allow(clippy::type_complexity)]
fn query_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<usize, &'static [CompiledQuery]>> {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, &'static [CompiledQuery]>>,
    > = OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

// The `Language` enum derives PartialEq/Eq/Debug; a `LangDef` is a singleton
// `static`, so identity IS pointer identity. These impls let
// `Language::Registered(&'static LangDef)` participate in those derives without
// requiring the (uncomparable) `OnceLock` field to be comparable.
impl PartialEq for LangDef {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}
impl Eq for LangDef {}
impl std::fmt::Debug for LangDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LangDef").field("name", &self.name).finish()
    }
}

inventory::collect!(LangDef);

impl LangDef {
    /// Iterate every registered language (deterministic within a build).
    pub fn all() -> impl Iterator<Item = &'static LangDef> {
        inventory::iter::<LangDef>.into_iter()
    }

    /// Look up the registry language for a path by exact filename first, then
    /// by extension. Case-insensitive. `None` = not a registry language (the
    /// caller then falls back to the legacy enum table).
    pub fn for_path(path: &std::path::Path) -> Option<&'static LangDef> {
        if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
            if let Some(d) =
                Self::all().find(|d| d.filenames.iter().any(|f| f.eq_ignore_ascii_case(fname)))
            {
                return Some(d);
            }
        }
        let ext = path.extension().and_then(|s| s.to_str())?;
        Self::all().find(|d| d.extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)))
    }

    /// Compile (once) and return this language's query set. Errors on a
    /// malformed query source; never silently returns an empty set.
    pub fn compiled_queries(
        &'static self,
    ) -> Result<&'static [CompiledQuery], tree_sitter::QueryError> {
        let key = self as *const LangDef as usize;
        if let Some(q) = query_cache().lock().unwrap().get(&key) {
            return Ok(q);
        }
        let lang = (self.grammar)();
        let mut v = Vec::new();
        for (kind, src) in [
            (QueryKind::Definitions, self.def_query),
            (QueryKind::Calls, self.call_query),
            (QueryKind::Imports, self.import_query),
        ] {
            if !src.is_empty() {
                v.push(CompiledQuery::new(kind, lang.clone(), src)?);
            }
        }
        let leaked: &'static [CompiledQuery] = Box::leak(v.into_boxed_slice());
        query_cache().lock().unwrap().insert(key, leaked);
        Ok(leaked)
    }
}
