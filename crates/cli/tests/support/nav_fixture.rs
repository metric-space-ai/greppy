//! A small Rust + TypeScript workspace for the navigation regression tests
//! (`nav_rust_callers`, `nav_typescript_members`, `nav_answers`). Each
//! construct in it reproduces a defect found on a real workspace; the tests
//! index it with the shipped binary and query it end to end.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

pub fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

pub fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-navcomplete-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

pub const LIB_RS: &str =
    "mod blobs;\nmod brief;\nmod calls;\nmod dock;\nmod fs;\nmod node;\nmod store_path;\npub use dock::*;\n";

pub const DOCK_RS: &str = r#"pub struct Dock;

impl Dock {
    pub async fn open(_folder: &str) -> Result<Dock, String> {
        Ok(Dock)
    }
    pub async fn serve(&self, _kinds: &[&str], _handler: fn()) -> Result<(), String> {
        Ok(())
    }
    pub async fn stand(&self, _subject: &str) -> Result<(), String> {
        Ok(())
    }
}

pub fn carry_out() {}

pub async fn publish() {}

pub struct Rights {
    pub may_lead: bool,
}

impl Dock {
    pub async fn rights_of(&self, _node: &str) -> Result<Rights, String> {
        Ok(Rights { may_lead: true })
    }
}
"#;

pub const NODE_RS: &str = r#"use std::sync::Arc;

use crate::dock::{carry_out, publish, Dock};

pub struct Shown;

impl Shown {
    pub fn say(&self, _subject: &str) {}
}

pub fn answer_one() {}

pub fn dispatch() {
    answer_one(); // site-1
    answer_one(); // site-2
    answer_one(); // site-3
}

pub async fn run() -> Result<(), String> {
    let dock = Dock::open("apps").await?;
    tokio::select! {
        ended = dock.serve(&["a"], carry_out) => ended, // select-serve
        _ = publish() => Ok(()), // select-publish
    }
}

pub fn keep(shown: Arc<Shown>) {
    shown.say("now"); // say-arc
    let later = shown.clone();
    std::thread::spawn(move || {
        later.say("later"); // say-closure
    });
}

pub async fn checked() {
    let dock = Dock::open("x").await.unwrap();
    dock.stand("up").await.unwrap(); // stand-unwrap
}

pub async fn admitted(dock: &Dock) -> Result<bool, String> {
    let rights = dock.rights_of("me").await.map_err(|failure| failure)?;
    Ok(rights.may_lead) // may-lead-read
}

pub fn may_lead() -> bool {
    false
}
"#;

pub const CALLS_RS: &str = r#"use std::collections::HashMap;

pub trait Service {
    fn admits(&self, asker: &str) -> bool;
    fn answer(&self, asker: &str);
}

pub fn ask(with: &dyn Service) -> bool {
    with.admits("me") // admits-dyn
}

pub fn route(services: &HashMap<String, Box<dyn Service>>) {
    if let Some(with) = services.get("blobs") {
        with.answer("me"); // answer-unproven
    }
}

pub struct Calls;
pub struct Call;

impl Calls {
    pub fn new() -> Calls {
        Calls
    }
}

impl Call {
    pub fn new() -> Call {
        Call
    }
}

pub fn make() {
    let _calls = Calls::new();
    let _call = Call::new();
}

pub struct Folder;

impl Folder {
    fn secret(&self, name: &str) -> String {
        name.to_string()
    }

    pub fn keep(&self, secrets: &[(String, String)]) -> Vec<String> {
        let _joined: Vec<String> = secrets.iter().map(|(name, secret)| format!("{name}{secret}")).collect();
        let mut out = Vec::new();
        for (name, secret) in secrets {
            out.push(self.secret(name)); // secret-call
            let _ = secret;
        }
        out
    }
}

pub struct SourceState {
    pub next_page: Option<usize>,
}

pub fn resume(state: Option<SourceState>) -> Option<usize> {
    state.and_then(|state| state.next_page.map(|page| page + 1)) // next-page-closure
}

pub struct Source {
    pub article_type: String,
}

pub fn article(source: &Source) -> String {
    serde_json::json!({ "type": source.article_type }).to_string() // article-type-macro
}

fn check_source() -> Result<Source, String> {
    Ok(Source { article_type: String::new() })
}

pub fn listed() -> String {
    let source = match check_source() {
        Ok(source) => source,
        Err(reason) => return reason,
    };
    source.article_type // article-type-match
}

pub trait Greets {
    fn greet(&self) -> String {
        String::new()
    }
}

pub struct Host;

impl Greets for Host {}

pub fn welcome(host: &Host) -> String {
    host.greet() // default-method-call
}

pub fn welcome_any<T: Greets>(guest: &T) -> String {
    guest.greet() // generic-bound-call
}

fn loopback(x: u32) -> bool {
    x == 1
}

pub fn base_ok(x: u32) -> bool {
    let loopback = loopback(x);
    loopback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loops() {
        assert!(loopback(1)); // loopback-test
    }
}
"#;

pub const BLOBS_RS: &str = r#"use crate::calls::Service;

pub struct Blobs;

impl Service for Blobs {
    fn admits(&self, _asker: &str) -> bool {
        true
    }
    fn answer(&self, _asker: &str) {}
}
"#;

pub const STORE_PATH_RS: &str = r#"pub struct Path;

impl Path {
    pub fn join(&self, _part: &str) -> Path {
        Path
    }
}

pub fn own(path: &Path) -> Path {
    path.join("own") // own-join
}

pub struct Connection;

impl Connection {
    pub fn transaction(&self) {}
}

pub fn begin(conn: &Connection) {
    conn.transaction(); // own-transaction
}
"#;

pub const FS_RS: &str = r#"use std::path::Path;

use rusqlite::Connection;

pub fn save(conn: &Connection) {
    conn.transaction(); // foreign-transaction
}

pub fn folder(folder: &Path) {
    let _ = folder.join("x"); // std-join
}

pub fn dup() {}
"#;

pub const BRIEF_RS: &str = r#"/// Run one trigger of an installed app while this node leads (aborted when it no longer does
/// or the app changes).
pub async fn run_trigger() {
    helper();
}

fn helper() {}

pub fn dup() {}

/// Write the event of a mission command that ended Failed, if it has none.
pub async fn audit(_dock: &u8) -> Result<(), &'static str> {
    Ok(())
}
"#;

/// Another package with its own `Dock::serve` and a call whose receiver
/// type is unknown: neither belongs to the first package's `Dock`.
pub const OTHER_LIB_RS: &str = r#"pub struct Dock;

impl Dock {
    pub fn serve(&self) {}
    pub fn stand_by(&self) {}
}

pub fn unrelated() {
    let conn = make();
    conn.serve(); // other-serve
}

fn make() -> Dock {
    Dock
}
"#;

/// A workspace package (`wj-dock`) imported by its crate name, which the
/// resolver cannot follow without a workspace manifest: the package's own
/// manifest still names the crate.
pub const WJ_DOCK_CARGO: &str = "[package]\nname = \"wj-dock\"\nversion = \"0.1.0\"\n";

pub const WJ_DOCK_LIB_RS: &str = r#"pub struct Dock;

impl Dock {
    pub fn stand_by(&self) {}
}
"#;

pub const APP_LIB_RS: &str = r#"use wj_dock::Dock;

pub fn wait(dock: &Dock) {
    dock.stand_by(); // workspace-import
}
"#;

pub const WINDOW_TS: &str = r#"export interface EngineWindow {
  /** The same as last heard; null before the node answered once. */
  agentNow(): string | null;
}

export type EngineClient = {
  agentNow(): string | null; // client-member
};

export function makeWindow(): EngineWindow {
  const heard: string | null = null;
  return {
    agentNow: () => heard,
  };
}

export function lastHeard(window: EngineWindow): string | null {
  return window.agentNow(); // agent-now-call
}

/**
 * The model of this name among those the agent offers. Throws, in words, if it
 * cannot be asked with that model.
 */
export function engineModelWanted(name: string): string {
  return name;
}

export function filter(rows: number[]): number[] {
  return rows;
}

export function positive(rows: number[]): number[] {
  return rows.filter((row) => row > 0); // builtin-filter
}
"#;

pub const ROWS_TS: &str = r#"import * as window from "./window";

export interface Repo {
  find(id: string): string;
}

export class Shelf {
  repo!: Repo;
  first(): string {
    return this.repo.find("a"); // member-find
  }
}

export function kept(rows: number[]): number[] {
  return window.filter(rows); // namespace-filter
}
"#;

pub fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    let web = repo.join("web");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(&web).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    for (name, text) in [
        ("lib.rs", LIB_RS),
        ("dock.rs", DOCK_RS),
        ("node.rs", NODE_RS),
        ("calls.rs", CALLS_RS),
        ("blobs.rs", BLOBS_RS),
        ("store_path.rs", STORE_PATH_RS),
        ("fs.rs", FS_RS),
        ("brief.rs", BRIEF_RS),
    ] {
        std::fs::write(src.join(name), text).unwrap();
    }
    std::fs::write(web.join("window.ts"), WINDOW_TS).unwrap();
    std::fs::write(web.join("rows.ts"), ROWS_TS).unwrap();
    let other = repo.join("other").join("src");
    std::fs::create_dir_all(&other).unwrap();
    std::fs::write(other.join("lib.rs"), OTHER_LIB_RS).unwrap();
    let dock = repo.join("engine").join("dock");
    std::fs::create_dir_all(dock.join("src")).unwrap();
    std::fs::write(dock.join("Cargo.toml"), WJ_DOCK_CARGO).unwrap();
    std::fs::write(dock.join("src").join("lib.rs"), WJ_DOCK_LIB_RS).unwrap();
    let app = repo.join("app").join("src");
    std::fs::create_dir_all(&app).unwrap();
    std::fs::write(app.join("lib.rs"), APP_LIB_RS).unwrap();
    (repo, root.join("store"))
}

pub fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store_dir)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

pub fn indexed(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");
    (repo, store)
}

/// The 1-based line of the fixture line carrying `marker`.
pub fn line_of(text: &str, marker: &str) -> usize {
    text.lines()
        .position(|line| line.contains(marker))
        .unwrap_or_else(|| panic!("fixture has no line with {marker}"))
        + 1
}

pub fn who_calls(repo: &Path, store: &Path, symbol: &str) -> String {
    let (code, out, err) = run(&["who-calls", symbol], repo, store);
    assert_eq!(code, 0, "who-calls {symbol}: stderr={err}\nstdout={out}");
    out
}
