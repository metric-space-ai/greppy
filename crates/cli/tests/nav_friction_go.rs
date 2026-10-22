//! Friction measured in the gotd/td bench (SWE-rebench July 2026, glm-5.3-flash,
//! pi+greppy, 10 runs): every rejected call below cost the agent one more turn.
//! - `search-symbol NAME --kind struct` found no Go struct (`type X struct` is
//!   shown as kind `type`), and `--kind type` was rejected.
//! - `read tg.Client::Method` (Go package-qualified) was taken for a file path.
//! - `search-pattern -i` / `-A N` were rejected as unknown flags.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn go_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-nav-friction-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("tg")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("tg/stickers.go"),
        "package tg\n\n\
         // MessagesStickerSet is a sticker set.\n\
         type MessagesStickerSet struct {\n\tSet string\n}\n\n\
         // StickerSource yields stickers.\n\
         type StickerSource interface {\n\tStickers() []string\n}\n\n\
         // ID is a plain alias.\n\
         type ID int64\n\n\
         // Client calls the API.\n\
         type Client struct{}\n\n\
         // MessagesGetFavedStickers returns the faved stickers.\n\
         func (c *Client) MessagesGetFavedStickers() []string {\n\treturn nil\n}\n",
    )
    .unwrap();
    let store = base.join("store");
    let (code, stdout, stderr) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    (repo, store)
}

#[test]
fn kind_struct_and_type_find_go_struct_types() {
    let (repo, store) = go_workspace("kind");
    for kind in ["struct", "type", "class"] {
        let (code, stdout, stderr) = run(
            &repo,
            &store,
            &["search-symbol", "MessagesStickerSet", "--kind", kind],
        );
        assert_eq!(code, 0, "--kind {kind}: stdout={stdout}\nstderr={stderr}");
        assert!(
            stdout.contains("tg/stickers.go") && stdout.contains("MessagesStickerSet"),
            "--kind {kind}: {stdout}"
        );
        assert!(!stdout.contains("ignored unknown --kind"), "{stdout}");
    }
    // Precise: an interface is no struct, a plain alias is no struct either.
    let (_, stdout, _) = run(
        &repo,
        &store,
        &["search-symbol", "StickerSource", "--kind", "struct"],
    );
    assert!(stdout.contains("no_matches"), "{stdout}");
    let (_, stdout, _) = run(&repo, &store, &["search-symbol", "ID", "--kind", "struct"]);
    assert!(!stdout.contains("tg/stickers.go:14"), "{stdout}");
    let (code, stdout, _) = run(
        &repo,
        &store,
        &["search-symbol", "StickerSource", "--kind", "interface"],
    );
    assert_eq!(code, 0, "{stdout}");
    assert!(stdout.contains("StickerSource"), "{stdout}");
}

#[test]
fn go_package_qualified_method_resolves() {
    let (repo, store) = go_workspace("pkgq");
    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "tg.Client::MessagesGetFavedStickers"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("func (c *Client) MessagesGetFavedStickers()"),
        "{stdout}"
    );
    // A real file head still narrows to that file.
    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "tg/stickers.go::MessagesGetFavedStickers"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
}

#[test]
fn search_pattern_accepts_grep_case_and_context_flags() {
    let (repo, store) = go_workspace("pattern");
    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["search-pattern", "-i", "messagesstickerset"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("tg/stickers.go:4"), "{stdout}");

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "search-pattern",
            "type MessagesStickerSet struct",
            "--fixed",
            "-A",
            "2",
        ],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("5- \tSet string"), "{stdout}");
    assert!(
        stdout.contains("4: type MessagesStickerSet struct {"),
        "{stdout}"
    );

    let (code, stdout, _) = run(&repo, &store, &["search-pattern", "-C", "1", "Set string"]);
    assert_eq!(code, 0, "{stdout}");
    assert!(
        stdout.contains("4- type MessagesStickerSet struct {"),
        "{stdout}"
    );
    assert!(stdout.contains("6- }"), "{stdout}");
}
