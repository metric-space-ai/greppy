//! `greppy web` CLI wiring. Does not require JavaScript engines.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn run(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .env_remove("GREPPY_WEB_RUNTIME")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env_remove("GREPPY_WEB_FIXTURE_URL")
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("greppy web");
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn inspect_ref_validates_syntax_before_runtime_start() {
    for reference in ["@0", "@invalid"] {
        let (code, stdout, stderr) = run(&[
            "web",
            "inspect",
            reference,
            "--session",
            "wrs_unstarted",
            "--tab",
            "tab_unstarted",
            "--json",
        ]);
        assert_eq!(code, 30, "stdout={stdout} stderr={stderr}");
        let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(value["error"]["code"], "QUERY_SYNTAX");
        assert_eq!(value["error"]["retryable"], false);
        assert!(value["error"]["message"].as_str().unwrap().contains("ref"));
        assert!(!stdout.contains("engine_error"));
        assert!(!stdout.contains("SyntaxError"));
    }
    let (code, stdout, stderr) = run(&["web", "inspect", "--help"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("accepts @N"), "{stdout}");
}

#[test]
fn session_create_help_explains_both_profiles_without_relaxing_policy() {
    let (code, stdout, stderr) = run(&["web", "session", "create", "--help"]);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.contains("research: public web"), "{stdout}");
    assert!(
        stdout.contains("project: public web plus loopback"),
        "{stdout}"
    );
    assert!(
        stdout.contains("LAN and cloud metadata remain blocked"),
        "{stdout}"
    );
}

#[test]
fn action_expect_help_distinguishes_query_type_from_shell_quoting() {
    for action in ["select", "check", "click", "fill"] {
        let (code, stdout, stderr) = run(&["web", action, "--help"]);
        assert_eq!(code, 0, "{action}: {stdout}\n{stderr}");
        assert!(stderr.is_empty(), "{stderr}");
        let stdout = stdout.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(stdout.contains("Bare QUERY is CSS"), "{stdout}");
        assert!(stdout.contains("text=Saved"), "{stdout}");
        assert!(stdout.contains("text~/Saved/i"), "{stdout}");
        assert!(
            stdout.contains("quotes alone do not change query type"),
            "{stdout}"
        );
        assert!(stdout.contains("not visibility"), "{stdout}");
    }
}

#[test]
fn session_new_refusal_names_create_and_nested_usage() {
    let (code, stdout, stderr) = run(&["web", "session", "new", "--profile", "project"]);
    assert_eq!(code, 64, "stdout={stdout} stderr={stderr}");
    let text = format!("{stdout}{stderr}");
    assert!(
        text.contains("greppy web session create --profile project"),
        "{text}"
    );
    assert!(text.contains("usage: greppy web session"), "{text}");
    assert!(!text.contains("usage: greppy web status|doctor"), "{text}");
}

#[test]
fn chain_session_position_refusal_names_per_step_recovery() {
    let (code, stdout, stderr) = run(&[
        "web",
        "do",
        "--explain",
        "--session",
        "STUDY",
        "click",
        "@1",
        "::",
        "observe",
    ]);
    assert_eq!(code, 30, "stdout={stdout} stderr={stderr}");
    let text = format!("{stdout}{stderr}");
    assert!(text.contains("after each step's command"), "{text}");
    assert!(
        text.contains("click @1 --session SID :: observe --session SID"),
        "{text}"
    );
    let (code, stdout, stderr) = run(&[
        "web",
        "do",
        "--explain",
        "click",
        "@1",
        "--session",
        "STUDY",
        "::",
        "observe",
        "--session",
        "STUDY",
    ]);
    assert_eq!(code, 0, "stdout={stdout} stderr={stderr}");
    assert!(stdout.contains("STUDY"), "{stdout}");
}

#[test]
fn web_is_not_grep_passthrough() {
    let (code, stdout, stderr) = run(&["web", "status", "--json"]);
    assert_ne!(code, 2, "web must not be unknown clap usage: {stderr}");
    assert!(
        stdout.contains("runtime_unavailable")
            || stdout.contains("experimental")
            || stdout.contains("greppy.web-runtime.v1"),
        "stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_status_json_is_unavailable_without_runtime_bins() {
    let (code, stdout, _stderr) = run(&["web", "status", "--json"]);
    assert_eq!(code, 31, "stdout={stdout}");
    assert!(
        stdout.contains("\"schema\":\"greppy.web-runtime.v1\"")
            || stdout.contains("greppy.web-runtime.v1")
    );
    assert!(stdout.contains("runtime_unavailable") || stdout.contains("error"));
    #[cfg(target_os = "windows")]
    assert!(
        stdout.contains("web tool is not available on Windows in 0.4.0"),
        "stdout={stdout}"
    );
}

#[test]
fn web_runtime_is_a_subcommand() {
    let (code, stdout, stderr) = run(&["web", "--help"]);
    assert_eq!(code, 0, "stderr={stderr}");
    assert!(
        stdout.contains("runtime"),
        "web --help must list runtime, stdout={stdout}"
    );
}

#[test]
fn web_runtime_status_does_not_spawn_and_reports_not_running() {
    let started = Instant::now();
    let (code, stdout, stderr) = run(&["web", "runtime", "status", "--json"]);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "runtime status must not hash/spawn the engine, elapsed={:?}",
        started.elapsed()
    );
    assert_eq!(code, 0, "stdout={stdout} stderr={stderr}");
    assert!(stdout.contains("\"running\":false") || stdout.contains("\"running\": false"));
    assert!(stdout.contains("greppy.web-runtime.v1"));
}

#[test]
fn web_artifact_and_result_are_subcommands() {
    let (code, stdout, stderr) = run(&["web", "--help"]);
    assert_eq!(code, 0, "stderr={stderr}");
    assert!(stdout.contains("artifact"), "stdout={stdout}");
    assert!(stdout.contains("result"), "stdout={stdout}");
}

#[test]
fn web_artifact_list_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "artifact", "list", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_artifact_show_path_export_are_listed() {
    let (code, stdout, stderr) = run(&["web", "artifact", "--help"]);
    assert_eq!(code, 0, "stderr={stderr}");
    for verb in ["list", "show", "path", "export"] {
        assert!(stdout.contains(verb), "missing {verb} in {stdout}");
    }
}

#[test]
fn web_artifact_show_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "artifact", "show", "abc", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_artifact_path_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "artifact", "path", "abc", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_artifact_export_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "artifact", "export", "abc", "--to", "/tmp/x"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_artifact_export_rejects_json() {
    let (code, stdout, _stderr) = run(&[
        "web",
        "artifact",
        "export",
        "abc",
        "--to",
        "/tmp/x",
        "--session",
        "wrs_1",
        "--json",
    ]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("raw bytes") || stdout.contains("--json"));
}

#[test]
fn web_result_next_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "result", "next", "offset=3", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_goto_back_forward_reload_open_are_listed() {
    let (code, stdout, stderr) = run(&["web", "--help"]);
    assert_eq!(code, 0, "stderr={stderr}");
    for verb in ["goto", "back", "forward", "reload", "open"] {
        assert!(stdout.contains(verb), "missing {verb} in {stdout}");
    }
    assert!(
        !stdout.contains("web nav"),
        "navigation must stay flat, stdout={stdout}"
    );
}

#[test]
fn web_goto_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "goto", "http://example.com/", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_act_verbs_are_listed_flat() {
    let (code, stdout, stderr) = run(&["web", "--help"]);
    assert_eq!(code, 0, "stderr={stderr}");
    for verb in [
        "click", "fill", "type", "clear", "select", "check", "uncheck", "press", "hover", "scroll",
        "upload",
    ] {
        assert!(stdout.contains(verb), "missing {verb} in {stdout}");
    }
    assert!(
        !stdout.contains("web act"),
        "actions must stay flat, stdout={stdout}"
    );
}

#[test]
fn web_click_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "click", "css=button", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_fill_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "fill", "css=input", "x", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_click_rejects_unknown_target_syntax() {
    let (code, stdout, _stderr) = run(&["web", "click", "button", "--session", "wrs_1", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("QUERY_SYNTAX") || stdout.contains("css="));
}

#[test]
fn web_run_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "run", "--script-file", "x.js", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_observe_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "observe", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_search_requires_query() {
    let (code, stdout, _stderr) = run(&["web", "search", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("query"));
}

#[test]
fn web_read_requires_url() {
    let (code, stdout, _stderr) = run(&["web", "read", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("url"));
}

#[test]
fn web_research_requires_query() {
    let (code, stdout, _stderr) = run(&["web", "research", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("query"));
}

#[test]
fn web_search_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "search", "--query", "greppy", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_read_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "read", "--url", "https://example.com", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_research_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "research", "--query", "greppy", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}

#[test]
fn web_search_accepts_fixture_url_and_search_endpoint_flags() {
    let (code, stdout, stderr) = run(&[
        "web",
        "search",
        "--query",
        "greppy",
        "--session",
        "wrs_1",
        "--fixture-url",
        "http://127.0.0.1:9/search.html",
        "--search-endpoint",
        "http://127.0.0.1:9/search",
        "--json",
    ]);
    assert_ne!(
        code, 2,
        "must not be unknown clap usage: stdout={stdout} stderr={stderr}"
    );
    assert!(
        !stderr.contains("panicked") && !stdout.contains("panicked"),
        "stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_search_limit_does_not_panic() {
    let (code, stdout, stderr) = run(&[
        "web",
        "search",
        "--query",
        "greppy",
        "--session",
        "wrs_1",
        "--limit",
        "3",
        "--json",
    ]);
    assert_ne!(
        code, 101,
        "web search --limit must not panic from clap type clash: stdout={stdout} stderr={stderr}"
    );
    assert!(
        !stderr.contains("panicked") && !stdout.contains("panicked"),
        "web search --limit panicked: stdout={stdout} stderr={stderr}"
    );
    assert_ne!(
        code, 2,
        "web search --limit must parse: stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_search_result_limit_is_parsed() {
    let (code, stdout, stderr) = run(&[
        "web",
        "search",
        "--query",
        "greppy",
        "--session",
        "wrs_1",
        "--result-limit",
        "3",
        "--json",
    ]);
    assert_ne!(
        code, 2,
        "web search --result-limit must be a recognised flag: stdout={stdout} stderr={stderr}"
    );
    assert!(
        !stderr.contains("panicked") && !stdout.contains("panicked"),
        "stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_fixture_url_env_is_not_a_production_path() {
    let output = Command::new(bin())
        .args(["web", "status", "--json"])
        .env_remove("GREPPY_WEB_RUNTIME")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env("GREPPY_WEB_FIXTURE_URL", "http://127.0.0.1:9/search.html")
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("greppy web status fixture env");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert_eq!(
        output.status.code().unwrap_or(1),
        30,
        "stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("GREPPY_WEB_FIXTURE_URL") || stderr.contains("GREPPY_WEB_FIXTURE_URL"),
        "stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("not a production path") || stderr.contains("not a production path"),
        "must fail-closed rather than silently forward env, stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_status_resolves_packaged_dist_env() {
    let runtime_root = tempfile::Builder::new()
        .prefix("g-diag-")
        .tempdir_in(if cfg!(unix) {
            std::path::Path::new("/tmp")
        } else {
            std::path::Path::new(".")
        })
        .unwrap();
    let pid = std::process::id();
    let dist = std::env::temp_dir().join(format!("greppy-web-dist-cli-{pid}"));
    let _ = std::fs::remove_dir_all(&dist);
    std::fs::create_dir_all(dist.join("bin")).unwrap();
    std::fs::write(
        dist.join(".greppy-web-runtime-dist"),
        "greppy.web-runtime.package.v1\n",
    )
    .unwrap();
    let dummy = dist.join("bin").join("web-runtime");
    std::fs::write(
        &dummy,
        "#!/bin/sh\necho 'fixture worker initialization failed' >&2\nexit 1\n",
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dummy).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dummy, perms).unwrap();
    }
    let mut child = Command::new(bin())
        .args(["web", "status", "--json"])
        .env_remove("GREPPY_WEB_RUNTIME")
        .env_remove("GREPPY_WEB_FIXTURE_URL")
        .env("GREPPY_WEB_RUNTIME_DIST", &dist)
        .env("GREPPY_RUNTIME_DIR", runtime_root.path())
        .env_remove("GREPPY_WEB_RUNTIME_DIR")
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("greppy web status dist");
    let mut stdout_pipe = child.stdout.take().expect("stdout");
    let mut stderr_pipe = child.stderr.take().expect("stderr");
    let deadline = Instant::now() + Duration::from_secs(8);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_dir_all(&dist);
                panic!(
                    "web status against dummy dist exceeded 8s; dummy must exit without blocking on worker capability FD"
                );
            }
            Err(error) => panic!("wait dummy dist status: {error}"),
        }
    };
    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();
    stdout_pipe.read_to_end(&mut stdout_buf).ok();
    stderr_pipe.read_to_end(&mut stderr_buf).ok();
    let stdout = String::from_utf8_lossy(&stdout_buf).into_owned();
    let stderr = String::from_utf8_lossy(&stderr_buf).into_owned();
    let _ = std::fs::remove_dir_all(&dist);
    #[cfg(unix)]
    {
        let dummy_path = dummy.display().to_string();
        let leaked = Command::new("ps")
            .args(["-ax", "-o", "args="])
            .output()
            .ok()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .any(|line| line.contains(&dummy_path) && !line.contains("ps "))
            })
            .unwrap_or(false);
        assert!(
            !leaked,
            "failed dummy web-runtime must be reaped, not left as PPID 1"
        );
    }
    assert_ne!(
        status.code().unwrap_or(1),
        2,
        "web must not be unknown clap usage: {stderr}"
    );
    assert!(
        !stdout.contains("web-runtime distributable is not installed"),
        "GREPPY_WEB_RUNTIME_DIST must resolve the stamped dist, stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("runtime_unavailable")
            || stdout.contains("failed to spawn")
            || stdout.contains("did not create")
            || stdout.contains("greppy.web-runtime.v1"),
        "expected a spawn/runtime failure against the dummy dist, stdout={stdout} stderr={stderr}"
    );
    #[cfg(unix)]
    {
        assert!(stdout.contains("startup stderr retained at"), "{stdout}");
        assert!(stdout.contains("exit status: 1"), "{stdout}");
        let logs: Vec<_> = std::fs::read_dir(runtime_root.path())
            .unwrap()
            .flatten()
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("web-startup-")
            })
            .collect();
        assert_eq!(logs.len(), 1, "failed startup needs one inspectable log");
        assert!(std::fs::read_to_string(logs[0].path())
            .unwrap()
            .contains("fixture worker initialization failed"));
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            logs[0].metadata().unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn web_doctor_json_does_not_spawn_runtime() {
    let pid = std::process::id();
    let dist = std::env::temp_dir().join(format!("greppy-web-dist-doctor-{pid}"));
    let store = std::env::temp_dir().join(format!("greppy-web-store-doctor-{pid}"));
    let _ = std::fs::remove_dir_all(&dist);
    let _ = std::fs::remove_dir_all(&store);
    std::fs::create_dir_all(dist.join("bin")).unwrap();
    std::fs::create_dir_all(&store).unwrap();
    std::fs::write(
        dist.join(".greppy-web-runtime-dist"),
        "greppy.web-runtime.package.v1\n",
    )
    .unwrap();
    let dummy = dist.join("bin").join("web-runtime");
    std::fs::write(&dummy, "#!/bin/sh\nexit 1\n").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dummy).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dummy, perms).unwrap();
    }
    let started = Instant::now();
    let output = Command::new(bin())
        .args(["web", "doctor", "--json"])
        .env_remove("GREPPY_WEB_RUNTIME")
        .env_remove("GREPPY_WEB_FIXTURE_URL")
        .env("GREPPY_WEB_RUNTIME_DIST", &dist)
        .env("GREPPY_STORE_DIR", &store)
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("greppy web doctor dist");
    let elapsed = started.elapsed();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let dummy_path = dummy.display().to_string();
    let _ = std::fs::remove_dir_all(&dist);
    let _ = std::fs::remove_dir_all(&store);
    assert!(
        elapsed < Duration::from_secs(30),
        "doctor hung instead of returning handshake facts, elapsed={elapsed:?} stdout={stdout} stderr={stderr}"
    );
    assert_eq!(
        output.status.code().unwrap_or(1),
        0,
        "doctor against dummy dist must succeed without exec, stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("\"protocol_version\"") || stdout.contains("greppy.web-runtime.v1"),
        "doctor must report handshake schema, stdout={stdout}"
    );
    assert!(
        stdout.contains("web-runtime-0.1.0"),
        "doctor must report runtime_build_id, stdout={stdout}"
    );
    assert!(
        !stdout.contains("process_health") && !stdout.contains("controller_alive"),
        "doctor must not report live workers, stdout={stdout}"
    );
    #[cfg(unix)]
    {
        let leaked = Command::new("ps")
            .args(["-ax", "-o", "args="])
            .output()
            .ok()
            .map(|output| {
                String::from_utf8_lossy(&output.stdout)
                    .lines()
                    .any(|line| line.contains(&dummy_path) && !line.contains("ps "))
            })
            .unwrap_or(false);
        assert!(
            !leaked,
            "doctor must not spawn the dummy web-runtime executable"
        );
    }
}

#[cfg(unix)]
fn locate_web_runtime() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("GREPPY_WEB_RUNTIME") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    let greppy = std::path::PathBuf::from(bin());
    if let Some(dir) = greppy.parent() {
        let sibling = dir.join("web-runtime");
        if sibling.is_file() {
            return Some(sibling);
        }
    }
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../web-runtime/runtime/target/debug/web-runtime"),
        manifest.join("../web-runtime/target/debug/web-runtime"),
        manifest.join("../../target/debug/web-runtime"),
        manifest.join("../web-runtime/runtime/target/release/web-runtime"),
    ];
    candidates.into_iter().find(|candidate| candidate.is_file())
}

#[cfg(unix)]
fn wait_unix_socket(path: &std::path::Path, child: &mut std::process::Child, budget: Duration) {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            return;
        }
        match child.try_wait() {
            Ok(Some(status)) => panic!(
                "supervisor exited {status} before socket {} accepted",
                path.display()
            ),
            Ok(None) | Err(_) => std::thread::sleep(Duration::from_millis(50)),
        }
    }
    panic!(
        "supervisor socket {} was not accepting within {budget:?}",
        path.display()
    );
}

#[cfg(unix)]
fn greppy_web_status(
    run_id: &str,
    runtime: &std::path::Path,
    token: Option<&str>,
) -> (i32, String, String) {
    use greppy::give_child_attach_token;
    let mut cmd = Command::new(bin());
    cmd.args(["web", "status", "--json"])
        .env("GREPPY_RUN_ID", run_id)
        .env("GREPPY_WEB_RUNTIME", runtime)
        .env_remove("GREPPY_WEB_ATTACH")
        .env_remove("GREPPY_WEB_FIXTURE_URL")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env("PATH", "/usr/bin:/bin")
        .stdin(Stdio::null());
    let hold = token
        .map(|token| give_child_attach_token(&mut cmd, token).expect("inherit attach token fd"));
    let output = cmd.output().expect("greppy web status child");
    drop(hold);
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[cfg(unix)]
#[test]
fn parent_owned_attach_authorizes_separate_cli_children_and_denies_others() {
    use greppy::{generate_attach_token, give_child_attach_token, web_runtime_socket};
    use greppy_web_client::Request;
    use serde_json::json;
    use std::os::unix::process::CommandExt;

    let Some(runtime) = locate_web_runtime() else {
        eprintln!("skipping attach CLI proof: optional web-runtime binary is not built");
        return;
    };
    let run_id = format!(
        "run_attachcli_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let token =
        generate_attach_token().expect("urandom must fail-closed, not yield zeros silently");
    assert!(token.len() >= 16, "token={token}");
    assert_ne!(
        token,
        "0".repeat(token.len()),
        "entropy must not be silent zeros"
    );
    let socket = web_runtime_socket(&run_id).expect("web-runtime socket path");
    let _ = std::fs::remove_file(&socket);

    let mut supervisor = Command::new(&runtime);
    supervisor
        .arg("--socket")
        .arg(&socket)
        .arg("--run-id")
        .arg(&run_id)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .process_group(0);
    let pass = give_child_attach_token(&mut supervisor, &token).expect("supervisor attach fd");
    let mut supervisor_child = supervisor.spawn().expect("spawn supervisor");
    drop(pass);
    wait_unix_socket(&socket, &mut supervisor_child, Duration::from_secs(60));

    let (code, stdout, stderr) = greppy_web_status(&run_id, &runtime, Some(&token));
    assert_eq!(
        code, 0,
        "authorized child 1 must attach via inherited fd, stdout={stdout} stderr={stderr}"
    );
    assert!(
        !stdout.contains("GREPPY_WEB_ATTACH") && !stderr.contains("GREPPY_WEB_ATTACH"),
        "capability must not travel through env, stdout={stdout} stderr={stderr}"
    );

    let (code, stdout, stderr) = greppy_web_status(&run_id, &runtime, Some(&token));
    assert_eq!(
        code, 0,
        "authorized child 2 is a separate process and must reuse the parent token via fd, stdout={stdout} stderr={stderr}"
    );

    let (code, stdout, stderr) = greppy_web_status(&run_id, &runtime, None);
    assert_ne!(
        code, 0,
        "missing fd must fail-closed, stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("session_not_owned") || stderr.contains("session_not_owned"),
        "missing attach must be session_not_owned, stdout={stdout} stderr={stderr}"
    );

    let wrong = "00".repeat(16);
    let (code, stdout, stderr) = greppy_web_status(&run_id, &runtime, Some(&wrong));
    assert_ne!(
        code, 0,
        "wrong token must fail-closed, stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("session_not_owned") || stderr.contains("session_not_owned"),
        "wrong attach must be session_not_owned, stdout={stdout} stderr={stderr}"
    );

    let mut missing = Request::new(&run_id, "web.status", json!({}));
    missing.capability = String::new();
    let denied = greppy_web_client::unix_request(&socket, &missing, Duration::from_secs(5))
        .expect("endpoint-only connect");
    assert_eq!(denied.status, "error", "{denied:?}");
    assert_eq!(
        denied.error.as_ref().map(|error| error.code.as_str()),
        Some("session_not_owned"),
        "{denied:?}"
    );

    let mut shutdown = Request::new(&run_id, "web.shutdown", json!({}));
    shutdown.capability = token;
    let stopped = greppy_web_client::unix_request(&socket, &shutdown, Duration::from_secs(5))
        .expect("authorized shutdown");
    assert_eq!(stopped.status, "ok", "{stopped:?}");
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match supervisor_child.try_wait() {
            Ok(Some(_)) | Err(_) => break,
            Ok(None) if Instant::now() >= deadline => {
                let _ = supervisor_child.try_wait();
                panic!("supervisor did not exit after web.shutdown; do not treat kill as success");
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
        }
    }
}

#[cfg(unix)]
fn unrelated_child_must_not_see_token(token: &str) -> (i32, String, String) {
    let output = Command::new("/usr/bin/perl")
        .arg("-e")
        .arg(
            r#"
use Fcntl;
my $token = $ARGV[0];
for my $fd (3 .. 64) {
    my $flags = fcntl($fd, F_GETFD, 0);
    next unless defined $flags;
    if ($fd == 4) {
        print "FD4_OPEN\n";
        exit 3;
    }
    my $buf = "";
    sysseek($fd, 0, 0);
    sysread($fd, $buf, 256);
    if (index($buf, $token) >= 0) {
        print "TOKEN_ON_FD_${fd}\n";
        exit 2;
    }
}
exit 0;
"#,
        )
        .arg(token)
        .stdin(Stdio::null())
        .output()
        .expect("unrelated child");
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[cfg(unix)]
#[test]
fn concurrent_unrelated_children_do_not_inherit_attach_token_fd() {
    use greppy::{generate_attach_token, give_child_attach_token};
    use std::thread;

    let token = generate_attach_token().expect("urandom");
    let mut holder = Command::new("/usr/bin/true");
    holder
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let pass = give_child_attach_token(&mut holder, &token).expect("atomic cloexec token fd");

    let mut joins = Vec::new();
    for _ in 0..12 {
        let token = token.clone();
        joins.push(thread::spawn(move || {
            unrelated_child_must_not_see_token(&token)
        }));
    }
    for join in joins {
        let (code, stdout, stderr) = join.join().expect("thread");
        assert_eq!(
            code, 0,
            "unrelated child inherited attach fd/token stdout={stdout} stderr={stderr}"
        );
        assert!(
            !stdout.contains(&token) && !stderr.contains(&token),
            "token leaked to unrelated child stdout={stdout} stderr={stderr}"
        );
        assert!(
            !stdout.contains("FD4_OPEN") && !stdout.contains("TOKEN_ON_FD_"),
            "attach fd leaked stdout={stdout} stderr={stderr}"
        );
    }

    let mut authorized = Command::new("/usr/bin/perl");
    authorized
        .arg("-e")
        .arg("my $buf=''; open(F,q{<&=},4) or die $!; sysread(F,$b,256); print $b;")
        .stdin(Stdio::null());
    let auth_pass = give_child_attach_token(&mut authorized, &token).expect("authorized fd");
    let output = authorized.output().expect("authorized child");
    drop(auth_pass);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains(&token),
        "authorized child must read token from fd 4, stdout={stdout:?}"
    );
    drop(pass);
    drop(holder);
}

#[cfg(unix)]
fn serve_scope_pages() -> String {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind scope pages");
    let address = listener.local_addr().expect("addr");
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut buffer = [0_u8; 2048];
            let n = stream.read(&mut buffer).unwrap_or(0);
            let req = String::from_utf8_lossy(&buffer[..n]);
            let body = if req.contains("GET /001/") {
                "<!DOCTYPE html><html><head><title>001</title></head><body><h1>Seite-001</h1></body></html>"
            } else {
                "<!DOCTYPE html><html><head><title>fx</title></head><body><h1>Kopf</h1></body></html>"
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(body.as_bytes());
        }
    });
    format!("http://{address}")
}

#[cfg(unix)]
fn run_scoped(
    workspace: &std::path::Path,
    runtime: &std::path::Path,
    run_id: &str,
    args: &[&str],
) -> (i32, String, String) {
    let output = Command::new(bin())
        .arg("--root")
        .arg(workspace)
        .args(args)
        .current_dir(workspace)
        .env("GREPPY_RUN_ID", run_id)
        .env("GREPPY_WEB_RUNTIME", runtime)
        .env_remove("GREPPY_WEB_SESSION")
        .env_remove("GREPPY_WEB_TAB")
        .env_remove("GREPPY_WEB_AGENT")
        .env_remove("GREPPY_WEB_ATTACH")
        .env_remove("GREPPY_WEB_FIXTURE_URL")
        .env_remove("GREPPY_WEB_RUNTIME_DIST")
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("greppy web scoped");
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[cfg(unix)]
#[test]
fn web_open_observe_goto_observe_share_session_without_flag() {
    let Some(runtime) = locate_web_runtime() else {
        eprintln!("skipping scoped web proof: optional web-runtime binary is not built");
        return;
    };
    let run_id = format!(
        "run_scope_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let workspace = std::env::temp_dir().join(format!("greppy-web-scope-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&workspace);
    std::fs::create_dir_all(&workspace).expect("workspace");
    let base = serve_scope_pages();
    let fx = format!("{base}/_fx/index.html");
    let page = format!("{base}/001/index.html");

    let (code, stdout, stderr) = run_scoped(
        &workspace,
        &runtime,
        &run_id,
        &["web", "open", &fx, "--json"],
    );
    if code != 0 {
        let _ = run_scoped(
            &workspace,
            &runtime,
            &run_id,
            &["web", "runtime", "stop", "--json"],
        );
        let _ = std::fs::remove_dir_all(&workspace);
        panic!("open failed code={code} stdout={stdout} stderr={stderr}");
    }
    let current = workspace.join(".greppy/web/current.json");
    assert!(
        current.is_file(),
        "open must write current.json at {}",
        current.display()
    );

    let (code, stdout, stderr) =
        run_scoped(&workspace, &runtime, &run_id, &["web", "observe", "--json"]);
    if code != 0 || !stdout.contains("Kopf") {
        let _ = run_scoped(
            &workspace,
            &runtime,
            &run_id,
            &["web", "runtime", "stop", "--json"],
        );
        let _ = std::fs::remove_dir_all(&workspace);
        panic!("observe after open must show Kopf without --session code={code} stdout={stdout} stderr={stderr}");
    }

    let (code, stdout, stderr) = run_scoped(
        &workspace,
        &runtime,
        &run_id,
        &["web", "goto", &page, "--json"],
    );
    if code != 0 {
        let _ = run_scoped(
            &workspace,
            &runtime,
            &run_id,
            &["web", "runtime", "stop", "--json"],
        );
        let _ = std::fs::remove_dir_all(&workspace);
        panic!("goto without --session failed code={code} stdout={stdout} stderr={stderr}");
    }

    let (code, stdout, stderr) =
        run_scoped(&workspace, &runtime, &run_id, &["web", "observe", "--json"]);
    let _ = run_scoped(
        &workspace,
        &runtime,
        &run_id,
        &["web", "runtime", "stop", "--json"],
    );
    let _ = std::fs::remove_dir_all(&workspace);
    assert_eq!(
        code, 0,
        "second observe failed stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("Seite-001"),
        "observe after goto must show the new page, stdout={stdout}"
    );
}

#[test]
fn web_goto_without_scope_is_no_session() {
    let (code, stdout, _stderr) = run(&["web", "goto", "http://example.com/", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(
        stdout.contains("NO_SESSION") || stdout.contains("session"),
        "stdout={stdout}"
    );
}
