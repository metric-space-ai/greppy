use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        let number = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "greppy-trial-it-{tag}-{}-{number}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create test scratch directory");
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn git(root: &Path, args: &[&str]) {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn repository(scratch: &Scratch) -> PathBuf {
    let root = scratch.0.join("repo");
    std::fs::create_dir_all(root.join("src")).expect("create source directory");
    std::fs::write(
        root.join("src/lib.rs"),
        "pub fn target_symbol() {}\npub fn caller_one() { target_symbol(); }\n",
    )
    .expect("write fixture source");
    git(&root, &["init", "-q"]);
    git(
        &root,
        &["config", "user.email", "trial-test@example.invalid"],
    );
    git(&root, &["config", "user.name", "Trial Test"]);
    git(&root, &["add", "."]);
    git(&root, &["commit", "-q", "-m", "fixture"]);
    root
}

#[cfg(unix)]
fn fake_pi(scratch: &Scratch) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;

    let bin_dir = scratch.0.join("fake-bin");
    std::fs::create_dir_all(&bin_dir).expect("create fake bin directory");
    let pi = bin_dir.join("pi");
    std::fs::write(
        &pi,
        r#"#!/bin/sh
if [ "$1" = "--version" ]; then
  printf '%s\n' 'pi fake 1.0.0'
  exit 0
fi

for flag in --provider --model --mode --no-session --tools --no-context-files --no-skills --no-prompt-templates --no-extensions --no-themes --session-dir --system-prompt; do
  case " $* " in
    *" $flag "*) ;;
    *) printf 'missing required flag: %s\n' "$flag" >&2; exit 91 ;;
  esac
done

case "$PI_CODING_AGENT_DIR" in
  *greppy-project-trial-*/baseline-pi-config|*greppy-project-trial-*/greppy-pi-config) ;;
  *) printf '%s\n' 'PI config was not isolated' >&2; exit 92 ;;
esac
case "$PI_CODING_AGENT_SESSION_DIR" in
  *greppy-project-trial-*/baseline-pi-session|*greppy-project-trial-*/greppy-pi-session) ;;
  *) printf '%s\n' 'PI session was not isolated' >&2; exit 93 ;;
esac

case "$PWD" in
  */greppy)
    case "$*" in
      *"who-calls 'target_symbol'"*) ;;
      *) printf '%s\n' 'system prompt omitted exact symbol command' >&2; exit 94 ;;
    esac
    command_text="greppy --root . who-calls target_symbol"
    if [ "$FAKE_PI_MODE" = "quality_regression" ]; then
      answer="no matching caller"
    else
      answer="caller_one calls target_symbol"
    fi
    ;;
  */baseline)
    if [ "$FAKE_PI_MODE" = "contaminated" ]; then
      command_text="greppy who-calls target_symbol"
    else
      command_text="grep -R target_symbol src"
    fi
    answer="caller_one calls target_symbol"
    ;;
  *) printf '%s\n' 'unexpected worktree' >&2; exit 95 ;;
esac

printf '%s\n' "{\"type\":\"turn_end\",\"toolResults\":[{\"toolCallId\":\"call-1\",\"toolName\":\"bash\",\"content\":[{\"type\":\"text\",\"text\":\"caller_one source\"}]}],\"message\":{\"content\":[{\"type\":\"toolCall\",\"id\":\"call-1\",\"name\":\"bash\",\"arguments\":{\"command\":\"$command_text\"}}],\"usage\":{\"input\":10,\"output\":2,\"cacheRead\":3}}}"
printf '%s\n' "{\"type\":\"turn_end\",\"toolResults\":[],\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"$answer\"}],\"usage\":{\"input\":7,\"output\":4,\"cacheWrite1h\":2}}}"
"#,
    )
    .expect("write fake Pi");
    let mut permissions = std::fs::metadata(&pi)
        .expect("read fake Pi metadata")
        .permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&pi, permissions).expect("make fake Pi executable");
    bin_dir
}

fn trial_args(root: &Path) -> Vec<String> {
    vec![
        "trial".into(),
        "--root".into(),
        root.display().to_string(),
        "--question".into(),
        "Who calls target_symbol?".into(),
        "--check".into(),
        "who-calls".into(),
        "--symbol".into(),
        "target_symbol".into(),
        "--expect".into(),
        "caller_one".into(),
        "--forbid".into(),
        "caller_two".into(),
        "--runner".into(),
        "pi".into(),
        "--provider".into(),
        "fake-provider".into(),
        "--model".into(),
        "fake-model".into(),
        "--timeout-seconds".into(),
        "30".into(),
    ]
}

#[cfg(unix)]
fn run_trial(scratch: &Scratch, root: &Path, mode: &str) -> Output {
    let fake_bin = fake_pi(scratch);
    let path = std::env::join_paths(std::iter::once(fake_bin).chain(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    )))
    .expect("build test PATH");
    Command::new(bin())
        .args(trial_args(root))
        .env("PATH", path)
        .env("FAKE_PI_MODE", mode)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env(
            "GREPPY_STORE_DIR",
            scratch.0.join("ambient-store-must-stay-unused"),
        )
        .output()
        .expect("run trial")
}

fn json_output(output: &Output) -> Value {
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "trial stdout was not JSON: {error}\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    })
}

#[cfg(unix)]
#[test]
fn valid_observation_records_isolation_identity_tokens_and_comparison() {
    let scratch = Scratch::new("valid");
    let root = repository(&scratch);
    let output = run_trial(&scratch, &root, "valid");
    assert_eq!(output.status.code(), Some(0));
    let report = json_output(&output);

    assert_eq!(report["schema_version"], "greppy.project-trial.v1");
    assert_eq!(report["status"], "valid_observation");
    assert_eq!(report["repository"]["commit"].as_str().unwrap().len(), 40);
    assert_eq!(report["repository"]["tree"].as_str().unwrap().len(), 40);
    assert_eq!(report["runtime"]["os"], std::env::consts::OS);
    assert_eq!(report["runtime"]["arch"], std::env::consts::ARCH);
    assert_eq!(
        report["runtime"]["pi"]["sha256"].as_str().unwrap().len(),
        64
    );
    assert_eq!(
        report["runtime"]["greppy"]["sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    assert_eq!(
        report["runtime"]["pi"]["version_output"]["stdout"],
        "pi fake 1.0.0"
    );
    assert_eq!(report["protocol"]["worktrees_removed"], true);
    assert_eq!(report["arms"][0]["arm"], "baseline");
    assert_eq!(report["arms"][1]["arm"], "greppy");
    assert_eq!(report["arms"][0]["metrics"]["tool_call_count"], 1);
    assert_eq!(
        report["arms"][0]["metrics"]["token_counters"]["first_turn"]["input_tokens"],
        13
    );
    assert_eq!(
        report["arms"][0]["metrics"]["token_counters"]["later_turns"]["input_tokens"],
        9
    );
    assert_eq!(
        report["arms"][0]["metrics"]["token_counters"]["aggregate"]["input_tokens"],
        22
    );
    assert_eq!(report["comparison"]["quality_relationship"], "both_passed");
    assert!(report["comparison"]["metrics"]["tool_calls"]["greppy_over_baseline"].is_number());
    assert!(!scratch.0.join("ambient-store-must-stay-unused").exists());
}

#[cfg(unix)]
#[test]
fn greppy_grade_failure_is_quality_regression() {
    let scratch = Scratch::new("quality-regression");
    let root = repository(&scratch);
    let output = run_trial(&scratch, &root, "quality_regression");
    assert_eq!(output.status.code(), Some(1));
    let report = json_output(&output);
    assert_eq!(report["status"], "quality_regression");
    assert_eq!(
        report["comparison"]["quality_relationship"],
        "greppy_failed_baseline_passed"
    );
}

#[cfg(unix)]
#[test]
fn baseline_greppy_invocation_makes_pair_inconclusive() {
    let scratch = Scratch::new("contaminated");
    let root = repository(&scratch);
    let output = run_trial(&scratch, &root, "contaminated");
    assert_eq!(output.status.code(), Some(2));
    let report = json_output(&output);
    assert_eq!(report["status"], "inconclusive");
    assert_eq!(report["arms"][0]["baseline_invoked_greppy"], true);
    assert_eq!(report["comparison"]["comparable"], false);
}

#[test]
fn dirty_git_root_is_rejected_before_runner_execution() {
    let scratch = Scratch::new("dirty");
    let root = repository(&scratch);
    std::fs::write(root.join("untracked.txt"), "dirty\n").expect("dirty repository");
    let output = Command::new(bin())
        .args(trial_args(&root))
        .env(
            "GREPPY_STORE_DIR",
            scratch.0.join("ambient-store-must-stay-unused"),
        )
        .output()
        .expect("run dirty-root trial");
    assert_eq!(output.status.code(), Some(2));
    let report = json_output(&output);
    assert_eq!(report["schema_version"], "greppy.project-trial.v1");
    assert_eq!(report["status"], "inconclusive");
    assert!(report["reasons"][0]
        .as_str()
        .unwrap()
        .contains("must be clean"));
    assert!(!scratch.0.join("ambient-store-must-stay-unused").exists());
}
