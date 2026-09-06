//! `greppy web` client. Does not link V8 or Servo.
//!
//! Command surface is split into bundles so independent verbs can land in
//! parallel without rewriting this file. `#[command(flatten)]` keeps the
//! CLI flat (`greppy web goto`, not `greppy web nav goto`).

mod act;
mod chain;
mod chain_output;
mod common;
mod diagnose;
mod expect;
mod nav;
mod results;
mod runtimes;
mod see;
mod sessions;
mod view;
mod workflow;

use clap::Subcommand;
use greppy_core::error::Result;

pub use act::ActCommand;
pub use chain::ChainCommand;
pub use common::{prestart_unsandboxed, shutdown_if_running, web_runtime_socket};
pub use diagnose::DiagnoseCommand;
pub use expect::ExpectCommand;
pub use nav::NavCommand;
pub use results::ResultsCommand;
pub use runtimes::RuntimesCommand;
pub use see::SeeCommand;
pub use sessions::SessionsCommand;

#[derive(Debug, Subcommand)]
pub enum WebCommand {
    #[command(flatten)]
    Sessions(SessionsCommand),
    #[command(flatten)]
    Results(ResultsCommand),
    #[command(flatten)]
    Nav(NavCommand),
    #[command(flatten)]
    Act(ActCommand),
    #[command(flatten)]
    See(SeeCommand),
    #[command(flatten)]
    Chain(ChainCommand),
    #[command(flatten)]
    Runtimes(RuntimesCommand),
    #[command(flatten)]
    Expect(ExpectCommand),
    #[command(flatten)]
    Diagnose(DiagnoseCommand),
}

pub fn dispatch(command: WebCommand, root: Option<&str>) -> Result<i32> {
    struct StandaloneShutdown;
    impl Drop for StandaloneShutdown {
        fn drop(&mut self) {
            if crate::web_attach::should_shutdown_on_scope_end() {
                common::shutdown_if_running();
            }
        }
    }
    let _standalone = StandaloneShutdown;
    dispatch_inner(command, root)
}

/// Dispatch without the standalone-shutdown guard.
///
/// `web do` runs several commands inside one process. Routing each step
/// through `dispatch` would arm a second guard whose `Drop` tears the runtime
/// down after the first step, so a chain must use this entry instead.
pub(super) fn dispatch_inner(command: WebCommand, root: Option<&str>) -> Result<i32> {
    match command {
        WebCommand::Sessions(command) => sessions::dispatch(command, root),
        WebCommand::Results(command) => results::dispatch(command, root),
        WebCommand::Nav(command) => nav::dispatch(command, root),
        WebCommand::Act(command) => act::dispatch(command, root),
        WebCommand::See(command) => see::dispatch(command, root),
        WebCommand::Chain(command) => chain::dispatch(command, root),
        WebCommand::Runtimes(command) => runtimes::dispatch(command, root),
        WebCommand::Expect(command) => expect::dispatch(command, root),
        WebCommand::Diagnose(command) => diagnose::dispatch(command, root),
    }
}

#[cfg(test)]
mod tests {
    use super::common::{
        export_regular_file, find_binary, images_from_dist, runtime_executable_name,
    };
    use super::sessions::SessionCommand;
    use super::*;
    use crate::{Cli, Command};
    use clap::Parser;
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;

    static EXPORT_CWD_LOCK: Mutex<()> = Mutex::new(());

    fn export_sandbox(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parse_web_status_json() {
        let cli = Cli::try_parse_from(["greppy", "web", "status", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Sessions(SessionsCommand::Status { json: true })
            })
        ));
    }

    #[test]
    fn parse_web_goto_is_flat_not_nested_under_nav() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "goto",
            "http://example.com/",
            "--session",
            "wrs_1",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Nav(NavCommand::Goto {
                    url,
                    session: Some(session),
                    json: true,
                    ..
                })
            }) if url == "http://example.com/" && session == "wrs_1"
        ));
        assert!(
            Cli::try_parse_from(["greppy", "web", "nav", "goto", "http://example.com/"]).is_err()
        );
    }

    #[test]
    fn parse_web_click_is_flat_not_nested_under_act() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "click",
            "css=button",
            "--session",
            "wrs_1",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Act(ActCommand::Click { target, opts, .. })
            }) if target == "css=button"
                && opts.session.as_deref() == Some("wrs_1")
                && opts.json
        ));
        assert!(Cli::try_parse_from(["greppy", "web", "act", "click", "css=button"]).is_err());
    }

    #[test]
    fn parse_web_fill_value_is_positional() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "fill",
            "css=input",
            "hello",
            "--session",
            "wrs_1",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Act(ActCommand::Fill {
                    target,
                    value: Some(value),
                    opts,
                    ..
                })
            }) if target == "css=input"
                && value == "hello"
                && opts.session.as_deref() == Some("wrs_1")
        ));
    }

    #[test]
    fn parse_web_doctor_json() {
        let cli = Cli::try_parse_from(["greppy", "web", "doctor", "--json"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Sessions(SessionsCommand::Doctor { json: true })
            })
        ));
    }

    #[test]
    fn parse_web_observe_search_read_research_flags() {
        let cli = Cli::try_parse_from(["greppy", "web", "observe", "--session", "wrs_1", "--json"])
            .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Observe {
                    session: Some(session),
                    json: true,
                    ..
                })
            }) if session == "wrs_1"
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "search",
            "--query",
            "greppy",
            "--domain",
            "example.com",
            "--session",
            "wrs_1",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Search {
                    query: Some(query),
                    domain: Some(domain),
                    session: Some(session),
                    json: true,
                    ..
                })
            }) if query == "greppy" && domain == "example.com" && session == "wrs_1"
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "read",
            "--url",
            "https://example.com/article",
            "--session",
            "wrs_1",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Read {
                    url: Some(url),
                    session: Some(session),
                    json: true,
                    ..
                })
            }) if url == "https://example.com/article" && session == "wrs_1"
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "research",
            "--query",
            "greppy",
            "--max-sources",
            "2",
            "--depth",
            "shallow",
            "--session",
            "wrs_1",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Research {
                    query: Some(query),
                    max_sources: Some(2),
                    depth: Some(depth),
                    session: Some(session),
                    json: true,
                    ..
                })
            }) if query == "greppy" && depth == "shallow" && session == "wrs_1"
        ));
    }

    #[test]
    fn parse_web_search_result_limit_and_global_limit() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "search",
            "--query",
            "greppy",
            "--session",
            "wrs_1",
            "--limit",
            "3",
            "--json",
        ])
        .expect("web search --limit must not panic from the global/subcommand clap type clash");
        assert_eq!(cli.limit, Some(3));
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Search {
                    result_limit: None,
                    json: true,
                    ..
                })
            })
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "search",
            "--query",
            "greppy",
            "--session",
            "wrs_1",
            "--result-limit",
            "7",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Search {
                    result_limit: Some(7),
                    json: true,
                    ..
                })
            })
        ));
    }

    #[test]
    fn parse_web_search_fixture_url_and_search_endpoint() {
        let cli = Cli::try_parse_from([
            "greppy",
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
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Search {
                    fixture_url: Some(fixture_url),
                    search_endpoint: Some(search_endpoint),
                    json: true,
                    ..
                })
            }) if fixture_url == "http://127.0.0.1:9/search.html"
                && search_endpoint == "http://127.0.0.1:9/search"
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "read",
            "--url",
            "https://example.com/article",
            "--session",
            "wrs_1",
            "--fixture-url",
            "http://127.0.0.1:9/page.html",
            "--search-endpoint",
            "http://127.0.0.1:9/search",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Read {
                    fixture_url: Some(fixture_url),
                    search_endpoint: Some(search_endpoint),
                    json: true,
                    ..
                })
            }) if fixture_url == "http://127.0.0.1:9/page.html"
                && search_endpoint == "http://127.0.0.1:9/search"
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "research",
            "--query",
            "greppy",
            "--session",
            "wrs_1",
            "--fixture-url",
            "http://127.0.0.1:9/search.html",
            "--search-endpoint",
            "http://127.0.0.1:9/search",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Research {
                    fixture_url: Some(fixture_url),
                    search_endpoint: Some(search_endpoint),
                    json: true,
                    ..
                })
            }) if fixture_url == "http://127.0.0.1:9/search.html"
                && search_endpoint == "http://127.0.0.1:9/search"
        ));
    }

    #[test]
    fn parse_web_screenshot_output() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "screenshot",
            "--session",
            "wrs_1",
            "--output",
            "/tmp/shot.png",
            "--render-complete",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Screenshot {
                    output: Some(path),
                    render_complete: true,
                    json: true,
                    ..
                })
            }) if path == "/tmp/shot.png"
        ));
    }

    #[cfg(unix)]
    #[test]
    fn export_regular_file_writes_and_refuses_symlink_and_directory() {
        let dir = export_sandbox("greppy-web-export");
        let file = dir.join("shot.png");
        export_regular_file(&file, b"png-bytes").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"png-bytes");

        let existing = dir.join("exists.bin");
        std::fs::write(&existing, b"keep-me").unwrap();
        let exist_err = export_regular_file(&existing, b"clobber").unwrap_err();
        assert!(exist_err.message.contains("existing"), "{exist_err:?}");
        assert_eq!(std::fs::read(&existing).unwrap(), b"keep-me");

        let subdir = dir.join("subdir");
        std::fs::create_dir(&subdir).unwrap();
        let dir_err = export_regular_file(&subdir, b"nope").unwrap_err();
        assert!(dir_err.message.contains("directory"), "{dir_err:?}");

        let target = dir.join("target.bin");
        std::fs::write(&target, b"orig").unwrap();
        let link = dir.join("link.png");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let link_err = export_regular_file(&link, b"hijack").unwrap_err();
        assert!(link_err.message.contains("symlink"), "{link_err:?}");
        assert_eq!(std::fs::read(&target).unwrap(), b"orig");
        assert!(std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink());

        let linked_parent = dir.join("linked-parent");
        std::os::unix::fs::symlink(&dir, &linked_parent).unwrap();
        let nested = linked_parent.join("nested.png");
        let parent_err = export_regular_file(&nested, b"escape").unwrap_err();
        assert!(parent_err.message.contains("symlink"), "{parent_err:?}");
        assert!(
            !nested.exists()
                || std::fs::symlink_metadata(&nested)
                    .unwrap()
                    .file_type()
                    .is_symlink()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn export_regular_file_rejects_relative_path_through_symlinked_ancestor() {
        let _cwd = EXPORT_CWD_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let dir = export_sandbox("greppy-web-export-rel");
        let real = dir.join("real");
        std::fs::create_dir(&real).unwrap();
        let link = dir.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&dir).unwrap();
        let err = export_regular_file(Path::new("link/shot.png"), b"rel");
        let cwd_after = std::env::current_dir();
        let _ = std::env::set_current_dir(&previous);
        let err = err.unwrap_err();
        assert!(
            err.message.contains("symlink"),
            "{err:?} cwd_after={cwd_after:?}"
        );
        assert!(
            !real.join("shot.png").exists(),
            "relative export must not write through symlink ancestor"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parse_web_session_and_run() {
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "session",
            "create",
            "--profile",
            "project",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Sessions(SessionsCommand::Session {
                    command: SessionCommand::Create { .. }
                })
            })
        ));
        let cli = Cli::try_parse_from([
            "greppy",
            "web",
            "run",
            "--session",
            "wrs_1",
            "--script-file",
            "spec.mjs",
            "--json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Web {
                command: WebCommand::Results(ResultsCommand::Run { json: true, .. })
            })
        ));
    }

    #[test]
    fn missing_named_binary_is_none() {
        assert!(find_binary("web-runtime-missing-name").is_none());
    }

    #[test]
    fn stamped_dist_resolves_one_linked_executable() {
        let root = std::env::temp_dir().join(format!("greppy-web-cli-dist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(
            root.join(".greppy-web-runtime-dist"),
            "greppy.web-runtime.package.v1\n",
        )
        .unwrap();
        std::fs::write(
            root.join("bin").join(runtime_executable_name()),
            b"web-runtime",
        )
        .unwrap();
        let runtime = images_from_dist(&root).expect("dist");
        assert_eq!(runtime.dist.as_ref(), Some(&root));
        assert!(runtime.executable.ends_with(runtime_executable_name()));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn stamped_dist_refuses_bin_member_symlink() {
        let root =
            std::env::temp_dir().join(format!("greppy-web-cli-dist-link-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("bin")).unwrap();
        std::fs::write(
            root.join(".greppy-web-runtime-dist"),
            "greppy.web-runtime.package.v1\n",
        )
        .unwrap();
        let target = root.join("payload");
        std::fs::write(&target, b"web-runtime").unwrap();
        std::os::unix::fs::symlink(&target, root.join("bin").join(runtime_executable_name()))
            .unwrap();
        assert!(images_from_dist(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn unstamped_dir_is_not_a_web_runtime_dist() {
        let root =
            std::env::temp_dir().join(format!("greppy-web-cli-nodist-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        assert!(images_from_dist(&root).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }
}
