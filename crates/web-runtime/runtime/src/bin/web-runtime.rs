use std::ffi::OsString;
use std::io;
use web_runtime::supervisor::{self, Config};

// Force relink after in-place libjs_static.a localization; cargo does not
// fingerprint that archive as a rustc input.

fn main() -> io::Result<()> {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match internal_role(&args)? {
        InternalRole::Controller => web_runtime::controller_worker::run(),
        InternalRole::Content => web_runtime::content_worker::run(),
        InternalRole::Supervisor => {
            let config = Config::parse(strip_internal_role(args))
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
            supervisor::run(config)
        }
    }
}

#[derive(Clone, Copy)]
enum InternalRole {
    Supervisor,
    Controller,
    Content,
}

fn internal_role(args: &[OsString]) -> io::Result<InternalRole> {
    let mut role = None;
    let mut args = args.iter();
    while let Some(argument) = args.next() {
        if argument.to_str() != Some("--internal-role") {
            continue;
        }
        let value = args.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "missing value after --internal-role",
            )
        })?;
        if role.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "duplicate --internal-role",
            ));
        }
        role = Some(match value.to_str() {
            Some("supervisor") => InternalRole::Supervisor,
            Some("controller") => InternalRole::Controller,
            Some("content") => InternalRole::Content,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("unknown --internal-role {value:?}"),
                ));
            }
        });
    }
    Ok(role.unwrap_or(InternalRole::Supervisor))
}

fn strip_internal_role(args: Vec<OsString>) -> Vec<OsString> {
    let mut out = Vec::with_capacity(args.len());
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        if argument.to_str() == Some("--internal-role") {
            let _ = args.next();
            continue;
        }
        out.push(argument);
    }
    out
}
