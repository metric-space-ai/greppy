use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "greppy-workspace-provider", version)]
struct Args {
    /// Private provider metadata and Chunk-CAS root.
    #[arg(long)]
    data_root: PathBuf,
    /// Persistent per-user mount point.
    #[arg(long)]
    mount_root: PathBuf,
}

fn main() {
    let args = Args::parse();
    if !args.data_root.is_absolute() || !args.mount_root.is_absolute() {
        eprintln!("workspace provider paths must be absolute");
        std::process::exit(64);
    }
    #[cfg(target_os = "linux")]
    if let Err(error) = linux::serve(args.data_root, args.mount_root) {
        eprintln!("greppy workspace provider failed: {error}");
        std::process::exit(1);
    }
    #[cfg(target_os = "windows")]
    if let Err(error) = windows::serve(args.data_root, args.mount_root) {
        eprintln!("greppy workspace provider failed: {error}");
        std::process::exit(1);
    }
    #[cfg(target_os = "macos")]
    {
        let _ = args;
        eprintln!(
            "this binary is the Linux FUSE3 adapter; macOS uses the bundled FSKit extension and Windows uses the bundled WinFsp provider"
        );
        std::process::exit(69);
    }
}

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "windows")]
mod windows;
