use greppy_workspace_core::ProviderInstallation;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::process::Stdio;
use std::thread;
use std::time::{Duration, Instant};

const START_TIMEOUT: Duration = Duration::from_secs(20);

pub(crate) fn setup(data_root: &Path) -> Result<ProviderInstallation, String> {
    if let Ok(provider) = ProviderInstallation::require_healthy(data_root) {
        provider
            .doctor_io(&format!("setup-{}", std::process::id()))
            .map_err(|error| error.to_string())?;
        return Ok(provider);
    }
    fs::create_dir_all(data_root).map_err(|error| {
        format!(
            "cannot create workspace data root {}: {error}",
            data_root.display()
        )
    })?;
    let mount_root = mount_root(data_root)?;
    fs::create_dir_all(&mount_root).map_err(|error| {
        format!(
            "cannot create workspace mount root {}: {error}",
            mount_root.display()
        )
    })?;
    start_platform_adapter(data_root, &mount_root)?;
    wait_until_healthy(data_root)
}

fn mount_root(data_root: &Path) -> Result<PathBuf, String> {
    let parent = data_root
        .parent()
        .ok_or_else(|| format!("workspace data root has no parent: {}", data_root.display()))?;
    Ok(parent.join("workspace-mount"))
}

fn sibling(current_exe: &Path, name: &str) -> Result<PathBuf, String> {
    let directory = current_exe.parent().ok_or_else(|| {
        format!(
            "cannot locate bundled workspace adapter beside {}",
            current_exe.display()
        )
    })?;
    Ok(directory.join(name))
}

#[cfg(target_os = "linux")]
fn start_platform_adapter(data_root: &Path, mount_root: &Path) -> Result<(), String> {
    use std::os::unix::process::CommandExt;

    if !Path::new("/dev/fuse").exists() {
        return Err(
            "FUSE3 is unavailable: /dev/fuse does not exist; install/enable the OS FUSE component and grant this user mount permission"
                .into(),
        );
    }
    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let adapter = sibling(&current, "greppy-workspace-provider")?;
    require_bundled_file(&adapter, "Linux FUSE3 adapter")?;
    let log_path = data_root.join("provider.log");
    let stdout = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|error| format!("cannot open {}: {error}", log_path.display()))?;
    let stderr = stdout
        .try_clone()
        .map_err(|error| format!("cannot clone provider log handle: {error}"))?;
    let child = Command::new(&adapter)
        .arg("--data-root")
        .arg(data_root)
        .arg("--mount-root")
        .arg(mount_root)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .process_group(0)
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", adapter.display()))?;
    publish_pid(data_root, child.id())?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn start_platform_adapter(data_root: &Path, _mount_root: &Path) -> Result<(), String> {
    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let app = sibling(&current, "GreppyWorkspaceFS.app")?;
    require_bundled_file(&app, "signed FSKit application")?;
    let status = Command::new("/usr/bin/open")
        .arg(&app)
        .status()
        .map_err(|error| format!("cannot open {}: {error}", app.display()))?;
    if !status.success() {
        return Err(format!(
            "macOS refused to open the signed FSKit application {}",
            app.display()
        ));
    }
    fs::write(data_root.join("activation-required"), b"fskit\n")
        .map_err(|error| format!("cannot record FSKit activation state: {error}"))?;
    Err(
        "FSKit activation is not complete; enable Greppy Workspace FS in System Settings > General > Login Items & Extensions, then run `greppy workspace setup` again"
            .into(),
    )
}

#[cfg(target_os = "windows")]
fn start_platform_adapter(data_root: &Path, mount_root: &Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    const DETACHED_PROCESS: u32 = 0x0000_0008;

    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let adapter = sibling(&current, "greppy-workspace-provider.exe")?;
    require_bundled_file(&adapter, "WinFsp provider")?;
    let child = Command::new(&adapter)
        .arg("--data-root")
        .arg(data_root)
        .arg("--mount-root")
        .arg(mount_root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
        .spawn()
        .map_err(|error| format!("cannot start {}: {error}", adapter.display()))?;
    publish_pid(data_root, child.id())?;
    Ok(())
}

fn require_bundled_file(path: &Path, label: &str) -> Result<(), String> {
    let metadata = fs::metadata(path)
        .map_err(|error| format!("bundled {label} is missing at {}: {error}", path.display()))?;
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(format!(
            "bundled {label} has an invalid type at {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
fn publish_pid(data_root: &Path, pid: u32) -> Result<(), String> {
    let final_path = data_root.join("provider.pid");
    let temporary = data_root.join(format!("provider.pid.{pid}.tmp"));
    fs::write(&temporary, format!("{pid}\n"))
        .map_err(|error| format!("cannot write {}: {error}", temporary.display()))?;
    fs::rename(&temporary, &final_path)
        .map_err(|error| format!("cannot publish {}: {error}", final_path.display()))?;
    Ok(())
}

fn wait_until_healthy(data_root: &Path) -> Result<ProviderInstallation, String> {
    let started = Instant::now();
    let mut last_error = String::from("provider has not published its identity");
    while started.elapsed() < START_TIMEOUT {
        match ProviderInstallation::require_healthy(data_root) {
            Ok(provider) => {
                provider
                    .doctor_io(&format!("setup-{}", std::process::id()))
                    .map_err(|error| error.to_string())?;
                return Ok(provider);
            }
            Err(error) => last_error = error.to_string(),
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!(
        "workspace provider did not become healthy within {} seconds: {last_error}; inspect {}",
        START_TIMEOUT.as_secs(),
        data_root.join("provider.log").display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_is_a_sibling_not_an_alias_of_private_data() {
        let data = Path::new("/tmp/greppy-test/workspace");
        assert_eq!(
            mount_root(data).unwrap(),
            Path::new("/tmp/greppy-test/workspace-mount")
        );
    }

    #[test]
    fn bundled_adapter_is_resolved_beside_the_cli() {
        assert_eq!(
            sibling(Path::new("/opt/greppy/bin/greppy"), "provider").unwrap(),
            Path::new("/opt/greppy/bin/provider")
        );
    }
}
