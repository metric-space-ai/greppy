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
fn start_platform_adapter(data_root: &Path, mount_root: &Path) -> Result<(), String> {
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
    let device = attach_fskit_anchor(data_root)?;
    let mount = Command::new("/sbin/mount")
        .arg("-t")
        .arg("greppy-cow")
        .arg(&device)
        .arg(mount_root)
        .status()
        .map_err(|error| format!("cannot invoke macOS mount for Greppy FSKit: {error}"))?;
    if mount.success() {
        fs::write(
            data_root.join("fskit-device"),
            format!("{}\n", device.display()),
        )
        .map_err(|error| format!("cannot record FSKit anchor device: {error}"))?;
        let _ = fs::remove_file(data_root.join("activation-required"));
        return Ok(());
    }
    let _ = Command::new("/usr/bin/hdiutil")
        .arg("detach")
        .arg(&device)
        .status();
    fs::write(data_root.join("activation-required"), b"fskit\n")
        .map_err(|error| format!("cannot record FSKit activation state: {error}"))?;
    let _ = Command::new("/usr/bin/open")
        .arg("x-apple.systempreferences:com.apple.LoginItems-Settings.extension")
        .status();
    Err(
        "macOS did not mount Greppy Workspace FS; enable it once in System Settings > General > Login Items & Extensions > File System Extensions, then rerun `greppy workspace setup`"
            .into(),
    )
}

#[cfg(target_os = "macos")]
fn attach_fskit_anchor(data_root: &Path) -> Result<PathBuf, String> {
    let anchor = data_root.join("fskit-anchor.sparseimage");
    if !anchor.is_file() {
        let output_base = data_root.join("fskit-anchor");
        let created = Command::new("/usr/bin/hdiutil")
            .arg("create")
            .arg("-size")
            .arg("8m")
            .arg("-layout")
            .arg("NONE")
            .arg("-type")
            .arg("SPARSE")
            .arg(&output_base)
            .output()
            .map_err(|error| format!("cannot create FSKit anchor image: {error}"))?;
        if !created.status.success() || !anchor.is_file() {
            return Err(format!(
                "macOS failed to create the private FSKit anchor image: {}",
                String::from_utf8_lossy(&created.stderr).trim()
            ));
        }
    }
    let attached = Command::new("/usr/bin/hdiutil")
        .arg("attach")
        .arg("-nomount")
        .arg("-nobrowse")
        .arg("-noverify")
        .arg(&anchor)
        .output()
        .map_err(|error| format!("cannot attach FSKit anchor image: {error}"))?;
    if !attached.status.success() {
        return Err(format!(
            "macOS failed to attach the private FSKit anchor image: {}",
            String::from_utf8_lossy(&attached.stderr).trim()
        ));
    }
    parse_hdiutil_device(&String::from_utf8_lossy(&attached.stdout)).ok_or_else(|| {
        format!(
            "hdiutil attached the FSKit anchor without reporting a device: {}",
            String::from_utf8_lossy(&attached.stdout).trim()
        )
    })
}

#[cfg(target_os = "macos")]
fn parse_hdiutil_device(output: &str) -> Option<PathBuf> {
    output
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .find(|value| value.starts_with("/dev/disk"))
        .map(PathBuf::from)
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

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_hdiutil_anchor_device_without_localized_columns() {
        assert_eq!(
            parse_hdiutil_device("/dev/disk9\tApple_partition_scheme\n"),
            Some(PathBuf::from("/dev/disk9"))
        );
        assert_eq!(parse_hdiutil_device("hdiutil: no device\n"), None);
    }
}
