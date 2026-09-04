use greppy_workspace_core::ProviderInstallation;
use std::fs;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::io::Write;
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
        install_platform_autostart(data_root, provider.mount_root())?;
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
    let provider = wait_until_healthy(data_root)?;
    install_platform_autostart(data_root, provider.mount_root())?;
    Ok(provider)
}

#[cfg(not(target_os = "macos"))]
fn mount_root(data_root: &Path) -> Result<PathBuf, String> {
    let parent = data_root
        .parent()
        .ok_or_else(|| format!("workspace data root has no parent: {}", data_root.display()))?;
    Ok(parent.join("workspace-mount"))
}

#[cfg(target_os = "macos")]
fn mount_root(_data_root: &Path) -> Result<PathBuf, String> {
    let home = PathBuf::from(
        std::env::var_os("HOME")
            .ok_or_else(|| "HOME is unavailable for the FSKit mount root".to_string())?,
    );
    macos_mount_root(&home)
}

#[cfg(target_os = "macos")]
fn macos_mount_root(home: &Path) -> Result<PathBuf, String> {
    if !home.is_absolute() {
        return Err(format!("HOME is not absolute: {}", home.display()));
    }
    Ok(home.join("Library/Application Support/greppy/workspace-mount"))
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
fn systemd_quote(value: &Path) -> Result<String, String> {
    let value = value
        .to_str()
        .ok_or_else(|| format!("systemd path is not UTF-8: {}", value.display()))?;
    if value.contains(['\n', '\r']) {
        return Err("systemd path contains a line break".into());
    }
    Ok(format!(
        "\"{}\"",
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('%', "%%")
    ))
}

#[cfg(target_os = "linux")]
fn render_systemd_user_unit(
    adapter: &Path,
    data_root: &Path,
    mount_root: &Path,
) -> Result<String, String> {
    Ok(format!(
        "[Unit]\n\
         Description=Greppy portable CoW workspace provider\n\
         After=default.target\n\n\
         [Service]\n\
         Type=simple\n\
         ExecStart={} --data-root {} --mount-root {}\n\
         Restart=on-failure\n\
         RestartSec=2\n\
         KillMode=mixed\n\n\
         [Install]\n\
         WantedBy=default.target\n",
        systemd_quote(adapter)?,
        systemd_quote(data_root)?,
        systemd_quote(mount_root)?,
    ))
}

#[cfg(target_os = "macos")]
fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(target_os = "macos")]
fn render_macos_launch_agent(cli: &Path, data_root: &Path) -> Result<String, String> {
    let cli = cli
        .to_str()
        .ok_or_else(|| format!("CLI path is not UTF-8: {}", cli.display()))?;
    let data_root = data_root
        .to_str()
        .ok_or_else(|| format!("workspace path is not UTF-8: {}", data_root.display()))?;
    Ok(format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
         <plist version=\"1.0\">\n\
         <dict>\n\
           <key>Label</key><string>ai.metric-space.greppy.workspace</string>\n\
           <key>ProgramArguments</key>\n\
           <array><string>{}</string><string>workspace</string><string>setup</string></array>\n\
           <key>EnvironmentVariables</key>\n\
           <dict><key>GREPPY_WORKSPACE_DIR</key><string>{}</string></dict>\n\
           <key>RunAtLoad</key><true/>\n\
           <key>ProcessType</key><string>Background</string>\n\
           <key>ThrottleInterval</key><integer>5</integer>\n\
         </dict>\n\
         </plist>\n",
        xml_escape(cli),
        xml_escape(data_root),
    ))
}

#[cfg(any(target_os = "windows", test))]
fn windows_quote_command_argument(value: &str) -> Result<String, String> {
    if value.contains('\0') {
        return Err("Windows command argument contains NUL".into());
    }
    let mut quoted = String::from("\"");
    let mut backslashes = 0usize;
    for character in value.chars() {
        if character == '\\' {
            backslashes += 1;
            continue;
        }
        if character == '"' {
            quoted.push_str(&"\\".repeat(backslashes * 2 + 1));
            quoted.push('"');
        } else {
            quoted.push_str(&"\\".repeat(backslashes));
            quoted.push(character);
        }
        backslashes = 0;
    }
    quoted.push_str(&"\\".repeat(backslashes * 2));
    quoted.push('"');
    Ok(quoted)
}

#[cfg(any(target_os = "windows", test))]
fn render_windows_machine_autostart(cli: &Path) -> Result<String, String> {
    let cli = cli
        .to_str()
        .ok_or_else(|| format!("CLI path is not Unicode: {}", cli.display()))?;
    Ok(format!(
        "{} workspace setup",
        windows_quote_command_argument(cli)?
    ))
}

#[cfg(target_os = "windows")]
fn read_windows_machine_autostart() -> Result<String, String> {
    use windows_sys::Win32::Foundation::{ERROR_MORE_DATA, ERROR_SUCCESS};
    use windows_sys::Win32::System::Registry::{RegGetValueW, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ};

    let key: Vec<u16> = "Software\\Microsoft\\Windows\\CurrentVersion\\Run\0"
        .encode_utf16()
        .collect();
    let name: Vec<u16> = "GreppyWorkspaceProvider\0".encode_utf16().collect();
    let mut bytes = 0u32;
    let first = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut bytes,
        )
    };
    if first != ERROR_SUCCESS && first != ERROR_MORE_DATA {
        return Err(format!(
            "Greppy machine autostart registry value is unavailable (Win32 error {first})"
        ));
    }
    if bytes < 2 || bytes % 2 != 0 {
        return Err(format!(
            "Greppy machine autostart registry value has invalid byte length {bytes}"
        ));
    }
    let mut value = vec![0u16; bytes as usize / 2];
    let second = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_SZ,
            std::ptr::null_mut(),
            value.as_mut_ptr().cast(),
            &mut bytes,
        )
    };
    if second != ERROR_SUCCESS {
        return Err(format!(
            "cannot read Greppy machine autostart registry value (Win32 error {second})"
        ));
    }
    if let Some(end) = value.iter().position(|unit| *unit == 0) {
        value.truncate(end);
    }
    String::from_utf16(&value)
        .map_err(|_| "Greppy machine autostart registry value is not valid UTF-16".into())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn atomic_write_autostart(path: &Path, contents: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("autostart path has no parent: {}", path.display()))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("autostart filename is not UTF-8: {}", path.display()))?;
    let mut temporary = tempfile::Builder::new()
        .prefix(&format!(".{name}."))
        .suffix(".tmp")
        .tempfile_in(parent)
        .map_err(|error| {
            format!(
                "cannot create temporary file in {}: {error}",
                parent.display()
            )
        })?;
    temporary.write_all(contents.as_bytes()).map_err(|error| {
        format!(
            "cannot write temporary file for {}: {error}",
            path.display()
        )
    })?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("cannot sync temporary file for {}: {error}", path.display()))?;
    temporary.persist(path).map_err(|error| {
        format!(
            "cannot atomically publish autostart file {}: {}",
            path.display(),
            error.error
        )
    })?;
    Ok(())
}

#[cfg(target_os = "linux")]
fn install_platform_autostart(data_root: &Path, mount_root: &Path) -> Result<(), String> {
    use std::os::unix::fs::symlink;

    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let adapter = sibling(&current, "greppy-workspace-provider")?;
    require_bundled_file(&adapter, "Linux FUSE3 adapter")?;
    let config_root = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(value) => PathBuf::from(value),
        None => PathBuf::from(std::env::var_os("HOME").ok_or_else(|| {
            "HOME is unavailable for systemd user service installation".to_string()
        })?)
        .join(".config"),
    };
    if !config_root.is_absolute() {
        return Err(format!(
            "systemd user configuration root is not absolute: {}",
            config_root.display()
        ));
    }
    let unit_root = config_root.join("systemd/user");
    let unit_name = "greppy-workspace-provider.service";
    fs::create_dir_all(&unit_root)
        .map_err(|error| format!("cannot create {}: {error}", unit_root.display()))?;
    let unit_path = unit_root.join(unit_name);
    atomic_write_autostart(
        &unit_path,
        &render_systemd_user_unit(&adapter, data_root, mount_root)?,
    )?;
    let wants = unit_root.join("default.target.wants");
    fs::create_dir_all(&wants)
        .map_err(|error| format!("cannot create {}: {error}", wants.display()))?;
    let enabled = wants.join(unit_name);
    match fs::read_link(&enabled) {
        Ok(target) if target == Path::new("../greppy-workspace-provider.service") => {}
        Ok(target) => {
            return Err(format!(
                "existing Greppy systemd enablement {} targets unexpected {}",
                enabled.display(),
                target.display()
            ));
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            symlink("../greppy-workspace-provider.service", &enabled)
                .map_err(|error| format!("cannot enable {}: {error}", unit_path.display()))?;
        }
        Err(error) => return Err(format!("cannot inspect {}: {error}", enabled.display())),
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn install_platform_autostart(data_root: &Path, _mount_root: &Path) -> Result<(), String> {
    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let home = PathBuf::from(
        std::env::var_os("HOME")
            .ok_or_else(|| "HOME is unavailable for LaunchAgent installation".to_string())?,
    );
    if !home.is_absolute() {
        return Err(format!("HOME is not absolute: {}", home.display()));
    }
    let agents = home.join("Library/LaunchAgents");
    fs::create_dir_all(&agents)
        .map_err(|error| format!("cannot create {}: {error}", agents.display()))?;
    let plist = agents.join("ai.metric-space.greppy.workspace.plist");
    atomic_write_autostart(&plist, &render_macos_launch_agent(&current, data_root)?)?;
    Ok(())
}

#[cfg(target_os = "windows")]
fn install_platform_autostart(_data_root: &Path, _mount_root: &Path) -> Result<(), String> {
    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let adapter = sibling(&current, "greppy-workspace-provider.exe")?;
    let runtime = sibling(&current, "greppyworkspacefsp-x64.dll")?;
    let driver = sibling(&current, "greppyworkspacefsp-x64.sys")?;
    require_bundled_file(&adapter, "Windows workspace provider")?;
    require_bundled_file(&runtime, "Greppy WinFsp runtime")?;
    require_bundled_file(&driver, "Greppy WinFsp driver")?;
    let expected = render_windows_machine_autostart(&current)?;
    let actual = read_windows_machine_autostart()?;
    if actual != expected {
        return Err(format!(
            "Greppy machine autostart is not bound to this installation; expected {expected:?}, found {actual:?}; repair the Greppy MSI"
        ));
    }
    Ok(())
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
    let mount_root_text = mount_root
        .to_str()
        .ok_or_else(|| format!("macOS mount root is not UTF-8: {}", mount_root.display()))?;
    atomic_write_autostart(
        &data_root.join("mount-root"),
        &format!("{mount_root_text}\n"),
    )
    .map_err(|error| format!("cannot publish the FSKit mount-root contract: {error}"))?;
    let current = std::env::current_exe()
        .map_err(|error| format!("cannot locate the greppy executable: {error}"))?;
    let app = locate_macos_app(&current)?;
    require_bundled_file(&app, "signed FSKit application")?;
    validate_macos_fskit_installation(&app)?;
    match macos_fskit_extension_status(&app) {
        Ok(MacosFsKitExtensionStatus::Enabled) => {}
        Ok(MacosFsKitExtensionStatus::UnavailableToFsKit) => {
            record_macos_fskit_activation_required(data_root)?;
            return Err(format!(
                "Greppy Workspace FS is registered in pluginkit, but FSKit does not enumerate this extension. A checked switch is not proof that macOS can mount it. No mount was attempted. Capture native status with `{}/Contents/MacOS/GreppyWorkspaceFS --fskit-status` and inspect the macOS fskitd logs; do not repeatedly reinstall the app or toggle the switch without resolving this registration mismatch",
                app.display()
            ));
        }
        Ok(status) => {
            record_macos_fskit_activation_required(data_root)?;
            open_macos_fskit_settings(&app)?;
            return Err(format!(
                "Greppy Workspace FS is {} for this exact application bundle; macOS may require approval again after installing or updating Greppy. Enable `Greppy Workspace FS` in the File System Extensions dialog that was opened, then rerun `greppy workspace setup`",
                status.description()
            ));
        }
        Err(error) => {
            record_macos_fskit_activation_required(data_root)?;
            return Err(format!(
                "cannot verify native FSKit availability for this exact application bundle: {error}; no mount was attempted. Use the matching notarized CLI and GreppyWorkspaceFS.app bundle, then rerun `greppy workspace setup`"
            ));
        }
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
    record_macos_fskit_activation_required(data_root)?;
    let _ = open_macos_fskit_settings(&app);
    Err(
        "macOS did not mount Greppy Workspace FS; its File System Extensions dialog was opened. Enable `Greppy Workspace FS` (approval may be required again after an update), then rerun `greppy workspace setup`"
            .into(),
    )
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MacosFsKitExtensionStatus {
    Enabled,
    Disabled,
    Missing,
    StaleRegistration,
    UnavailableToFsKit,
}

#[cfg(target_os = "macos")]
impl MacosFsKitExtensionStatus {
    fn description(self) -> &'static str {
        match self {
            Self::Enabled => "enabled",
            Self::Disabled => "disabled",
            Self::Missing => "not registered",
            Self::StaleRegistration => "registered without a valid version",
            Self::UnavailableToFsKit => "not enumerated by FSKit",
        }
    }
}

#[cfg(target_os = "macos")]
fn parse_macos_fskit_extension_status(
    exit_code: Option<i32>,
    stdout: &[u8],
    stderr: &[u8],
    expected_extension: &Path,
) -> Result<MacosFsKitExtensionStatus, String> {
    if exit_code != Some(0) {
        let diagnostic = String::from_utf8_lossy(stderr).trim().to_string();
        return Err(format!(
            "pluginkit exited with {}{}",
            exit_code
                .map(|code| code.to_string())
                .unwrap_or_else(|| "a signal".into()),
            if diagnostic.is_empty() {
                String::new()
            } else {
                format!(": {diagnostic}")
            }
        ));
    }
    const BUNDLE_ID: &str = "ai.metricspace.greppy.workspacefs.extension";
    let expected =
        fs::canonicalize(expected_extension).unwrap_or_else(|_| expected_extension.to_path_buf());
    let output = String::from_utf8_lossy(stdout);
    let mut saw_bundle = false;
    let mut matching_line = None;
    for line in output.lines() {
        if !line.contains(BUNDLE_ID) {
            continue;
        }
        saw_bundle = true;
        if line.split('\t').next_back().is_some_and(|path| {
            let registered = Path::new(path.trim());
            fs::canonicalize(registered).unwrap_or_else(|_| registered.to_path_buf()) == expected
        }) {
            matching_line = Some(line.trim_start());
            break;
        }
    }
    let Some(line) = matching_line else {
        return Ok(if saw_bundle {
            MacosFsKitExtensionStatus::StaleRegistration
        } else {
            MacosFsKitExtensionStatus::Missing
        });
    };
    if line.contains("((null))") {
        return Ok(MacosFsKitExtensionStatus::StaleRegistration);
    }
    if line.starts_with('+') {
        Ok(MacosFsKitExtensionStatus::Enabled)
    } else {
        Ok(MacosFsKitExtensionStatus::Disabled)
    }
}

#[cfg(target_os = "macos")]
fn macos_fskit_extension_status(app: &Path) -> Result<MacosFsKitExtensionStatus, String> {
    let expected_extension = app.join("Contents/Extensions/GreppyWorkspaceFS.appex");
    require_bundled_file(&expected_extension, "FSKit extension")?;
    let output = Command::new("/usr/bin/pluginkit")
        .args([
            "-m",
            "-A",
            "-D",
            "-v",
            "-i",
            "ai.metricspace.greppy.workspacefs.extension",
        ])
        .output()
        .map_err(|error| format!("cannot query FSKit registration with pluginkit: {error}"))?;
    let registration = parse_macos_fskit_extension_status(
        output.status.code(),
        &output.stdout,
        &output.stderr,
        &expected_extension,
    )?;
    if registration != MacosFsKitExtensionStatus::Enabled {
        return Ok(registration);
    }
    // pluginkit's election and FSKit's mount eligibility are different state.
    // Ask the signed host through the public FSClient API before allocating a
    // disk image or attempting a mount.
    let protocol = Command::new("/usr/libexec/PlistBuddy")
        .args(["-c", "Print :GreppyFSKitStatusProtocolVersion"])
        .arg(app.join("Contents/Info.plist"))
        .output()
        .map_err(|error| format!("cannot inspect FSKit status-helper contract: {error}"))?;
    if !protocol.status.success() || protocol.stdout.trim_ascii() != b"1" {
        return Err("installed app lacks the native FSKit status helper; install the matching notarized app bundle (the existing app was not modified)".into());
    }
    let helper = app.join("Contents/MacOS/GreppyWorkspaceFS");
    require_bundled_file(&helper, "native FSKit status helper")?;
    let output = Command::new(&helper)
        .arg("--fskit-status")
        .output()
        .map_err(|error| format!("cannot invoke native FSKit status helper: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "native FSKit status helper exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    parse_native_fskit_status(&output.stdout, &expected_extension)
}

#[cfg(target_os = "macos")]
fn parse_native_fskit_status(
    bytes: &[u8],
    expected_extension: &Path,
) -> Result<MacosFsKitExtensionStatus, String> {
    #[derive(serde::Deserialize)]
    struct Module {
        bundle_id: String,
        path: PathBuf,
        enabled: bool,
    }
    #[derive(serde::Deserialize)]
    struct Status {
        schema: String,
        modules: Vec<Module>,
    }
    let status: Status = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid native FSKit status: {error}"))?;
    if status.schema != "greppy.fskit-status.v1" {
        return Err("unsupported native FSKit status schema".into());
    }
    let modules: Vec<_> = status
        .modules
        .iter()
        .filter(|module| module.bundle_id == "ai.metricspace.greppy.workspacefs.extension")
        .collect();
    let [module] = modules.as_slice() else {
        return if modules.is_empty() {
            Ok(MacosFsKitExtensionStatus::UnavailableToFsKit)
        } else {
            Err(
                "FSKit enumerated multiple Greppy extensions; exact registration is ambiguous"
                    .into(),
            )
        };
    };
    let expected =
        fs::canonicalize(expected_extension).unwrap_or_else(|_| expected_extension.to_path_buf());
    let actual = fs::canonicalize(&module.path).unwrap_or_else(|_| module.path.clone());
    if actual != expected {
        return Ok(MacosFsKitExtensionStatus::StaleRegistration);
    }
    Ok(if module.enabled {
        MacosFsKitExtensionStatus::Enabled
    } else {
        MacosFsKitExtensionStatus::Disabled
    })
}

#[cfg(target_os = "macos")]
fn open_macos_fskit_settings(app: &Path) -> Result<(), String> {
    let status = Command::new("/usr/bin/open")
        .arg("-n")
        .arg(app)
        .status()
        .map_err(|error| format!("cannot open {}: {error}", app.display()))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "macOS refused to open the File System Extensions dialog through {}",
            app.display()
        ))
    }
}

#[cfg(target_os = "macos")]
fn record_macos_fskit_activation_required(data_root: &Path) -> Result<(), String> {
    fs::write(data_root.join("activation-required"), b"fskit\n")
        .map_err(|error| format!("cannot record FSKit activation state: {error}"))
}

#[cfg(target_os = "macos")]
fn locate_macos_app(current_exe: &Path) -> Result<PathBuf, String> {
    locate_macos_app_with_fallback(
        current_exe,
        Path::new("/Applications/GreppyWorkspaceFS.app"),
    )
}

#[cfg(target_os = "macos")]
fn locate_macos_app_with_fallback(
    current_exe: &Path,
    installed_app: &Path,
) -> Result<PathBuf, String> {
    if let Some(bundle) = current_exe.ancestors().find(|path| {
        path.file_name()
            .is_some_and(|name| name == "GreppyWorkspaceFS.app")
    }) {
        return Ok(bundle.to_path_buf());
    }
    let bundled_app = sibling(current_exe, "GreppyWorkspaceFS.app")?;
    if bundled_app.is_dir() {
        return Ok(bundled_app);
    }
    if installed_app.is_dir() {
        return Ok(installed_app.to_path_buf());
    }
    Err(format!(
        "signed FSKit application is missing beside {} and at {}; install the notarized GreppyWorkspaceFS.app in /Applications or place it beside this development binary, then rerun `greppy workspace setup`",
        current_exe.display(),
        installed_app.display()
    ))
}

#[cfg(target_os = "macos")]
fn macos_fskit_profile_paths(app: &Path) -> [(PathBuf, &'static str); 2] {
    [
        (
            app.join("Contents/embedded.provisionprofile"),
            "FSKit host application provisioning profile",
        ),
        (
            app.join(
                "Contents/Extensions/GreppyWorkspaceFS.appex/Contents/embedded.provisionprofile",
            ),
            "FSKit extension provisioning profile",
        ),
    ]
}

#[cfg(target_os = "macos")]
fn require_macos_fskit_profiles(app: &Path) -> Result<[PathBuf; 2], String> {
    let mut accepted = Vec::with_capacity(2);
    for (profile, label) in macos_fskit_profile_paths(app) {
        let metadata = fs::symlink_metadata(&profile).map_err(|error| {
            format!(
                "incomplete Greppy FSKit installation: {label} is missing at {}: {error}; reinstall the signed Greppy package",
                profile.display()
            )
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!(
                "incomplete Greppy FSKit installation: {label} must be a regular embedded file at {}; reinstall the signed Greppy package",
                profile.display()
            ));
        }
        accepted.push(profile);
    }
    accepted.try_into().map_err(|_| {
        "internal error while validating the two required Greppy FSKit profiles".to_string()
    })
}

#[cfg(target_os = "macos")]
fn validate_macos_fskit_installation(app: &Path) -> Result<(), String> {
    let profiles = require_macos_fskit_profiles(app)?;
    for profile in profiles {
        let decoded = Command::new("/usr/bin/security")
            .arg("cms")
            .arg("-D")
            .arg("-i")
            .arg(&profile)
            .output()
            .map_err(|error| {
                format!(
                    "cannot validate embedded FSKit provisioning profile {}: {error}",
                    profile.display()
                )
            })?;
        if !decoded.status.success() {
            return Err(format!(
                "invalid embedded FSKit provisioning profile at {}: {}; reinstall the signed Greppy package",
                profile.display(),
                String::from_utf8_lossy(&decoded.stderr).trim()
            ));
        }
    }

    let verified = Command::new("/usr/bin/codesign")
        .arg("--verify")
        .arg("--deep")
        .arg("--strict")
        .arg(app)
        .output()
        .map_err(|error| format!("cannot verify Greppy FSKit code signature: {error}"))?;
    if !verified.status.success() {
        return Err(format!(
            "Greppy FSKit code signature is invalid: {}; reinstall the signed Greppy package",
            String::from_utf8_lossy(&verified.stderr).trim()
        ));
    }

    let assessed = Command::new("/usr/sbin/spctl")
        .arg("--assess")
        .arg("--type")
        .arg("execute")
        .arg(app)
        .output()
        .map_err(|error| format!("cannot run Gatekeeper assessment for Greppy FSKit: {error}"))?;
    if !assessed.status.success() {
        return Err(format!(
            "Gatekeeper rejected Greppy FSKit: {}; install the notarized Greppy package before activating the extension",
            String::from_utf8_lossy(&assessed.stderr).trim()
        ));
    }
    Ok(())
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
    #[cfg(not(target_os = "macos"))]
    fn mount_is_a_sibling_not_an_alias_of_private_data() {
        let data = Path::new("/tmp/greppy-test/workspace");
        assert_eq!(
            mount_root(data).unwrap(),
            Path::new("/tmp/greppy-test/workspace-mount")
        );
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macos_mount_is_outside_the_application_group_container() {
        assert_eq!(
            macos_mount_root(Path::new("/Users/greppy")).unwrap(),
            Path::new("/Users/greppy/Library/Application Support/greppy/workspace-mount")
        );
    }

    #[test]
    fn bundled_adapter_is_resolved_beside_the_cli() {
        assert_eq!(
            sibling(Path::new("/opt/greppy/bin/greppy"), "provider").unwrap(),
            Path::new("/opt/greppy/bin/provider")
        );
    }

    #[test]
    fn windows_machine_autostart_quotes_spaces_and_trailing_backslashes() {
        assert_eq!(
            render_windows_machine_autostart(Path::new(r"C:\Program Files\Greppy\greppy.exe"))
                .unwrap(),
            r#""C:\Program Files\Greppy\greppy.exe" workspace setup"#
        );
        assert_eq!(
            windows_quote_command_argument(r"C:\Greppy\").unwrap(),
            r#""C:\Greppy\\""#
        );
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn autostart_publish_replaces_a_symlink_without_following_it() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let outside = root.path().join("outside");
        let target = root.path().join("autostart");
        fs::write(&outside, "protected").unwrap();
        symlink(&outside, &target).unwrap();

        atomic_write_autostart(&target, "managed").unwrap();

        assert_eq!(fs::read_to_string(&outside).unwrap(), "protected");
        assert_eq!(fs::read_to_string(&target).unwrap(), "managed");
        assert!(!fs::symlink_metadata(&target)
            .unwrap()
            .file_type()
            .is_symlink());

        fs::write(root.path().join(".autostart.stale.tmp"), "stale").unwrap();
        atomic_write_autostart(&target, "updated").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "updated");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn systemd_user_unit_is_persistent_restartable_and_argument_safe() {
        let unit = render_systemd_user_unit(
            Path::new("/opt/greppy %/bin/greppy-workspace-provider"),
            Path::new("/home/test/workspace data"),
            Path::new("/home/test/workspace mount"),
        )
        .unwrap();
        assert!(unit.contains(
            "ExecStart=\"/opt/greppy %%/bin/greppy-workspace-provider\" \
             --data-root \"/home/test/workspace data\" \
             --mount-root \"/home/test/workspace mount\""
        ));
        assert!(unit.contains("Restart=on-failure"));
        assert!(unit.contains("WantedBy=default.target"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn launch_agent_replays_setup_at_login_with_exact_workspace_root() {
        let plist = render_macos_launch_agent(
            Path::new("/Applications/Greppy & Tools/greppy"),
            Path::new("/Users/test/Greppy & Data"),
        )
        .unwrap();
        assert!(plist.contains("/Applications/Greppy &amp; Tools/greppy"));
        assert!(plist.contains("/Users/test/Greppy &amp; Data"));
        assert!(plist.contains("<string>workspace</string><string>setup</string>"));
        assert!(plist.contains("<key>RunAtLoad</key><true/>"));
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

    #[cfg(target_os = "macos")]
    #[test]
    fn locates_installed_or_sibling_fskit_application() {
        assert_eq!(
            locate_macos_app_with_fallback(
                Path::new("/Applications/GreppyWorkspaceFS.app/Contents/Resources/bin/greppy"),
                Path::new("/unused/GreppyWorkspaceFS.app"),
            )
            .unwrap(),
            Path::new("/Applications/GreppyWorkspaceFS.app")
        );

        let root = tempfile::tempdir().unwrap();
        let debug_dir = root.path().join("target/debug");
        std::fs::create_dir_all(&debug_dir).unwrap();
        let debug_binary = debug_dir.join("greppy");
        let installed = root.path().join("Applications/GreppyWorkspaceFS.app");
        std::fs::create_dir_all(&installed).unwrap();
        assert_eq!(
            locate_macos_app_with_fallback(&debug_binary, &installed).unwrap(),
            installed
        );

        let sibling_app = debug_dir.join("GreppyWorkspaceFS.app");
        std::fs::create_dir_all(&sibling_app).unwrap();
        assert_eq!(
            locate_macos_app_with_fallback(&debug_binary, &installed).unwrap(),
            sibling_app
        );

        std::fs::remove_dir_all(&sibling_app).unwrap();
        std::fs::remove_dir_all(&installed).unwrap();
        let error = locate_macos_app_with_fallback(&debug_binary, &installed).unwrap_err();
        assert!(error.contains("install the notarized GreppyWorkspaceFS.app"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn parses_exact_fskit_extension_approval_states() {
        let expected = Path::new(
            "/Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex",
        );
        assert_eq!(
            parse_macos_fskit_extension_status(
                Some(0),
                b"+    ai.metricspace.greppy.workspacefs.extension(0.4.0)\tUUID\tDATE\t/Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex\n (1 plug-in)\n",
                b"",
                expected,
            )
            .unwrap(),
            MacosFsKitExtensionStatus::Enabled
        );
        assert_eq!(
            parse_macos_fskit_extension_status(
                Some(0),
                b"-    ai.metricspace.greppy.workspacefs.extension(0.4.0)\tUUID\tDATE\t/Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex\n",
                b"",
                expected,
            )
            .unwrap(),
            MacosFsKitExtensionStatus::Disabled
        );
        assert_eq!(
            parse_macos_fskit_extension_status(Some(0), b" (0 plug-ins)\n", b"", expected).unwrap(),
            MacosFsKitExtensionStatus::Missing
        );
        assert_eq!(
            parse_macos_fskit_extension_status(
                Some(0),
                b"+    ai.metricspace.greppy.workspacefs.extension((null))\tUUID\tDATE\t/Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex\n",
                b"",
                expected,
            )
            .unwrap(),
            MacosFsKitExtensionStatus::StaleRegistration
        );
        assert_eq!(
            parse_macos_fskit_extension_status(
                Some(0),
                b"+    ai.metricspace.greppy.workspacefs.extension(0.3.4)\tUUID\tDATE\t/Old/GreppyWorkspaceFS.appex\n",
                b"",
                expected,
            )
            .unwrap(),
            MacosFsKitExtensionStatus::StaleRegistration
        );
        let error =
            parse_macos_fskit_extension_status(Some(1), b"", b"connection invalid", expected)
                .unwrap_err();
        assert!(error.contains("exited with 1"));
        assert!(error.contains("connection invalid"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_fskit_status_requires_exact_available_enabled_module() {
        let expected = Path::new(
            "/Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex",
        );
        let module = serde_json::json!({
            "bundle_id": "ai.metricspace.greppy.workspacefs.extension",
            "path": expected,
            "enabled": true,
        });
        let evaluate = |modules| {
            parse_native_fskit_status(
                &serde_json::to_vec(
                    &serde_json::json!({"schema": "greppy.fskit-status.v1", "modules": modules}),
                )
                .unwrap(),
                expected,
            )
        };
        assert_eq!(
            evaluate(vec![module.clone()]).unwrap(),
            MacosFsKitExtensionStatus::Enabled
        );
        assert_eq!(
            evaluate(Vec::<serde_json::Value>::new()).unwrap(),
            MacosFsKitExtensionStatus::UnavailableToFsKit
        );
        let mut disabled = module.clone();
        disabled["enabled"] = serde_json::json!(false);
        assert_eq!(
            evaluate(vec![disabled]).unwrap(),
            MacosFsKitExtensionStatus::Disabled
        );
        let mut wrong_bundle = module.clone();
        wrong_bundle["path"] = serde_json::json!("/Old/GreppyWorkspaceFS.appex");
        assert_eq!(
            evaluate(vec![wrong_bundle]).unwrap(),
            MacosFsKitExtensionStatus::StaleRegistration
        );
        assert!(evaluate(vec![module.clone(), module]).is_err());
        assert!(parse_native_fskit_status(b"{}", expected).is_err());
        assert!(
            parse_native_fskit_status(br#"{"schema":"unknown","modules":[]}"#, expected).is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn accepts_fskit_registration_through_sibling_app_symlink() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let installed_extension = root
            .path()
            .join("Applications/GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex");
        fs::create_dir_all(&installed_extension).unwrap();
        let bin = root.path().join("bin");
        fs::create_dir_all(&bin).unwrap();
        symlink(
            root.path().join("Applications/GreppyWorkspaceFS.app"),
            bin.join("GreppyWorkspaceFS.app"),
        )
        .unwrap();
        let sibling_extension =
            bin.join("GreppyWorkspaceFS.app/Contents/Extensions/GreppyWorkspaceFS.appex");
        let output = format!(
            "+    ai.metricspace.greppy.workspacefs.extension(0.4.0)\tUUID\tDATE\t{}\n",
            installed_extension.display()
        );

        assert_eq!(
            parse_macos_fskit_extension_status(
                Some(0),
                output.as_bytes(),
                b"",
                &sibling_extension,
            )
            .unwrap(),
            MacosFsKitExtensionStatus::Enabled
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fskit_setup_requires_two_regular_embedded_profiles() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().unwrap();
        let app = root.path().join("GreppyWorkspaceFS.app");
        let paths = macos_fskit_profile_paths(&app);
        for (profile, _) in &paths {
            fs::create_dir_all(profile.parent().unwrap()).unwrap();
            fs::write(profile, b"profile").unwrap();
        }
        let accepted = require_macos_fskit_profiles(&app).unwrap();
        assert_eq!(accepted[0], paths[0].0);
        assert_eq!(accepted[1], paths[1].0);

        fs::remove_file(&paths[1].0).unwrap();
        let outside = root.path().join("outside.provisionprofile");
        fs::write(&outside, b"profile").unwrap();
        symlink(&outside, &paths[1].0).unwrap();
        let error = require_macos_fskit_profiles(&app).unwrap_err();
        assert!(error.contains("must be a regular embedded file"));
        assert!(error.contains("reinstall the signed Greppy package"));
    }
}
