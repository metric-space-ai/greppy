//! Shared contract executed against every real mounted provider.

use greppy_workspace_core::{WorkspaceCore, CHUNK_SIZE};
use memmap2::MmapOptions;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

struct RangeLock<'a> {
    file: &'a File,
    offset: u64,
    length: u64,
}

impl Drop for RangeLock<'_> {
    fn drop(&mut self) {
        unlock_range(self.file, self.offset, self.length).unwrap();
    }
}

#[cfg(unix)]
fn try_lock_range(file: &File, offset: u64, length: u64) -> io::Result<RangeLock<'_>> {
    use std::os::fd::AsRawFd as _;

    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_WRLCK as libc::c_short;
    lock.l_whence = libc::SEEK_SET as libc::c_short;
    lock.l_start = offset.try_into().unwrap();
    lock.l_len = length.try_into().unwrap();
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLK, &lock) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(RangeLock {
        file,
        offset,
        length,
    })
}

#[cfg(unix)]
fn unlock_range(file: &File, offset: u64, length: u64) -> io::Result<()> {
    use std::os::fd::AsRawFd as _;

    let mut lock: libc::flock = unsafe { std::mem::zeroed() };
    lock.l_type = libc::F_UNLCK as libc::c_short;
    lock.l_whence = libc::SEEK_SET as libc::c_short;
    lock.l_start = offset.try_into().unwrap();
    lock.l_len = length.try_into().unwrap();
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETLK, &lock) } == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn overlapped_at(offset: u64) -> windows_sys::Win32::System::IO::OVERLAPPED {
    use windows_sys::Win32::System::IO::OVERLAPPED_0_0;

    let mut overlapped = windows_sys::Win32::System::IO::OVERLAPPED::default();
    overlapped.Anonymous.Anonymous = OVERLAPPED_0_0 {
        Offset: offset as u32,
        OffsetHigh: (offset >> 32) as u32,
    };
    overlapped
}

#[cfg(windows)]
fn try_lock_range(file: &File, offset: u64, length: u64) -> io::Result<RangeLock<'_>> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{
        LockFileEx, LOCKFILE_EXCLUSIVE_LOCK, LOCKFILE_FAIL_IMMEDIATELY,
    };

    let mut overlapped = overlapped_at(offset);
    let result = unsafe {
        LockFileEx(
            file.as_raw_handle(),
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
            0,
            length as u32,
            (length >> 32) as u32,
            &mut overlapped,
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(RangeLock {
        file,
        offset,
        length,
    })
}

#[cfg(windows)]
fn unlock_range(file: &File, offset: u64, length: u64) -> io::Result<()> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::UnlockFileEx;

    let mut overlapped = overlapped_at(offset);
    let result = unsafe {
        UnlockFileEx(
            file.as_raw_handle(),
            0,
            length as u32,
            (length >> 32) as u32,
            &mut overlapped,
        )
    };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
fn windows_link_count(file: &File) -> io::Result<u32> {
    use std::os::windows::io::AsRawHandle as _;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };

    let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    let result = unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(information.nNumberOfLinks)
}

#[test]
fn mounted_contract_range_lock_helper() {
    let Some(path) = std::env::var_os("GREPPY_RANGE_LOCK_FILE") else {
        return;
    };
    let ready = std::env::var_os("GREPPY_RANGE_LOCK_READY").unwrap();
    let release = std::env::var_os("GREPPY_RANGE_LOCK_RELEASE").unwrap();
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    let _lock = try_lock_range(&file, 16, 16).unwrap();
    fs::write(ready, b"ready").unwrap();
    let started = Instant::now();
    while !Path::new(&release).exists() {
        assert!(
            started.elapsed() < Duration::from_secs(15),
            "parent never released the byte-range-lock helper"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn exercise_byte_range_lock_contract(root: &Path) {
    let path = root.join("contract-range-lock.bin");
    let ready = root.join("contract-range-lock.ready");
    let release = root.join("contract-range-lock.release");
    fs::write(&path, [0_u8; 128]).unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "mounted_contract::mounted_contract_range_lock_helper",
            "--nocapture",
        ])
        .env("GREPPY_RANGE_LOCK_FILE", &path)
        .env("GREPPY_RANGE_LOCK_READY", &ready)
        .env("GREPPY_RANGE_LOCK_RELEASE", &release)
        .spawn()
        .unwrap();
    let started = Instant::now();
    while !ready.exists() {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "byte-range-lock helper did not become ready"
        );
        thread::sleep(Duration::from_millis(10));
    }

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    assert!(
        try_lock_range(&file, 24, 4).is_err(),
        "an overlapping byte-range lock from another process must conflict"
    );
    drop(try_lock_range(&file, 40, 4).unwrap());

    fs::write(&release, b"release").unwrap();
    assert!(child.wait().unwrap().success());
    drop(try_lock_range(&file, 24, 4).unwrap());
}

pub fn exercise_mounted_contract(root: &Path, core: &WorkspaceCore) {
    let regular = root.join("contract-regular.bin");
    fs::write(&regular, b"alpha").unwrap();
    assert!(
        fs::write(root.join("contract-ads.txt:stream"), b"forbidden").is_err(),
        "the mounted portable namespace must reject Windows alternate data streams"
    );
    OpenOptions::new()
        .append(true)
        .open(&regular)
        .unwrap()
        .write_all(b"-beta")
        .unwrap();
    assert_eq!(fs::read(&regular).unwrap(), b"alpha-beta");
    OpenOptions::new()
        .write(true)
        .open(&regular)
        .unwrap()
        .set_len(5)
        .unwrap();
    assert_eq!(fs::read(&regular).unwrap(), b"alpha");

    let mapped = root.join("contract-mmap.bin");
    fs::write(&mapped, vec![3_u8; CHUNK_SIZE * 2]).unwrap();
    let before = core.chunks().stats().unwrap();
    let mapped_file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&mapped)
        .unwrap();
    let mut mapping = unsafe { MmapOptions::new().map_mut(&mapped_file).unwrap() };
    mapping[CHUNK_SIZE + 31] = 8;
    mapping.flush().unwrap();
    drop(mapping);
    mapped_file.sync_all().unwrap();
    let after = core.chunks().stats().unwrap();
    assert_eq!(after.chunk_count, before.chunk_count + 1);
    let mut check = File::open(&mapped).unwrap();
    check.seek(SeekFrom::Start(CHUNK_SIZE as u64 + 31)).unwrap();
    let mut byte = [0_u8; 1];
    check.read_exact(&mut byte).unwrap();
    assert_eq!(byte, [8]);

    let locked = root.join("contract-lock.bin");
    fs::write(&locked, b"lock").unwrap();
    let first = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&locked)
        .unwrap();
    let second = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&locked)
        .unwrap();
    first.try_lock().unwrap();
    assert!(second.try_lock().is_err());
    first.unlock().unwrap();
    second.try_lock().unwrap();
    second.unlock().unwrap();
    exercise_byte_range_lock_contract(root);

    let directory = root.join("contract-directory");
    let nested = directory.join("segment-0000000000000001/segment-0000000000000002/segment-0000000000000003/segment-0000000000000004/segment-0000000000000005/segment-0000000000000006");
    fs::create_dir_all(&nested).unwrap();
    let long_path = nested.join("long-file-name-ä-東京.txt");
    fs::write(&long_path, b"long path").unwrap();
    assert_eq!(fs::read(&long_path).unwrap(), b"long path");
    assert!(
        fs::remove_dir(&directory).is_err(),
        "removing a non-empty directory must fail"
    );

    let replace_source = root.join("contract-replace-source.txt");
    let replace_destination = root.join("contract-replace-destination.txt");
    fs::write(&replace_source, b"new").unwrap();
    fs::write(&replace_destination, b"old").unwrap();
    fs::rename(&replace_source, &replace_destination).unwrap();
    assert_eq!(fs::read(&replace_destination).unwrap(), b"new");
    assert!(!replace_source.exists());

    let enumerated = root.join("contract-enumeration");
    fs::create_dir(&enumerated).unwrap();
    let writer_root = enumerated.clone();
    let writer = thread::spawn(move || {
        for index in 0..64 {
            let path = writer_root.join(format!("entry-{index:03}.txt"));
            fs::write(&path, index.to_string()).unwrap();
            if index % 2 == 0 {
                fs::remove_file(path).unwrap();
            }
        }
    });
    while !writer.is_finished() {
        for entry in fs::read_dir(&enumerated).unwrap() {
            let name = entry.unwrap().file_name();
            assert!(name.to_string_lossy().starts_with("entry-"));
        }
    }
    writer.join().unwrap();
    let mut final_entries = fs::read_dir(&enumerated)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect::<Vec<_>>();
    final_entries.sort();
    assert_eq!(final_entries.len(), 32);
    assert!(final_entries
        .iter()
        .all(|name| name.to_string_lossy().starts_with("entry-")));

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let executable = root.join("contract-executable.sh");
        fs::write(&executable, b"#!/bin/sh\nexit 0\n").unwrap();
        let mut permissions = fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&executable, permissions).unwrap();
        assert_eq!(
            fs::metadata(&executable).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }

    let missing = root.join("contract-missing.txt");
    assert_eq!(
        fs::read(&missing).unwrap_err().kind(),
        io::ErrorKind::NotFound
    );

    let open_source = root.join("contract-open-source.txt");
    let open_destination = root.join("contract-open-destination.txt");
    fs::write(&open_source, b"before").unwrap();
    let mut open_handle = OpenOptions::new()
        .read(true)
        .write(true)
        .open(&open_source)
        .unwrap();
    fs::rename(&open_source, &open_destination).unwrap();
    open_handle.seek(SeekFrom::End(0)).unwrap();
    open_handle.write_all(b"-after").unwrap();
    open_handle.sync_all().unwrap();
    assert_eq!(fs::read(&open_destination).unwrap(), b"before-after");
    fs::remove_file(&open_destination).unwrap();
    open_handle.seek(SeekFrom::End(0)).unwrap();
    open_handle.write_all(b"-unlinked").unwrap();
    open_handle.seek(SeekFrom::Start(0)).unwrap();
    let mut unlinked_contents = Vec::new();
    open_handle.read_to_end(&mut unlinked_contents).unwrap();
    assert_eq!(unlinked_contents, b"before-after-unlinked");
    drop(open_handle);
    assert!(!open_destination.exists());

    let unicode = root.join("contract-ä-東京.txt");
    fs::write(&unicode, b"unicode").unwrap();
    assert!(fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .any(|name| name == "contract-ä-東京.txt"));

    let lowercase = root.join("contract-case.txt");
    let uppercase = root.join("CONTRACT-CASE.TXT");
    fs::write(&lowercase, b"lowercase").unwrap();
    #[cfg(unix)]
    {
        fs::write(&uppercase, b"uppercase").unwrap();
        assert_eq!(fs::read(&lowercase).unwrap(), b"lowercase");
        assert_eq!(fs::read(&uppercase).unwrap(), b"uppercase");
    }
    #[cfg(windows)]
    {
        assert_eq!(fs::read(&uppercase).unwrap(), b"lowercase");
        let error = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&uppercase)
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        let renamed_case = root.join("Contract-Case.txt");
        fs::rename(&lowercase, &renamed_case).unwrap();
        assert_eq!(fs::read(&uppercase).unwrap(), b"lowercase");
        assert!(fs::read_dir(root)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .any(|name| name == "Contract-Case.txt"));
    }

    let link_source = root.join("contract-link-source.txt");
    let hard_link = root.join("contract-hard-link.txt");
    fs::write(&link_source, b"shared").unwrap();
    fs::hard_link(&link_source, &hard_link).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        assert_eq!(fs::metadata(&link_source).unwrap().nlink(), 2);
        assert_eq!(
            fs::metadata(&link_source).unwrap().ino(),
            fs::metadata(&hard_link).unwrap().ino()
        );
    }
    #[cfg(windows)]
    assert_eq!(
        windows_link_count(&File::open(&link_source).unwrap()).unwrap(),
        2
    );
    fs::write(&hard_link, b"updated").unwrap();
    assert_eq!(fs::read(&link_source).unwrap(), b"updated");
    fs::remove_file(&link_source).unwrap();
    assert_eq!(fs::read(&hard_link).unwrap(), b"updated");
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        assert_eq!(fs::metadata(&hard_link).unwrap().nlink(), 1);
    }
    #[cfg(windows)]
    assert_eq!(
        windows_link_count(&File::open(&hard_link).unwrap()).unwrap(),
        1
    );
    fs::remove_file(&hard_link).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let symbolic = root.join("contract-symbolic-link.txt");
        symlink("contract-regular.bin", &symbolic).unwrap();
        assert_eq!(
            fs::read_link(&symbolic).unwrap(),
            Path::new("contract-regular.bin")
        );
        assert!(
            symlink(
                "../contract-outside.txt",
                root.join("contract-escape-link.txt")
            )
            .is_err(),
            "a mounted symlink must not escape the workspace root"
        );
        assert!(
            symlink("/etc/passwd", root.join("contract-absolute-link.txt")).is_err(),
            "a mounted symlink must not target the host namespace"
        );
    }

    #[cfg(windows)]
    {
        use std::os::windows::fs::symlink_file;
        let symbolic = root.join("contract-symbolic-link.txt");
        symlink_file("contract-regular.bin", &symbolic).unwrap();
        assert_eq!(
            fs::read_link(&symbolic).unwrap(),
            Path::new("contract-regular.bin")
        );
        assert!(
            symlink_file(
                r"..\contract-outside.txt",
                root.join("contract-escape-link.txt")
            )
            .is_err(),
            "a mounted symlink must not escape the workspace root"
        );
        assert!(
            symlink_file(
                r"C:\Windows\System32",
                root.join("contract-absolute-link.txt")
            )
            .is_err(),
            "a mounted symlink must not target the host namespace"
        );
    }
}
