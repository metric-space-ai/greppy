//! Shared contract executed against every real mounted provider.

use greppy_workspace_core::{WorkspaceCore, CHUNK_SIZE};
use memmap2::MmapOptions;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub fn exercise_mounted_contract(root: &Path, core: &WorkspaceCore) {
    let regular = root.join("contract-regular.bin");
    fs::write(&regular, b"alpha").unwrap();
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
    fs::write(&hard_link, b"updated").unwrap();
    assert_eq!(fs::read(&link_source).unwrap(), b"updated");
    fs::remove_file(&link_source).unwrap();
    assert_eq!(fs::read(&hard_link).unwrap(), b"updated");
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
    }
}
