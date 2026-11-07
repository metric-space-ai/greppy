//! One session owns one staging allowance across all of its contexts/pages.
//! This bounds staged files, not Servo's already-received network body.
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug)]
pub struct DownloadBudget {
    state: Mutex<(u64, u64)>, // configured limit, reserved live file bytes
}
impl DownloadBudget {
    pub fn new(limit: u64) -> Self { Self { state: Mutex::new((limit, 0)) } }
    pub fn set_limit(&self, limit: u64) {
        self.state.lock().unwrap_or_else(|error| error.into_inner()).0 = limit;
    }
    fn reserve(self: &Arc<Self>, bytes: u64) -> Option<DownloadReservation> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let next = state.1.checked_add(bytes)?;
        if next > state.0 { return None; }
        state.1 = next;
        Some(DownloadReservation { budget: Arc::clone(self), bytes })
    }
}
#[derive(Debug)]
struct DownloadReservation { budget: Arc<DownloadBudget>, bytes: u64 }
impl Drop for DownloadReservation {
    fn drop(&mut self) {
        let mut state = self.budget.state.lock().unwrap_or_else(|error| error.into_inner());
        state.1 -= self.bytes;
    }
}

/// Weak lookup never owns a session or its file leases.
#[derive(Default)]
pub struct DownloadBudgets { sessions: HashMap<String, Weak<DownloadBudget>> }
impl DownloadBudgets {
    pub fn for_session(&mut self, session: &str, limit: u64) -> Arc<DownloadBudget> {
        self.sessions.retain(|_, budget| budget.strong_count() > 0);
        if let Some(budget) = self.sessions.get(session).and_then(Weak::upgrade) {
            budget.set_limit(limit);
            return budget;
        }
        let budget = Arc::new(DownloadBudget::new(limit));
        self.sessions.insert(session.to_owned(), Arc::downgrade(&budget));
        budget
    }
}

#[derive(Debug)]
pub struct CapturedDownload {
    pub path: Option<PathBuf>,
    pub failure: Option<String>,
    _reservation: Option<DownloadReservation>,
}
impl CapturedDownload {
    /// The allocator must create a new private writable file. It is never
    /// called unless the whole body fits the session's remaining allowance.
    pub fn stage(
        body: &[u8],
        budget: Option<Arc<DownloadBudget>>,
        allocate: impl FnOnce() -> io::Result<(PathBuf, File)>,
    ) -> Arc<Self> {
        let reservation = budget.and_then(|budget| budget.reserve(body.len() as u64));
        let Some(reservation) = reservation else {
            return Arc::new(Self { path: None,
                failure: Some("resource_limit: download staging exceeds session byte budget".into()),
                _reservation: None });
        };
        let result = allocate().and_then(|(path, mut file)| {
            if let Err(error) = file.write_all(body) {
                drop(file);
                let _ = std::fs::remove_file(&path);
                return Err(error);
            }
            Ok(path)
        });
        Arc::new(match result {
            Ok(path) => Self { path: Some(path), failure: None, _reservation: Some(reservation) },
            Err(error) => Self { path: None, failure: Some(error.to_string()), _reservation: None },
        })
    }
}
impl Drop for CapturedDownload {
    fn drop(&mut self) {
        if let Some(path) = &self.path { let _ = std::fs::remove_file(path); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn download_staging_shares_session_quota_and_releases_last_file_owner() {
        let mut budgets = DownloadBudgets::default();
        let first_context = budgets.for_session("session-a", 10);
        let second_context = budgets.for_session("session-a", 10);
        let other_session = budgets.for_session("session-b", 10);
        assert!(Arc::ptr_eq(&first_context, &second_context), "session allowance was duplicated by context");
        let root = std::env::temp_dir().join(format!("greppy-staging-contract-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        struct OwnedDirectory(PathBuf);
        impl Drop for OwnedDirectory {
            fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
        }
        let _owned_directory = OwnedDirectory(root.clone());
        let allocated = AtomicUsize::new(0);
        let allocate = || {
            let index = allocated.fetch_add(1, Ordering::SeqCst);
            let path = root.join(index.to_string());
            let file = File::options().create_new(true).write(true).open(&path)?;
            Ok((path, file))
        };
        let first = CapturedDownload::stage(b"12345678", Some(first_context), &allocate);
        let first_path = first.path.clone().expect("first file must fit");
        assert_eq!(std::fs::read(&first_path).unwrap(), b"12345678");
        let refused = CapturedDownload::stage(b"abcd", Some(Arc::clone(&second_context)), &allocate);
        assert!(refused.path.is_none(), "second context bypassed the shared session budget");
        assert!(refused.failure.as_deref().unwrap().contains("resource_limit"));
        assert_eq!(allocated.load(Ordering::SeqCst), 1, "file allocator ran before quota admission");
        let unauthorized = CapturedDownload::stage(b"x", None, &allocate);
        assert!(unauthorized.path.is_none());
        assert_eq!(allocated.load(Ordering::SeqCst), 1, "missing authority created a file");
        let other = CapturedDownload::stage(b"abcd", Some(other_session), &allocate);
        assert!(other.path.is_some(), "foreign session consumed this session's quota");
        let held = Arc::clone(&first);
        drop(first);
        assert!(first_path.is_file(), "file removed before its last owner");
        drop(held);
        assert!(!first_path.exists(), "last file owner did not remove its file");
        let restored = CapturedDownload::stage(b"abcdefghij", Some(Arc::clone(&second_context)), &allocate);
        assert!(restored.path.is_some(), "last owner did not return the session allowance");
        drop(restored);
        let failed = CapturedDownload::stage(b"abcdefghij", Some(Arc::clone(&second_context)),
            || Err(io::Error::new(io::ErrorKind::PermissionDenied, "allocator denied")));
        assert!(failed.path.is_none());
        let after_error = CapturedDownload::stage(b"abcdefghij", Some(Arc::clone(&second_context)), &allocate);
        assert!(after_error.path.is_some(), "allocator failure leaked the byte reservation");
        drop(after_error);
        let readonly = root.join("readonly");
        let write_failure = CapturedDownload::stage(b"abcdefghij", Some(Arc::clone(&budgets.for_session("session-a", 10))), || {
            File::create(&readonly)?;
            Ok((readonly.clone(), File::open(&readonly)?))
        });
        assert!(write_failure.path.is_none());
        assert!(!readonly.exists(), "failed write retained its partially created file");
        let after_write = CapturedDownload::stage(b"abcdefghij", Some(budgets.for_session("session-a", 10)), &allocate);
        assert!(after_write.path.is_some(), "write failure leaked the byte reservation");
        drop(after_write);
        drop(other);
        std::fs::remove_dir(&root).unwrap();
    }
}
