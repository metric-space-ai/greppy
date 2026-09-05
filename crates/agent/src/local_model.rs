//! Opt-in preparation primitives for a native agent backend.
//!
//! Nothing here downloads or loads a model implicitly. Constructing a cache,
//! checking it, and constructing an on-demand stream have no network effects.
//! The caller supplies a persistent cache root and, only on an explicit local
//! agent request, a fetcher and engine factory. No model assets are bundled.
//!
//! Artifact hashes must come from an authenticated release manifest. This
//! cache checks byte integrity, not publisher identity or engine admission.
//! The engine must still verify the complete signed release on load.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::{ClientError, ModelRequest, ModelStream, StreamEvent, TurnResult};

/// Cooperative provisioning cancellation; independent of an active socket read.
#[derive(Debug, Clone, Default)]
pub struct ProvisionCancel(Arc<AtomicBool>);

impl ProvisionCancel {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    pub(crate) fn check(&self) -> io::Result<()> {
        if self.is_cancelled() {
            // Interrupted would be retried indefinitely by Write::write_all.
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "model provisioning cancelled",
            ))
        } else {
            Ok(())
        }
    }

    pub(crate) fn wait(&self, delay: Duration) -> io::Result<()> {
        let started = Instant::now();
        loop {
            self.check()?;
            let remaining = delay.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Ok(());
            }
            std::thread::sleep(remaining.min(Duration::from_millis(25)));
        }
    }
}

/// Artifact progress, separate from signed release admission and engine readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvisionEvent {
    WaitingForCache,
    CheckingCache,
    Downloading { received: u64, total: u64 },
    Verifying,
    ArtifactReady { cached: bool },
    Retrying { next_attempt: u32 },
}

/// Expected bytes of one artifact from an authenticated release.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactSpec {
    pub sha256: [u8; 32],
    pub size_bytes: u64,
}

impl ArtifactSpec {
    fn key(&self) -> String {
        self.sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// Content-addressed storage in a caller-selected, durable directory.
///
/// No default temp directory is used. Objects remain cached when the model
/// is unloaded. This is a blob cache, not a complete signed release installer.
#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
}

impl ArtifactCache {
    /// Record the cache location without creating directories or doing I/O.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        if !root.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "local model cache requires an absolute persistent path",
            ));
        }
        Ok(Self { root })
    }

    fn object_path(&self, spec: &ArtifactSpec) -> PathBuf {
        self.root.join("objects").join(spec.key())
    }

    /// Read-only, fully verified lookup. Missing or corrupt bytes are a miss.
    ///
    /// A partial download is never returned. The check streams the whole
    /// artifact with bounded memory; it must not be used as a cheap UI poll.
    pub fn lookup_verified(&self, spec: &ArtifactSpec) -> io::Result<Option<PathBuf>> {
        self.lookup_controlled(spec, &ProvisionCancel::default())
    }

    fn lookup_controlled(
        &self,
        spec: &ArtifactSpec,
        cancel: &ProvisionCancel,
    ) -> io::Result<Option<PathBuf>> {
        cancel.check()?;
        let path = self.object_path(spec);
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        if !file.metadata()?.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "local model cache object is not a regular file",
            ));
        }
        if file.metadata()?.len() != spec.size_bytes {
            return Ok(None);
        }
        let mut hash = Sha256::new();
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            cancel.check()?;
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hash.update(&buffer[..read]);
        }
        cancel.check()?;
        let digest: [u8; 32] = hash.finalize().into();
        Ok((digest == spec.sha256).then_some(path))
    }

    /// Explicitly fetch a missing object, then atomically publish verified bytes.
    ///
    /// Only the local agent's first-use path may call this. The fetcher supplies
    /// bytes through a bounded writer; it owns HTTP, timeouts and cancellation.
    /// Progress reports bytes written, not model readiness. The return value
    /// means byte verification succeeded; signed engine admission is separate.
    ///
    /// An OS file lock serializes downloads of the same digest across processes.
    /// An interrupted or invalid transfer leaves no published object. Staging
    /// files live on the same filesystem and are removed on ordinary failure;
    /// crash leftovers are ignored. Cache hits never invoke the fetcher.
    pub fn ensure_with(
        &self,
        spec: &ArtifactSpec,
        fetch: impl FnOnce(&mut dyn Write) -> io::Result<()>,
        mut progress: impl FnMut(u64, u64),
    ) -> io::Result<PathBuf> {
        self.ensure_controlled(spec, &ProvisionCancel::default(), fetch, |event| {
            if let ProvisionEvent::Downloading { received, total } = event {
                progress(received, total);
            }
        })
    }

    /// Explicit first-use provisioning with cancellation during lock waits,
    /// hashing and writes. The fetcher must bound its own blocking I/O.
    /// Cancellation uses ConnectionAborted so write_all never retries it.
    pub fn ensure_controlled(
        &self,
        spec: &ArtifactSpec,
        cancel: &ProvisionCancel,
        fetch: impl FnOnce(&mut dyn Write) -> io::Result<()>,
        mut progress: impl FnMut(ProvisionEvent),
    ) -> io::Result<PathBuf> {
        cancel.check()?;
        let objects = self.root.join("objects");
        let locks = self.root.join("locks");
        fs::create_dir_all(&objects)?;
        fs::create_dir_all(&locks)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(locks.join(spec.key()))?;
        let mut waiting_reported = false;
        loop {
            cancel.check()?;
            match lock.try_lock() {
                Ok(()) => break,
                Err(fs::TryLockError::WouldBlock) => {
                    if !waiting_reported {
                        progress(ProvisionEvent::WaitingForCache);
                        waiting_reported = true;
                    }
                    cancel.wait(Duration::from_millis(25))?;
                }
                Err(fs::TryLockError::Error(error)) => return Err(error),
            }
        }
        // Always recheck under the lock: another caller may have installed it.
        progress(ProvisionEvent::CheckingCache);
        if let Some(path) = self.lookup_controlled(spec, cancel)? {
            cancel.check()?;
            progress(ProvisionEvent::ArtifactReady { cached: true });
            return Ok(path);
        }
        cancel.check()?;

        let mut staging = tempfile::Builder::new()
            .prefix(".download-")
            .tempfile_in(&objects)?;
        progress(ProvisionEvent::Downloading {
            received: 0,
            total: spec.size_bytes,
        });
        let mut writer = VerifiedWriter {
            output: staging.as_file_mut(),
            expected: spec.size_bytes,
            written: 0,
            hash: Sha256::new(),
            progress: &mut progress,
            cancel,
        };
        cancel.check()?;
        fetch(&mut writer)?;
        cancel.check()?;
        writer.flush()?;
        if writer.written != spec.size_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "local model artifact size mismatch",
            ));
        }
        let digest: [u8; 32] = writer.hash.finalize().into();
        progress(ProvisionEvent::Verifying);
        cancel.check()?;
        if digest != spec.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "local model artifact SHA-256 mismatch",
            ));
        }
        staging.as_file().sync_all()?;
        cancel.check()?;
        let path = self.object_path(spec);
        staging.persist(&path).map_err(|error| error.error)?;
        progress(ProvisionEvent::ArtifactReady { cached: false });
        Ok(path)
    }
}

struct VerifiedWriter<'a> {
    output: &'a mut File,
    expected: u64,
    written: u64,
    hash: Sha256,
    progress: &'a mut dyn FnMut(ProvisionEvent),
    cancel: &'a ProvisionCancel,
}

impl Write for VerifiedWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.cancel.check()?;
        if bytes.len() as u64 > self.expected.saturating_sub(self.written) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "local model artifact exceeds declared size",
            ));
        }
        let written = self.output.write(bytes)?;
        self.hash.update(&bytes[..written]);
        self.written += written as u64;
        (self.progress)(ProvisionEvent::Downloading {
            received: self.written,
            total: self.expected,
        });
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.cancel.check()?;
        self.output.flush()
    }
}

/// Initialize a local model adapter on its first actual agent turn.
///
/// The factory owns provisioning and signed engine admission. It must return
/// a ready ModelStream or an error, never silently fall back to the gateway.
/// Existing agent history, tool execution and event handling stay unchanged.
pub struct OnDemandModel<M, F> {
    model: Option<M>,
    factory: F,
}

impl<M, F> std::fmt::Debug for OnDemandModel<M, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnDemandModel")
            .field("initialized", &self.model.is_some())
            .finish_non_exhaustive()
    }
}

impl<M, F> OnDemandModel<M, F> {
    /// No factory call, disk access, download, daemon spawn or prewarm.
    pub fn new(factory: F) -> Self {
        Self {
            model: None,
            factory,
        }
    }

    /// Inspect adapter state without initializing the model.
    pub fn is_initialized(&self) -> bool {
        self.model.is_some()
    }
}

impl<M, F> ModelStream for OnDemandModel<M, F>
where
    M: ModelStream,
    F: FnMut() -> Result<M, ClientError>,
{
    fn stream_turn(
        &mut self,
        req: &ModelRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<TurnResult, ClientError> {
        if self.model.is_none() {
            self.model = Some((self.factory)()?);
        }
        let result = self
            .model
            .as_mut()
            .expect("initialized model")
            .stream_turn(req, on_event);
        if result.is_err() {
            // Never reuse a potentially partially advanced native session.
            // Its adapter must cancel/reset and release ownership on drop.
            self.model = None;
        }
        result
    }
}
