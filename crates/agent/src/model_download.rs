//! Explicit HTTPS provisioning for an optional local agent release artifact.
//! Enabled only by local-model-download. No global client, release URL or weights.
//! The caller authenticates the release catalog before supplying its URL and hash.

use std::cell::RefCell;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Duration;

use crate::local_model::{ArtifactCache, ArtifactSpec, ProvisionCancel, ProvisionEvent};

/// Bounded retries restart a failed transfer from byte zero.
/// Run on a provisioning worker, never on the UI thread.
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub attempts: u32,
    pub retry_delay: Duration,
    pub connect_timeout: Duration,
    pub read_timeout: Duration,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            attempts: 3,
            retry_delay: Duration::from_millis(500),
            connect_timeout: Duration::from_secs(10),
            read_timeout: Duration::from_secs(2),
        }
    }
}

/// Dedicated TLS client; never reuses gateway credentials or its HTTP policy.
#[derive(Debug)]
pub struct ModelDownloader {
    agent: ureq::Agent,
    options: DownloadOptions,
}

impl ModelDownloader {
    /// Constructs configuration only; performs no network or filesystem I/O.
    pub fn new(options: DownloadOptions) -> io::Result<Self> {
        if options.attempts == 0
            || options.attempts > 5
            || options.connect_timeout.is_zero()
            || options.read_timeout.is_zero()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "model download requires 1..=5 attempts and nonzero I/O timeouts",
            ));
        }
        let agent = Self::agent_builder(&options).build();
        Ok(Self { agent, options })
    }

    fn agent_builder(options: &DownloadOptions) -> ureq::AgentBuilder {
        ureq::AgentBuilder::new()
            .https_only(true)
            .redirects(5)
            .timeout_connect(options.connect_timeout)
            .timeout_read(options.read_timeout)
            .timeout_write(options.read_timeout)
    }

    /// Fetch only on explicit first use. Hash/length must be authenticated by
    /// the release catalog; successful download does not admit the engine.
    ///
    /// Cancellation is checked between chunks and in cache/retry waits.
    /// A blocked network operation returns at its configured timeout; OS DNS
    /// resolution can exceed that timeout. No range/resume is attempted.
    pub fn ensure(
        &self,
        cache: &ArtifactCache,
        spec: &ArtifactSpec,
        url: &str,
        cancel: &ProvisionCancel,
        mut progress: impl FnMut(ProvisionEvent),
    ) -> io::Result<PathBuf> {
        cancel.check()?;
        // ureq enforces HTTPS again on every redirect. Do not put a signed URL
        // or its credentials in diagnostics.
        if !url.starts_with("https://") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "model artifacts require an HTTPS URL",
            ));
        }
        for attempt in 1..=self.options.attempts {
            cancel.check()?;
            let retryable = RefCell::new(false);
            let result = cache.ensure_controlled(
                spec,
                cancel,
                |out| {
                    let response = match self
                        .agent
                        .get(url)
                        .set("Accept-Encoding", "identity")
                        .call()
                    {
                        Ok(response) => response,
                        Err(ureq::Error::Status(status, _)) => {
                            *retryable.borrow_mut() =
                                status == 408 || status == 429 || status >= 500;
                            return Err(io::Error::other(format!(
                                "model download HTTP status {status}"
                            )));
                        }
                        Err(ureq::Error::Transport(_)) => {
                            *retryable.borrow_mut() = true;
                            return Err(io::Error::other("model download transport failed"));
                        }
                    };
                    cancel.check()?;
                    if response.status() != 200 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "model download requires HTTP 200, received {}",
                                response.status()
                            ),
                        ));
                    }
                    if response
                        .header("Content-Encoding")
                        .is_some_and(|v| !v.eq_ignore_ascii_case("identity"))
                    {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "encoded model response refused",
                        ));
                    }
                    if let Some(length) = response.header("Content-Length") {
                        if length.parse::<u64>().ok() != Some(spec.size_bytes) {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "model download Content-Length mismatch",
                            ));
                        }
                    }
                    let mut reader = response.into_reader();
                    let mut buffer = [0_u8; 64 * 1024];
                    let mut received = 0_u64;
                    loop {
                        cancel.check()?;
                        let count = match reader.read(&mut buffer) {
                            Ok(count) => count,
                            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                            Err(_) => {
                                *retryable.borrow_mut() = true;
                                return Err(io::Error::other("model download body read failed"));
                            }
                        };
                        cancel.check()?;
                        if count == 0 {
                            if received < spec.size_bytes {
                                *retryable.borrow_mut() = true;
                                return Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    "model download ended early",
                                ));
                            }
                            return Ok(());
                        }
                        // Writer errors (disk full, cancellation, oversized data)
                        // are never classified as retryable network failures.
                        out.write_all(&buffer[..count])?;
                        received += count as u64;
                    }
                },
                &mut progress,
            );
            cancel.check()?;
            match result {
                Ok(path) => return Ok(path),
                Err(error) if !*retryable.borrow() || attempt == self.options.attempts => {
                    return Err(error)
                }
                Err(_) => {
                    progress(ProvisionEvent::Retrying {
                        next_attempt: attempt + 1,
                    });
                    cancel.wait(self.options.retry_delay)?;
                }
            }
        }
        unreachable!("nonempty bounded attempt loop")
    }
}

#[cfg(test)]
#[path = "model_download_tests.rs"]
mod tests;
