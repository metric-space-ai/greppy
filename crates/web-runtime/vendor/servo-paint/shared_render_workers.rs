// SPDX-License-Identifier: MPL-2.0
//! Share render capacity without sharing renderer, document or network state.
//!
//! The bootstrap Painter owns the process resources until ContentEngine exits.
//! Context Painters borrow the same pool. The cache owns only a Weak plus one
//! generation's join handles, never a strong reference to the pool. A changed
//! CPU/thread preference takes effect after the last live pool owner releases.
//! Rayon requests termination on ThreadPool::drop; it does not join its workers.
//! Rebuild therefore joins the retired generation before creating any new one.
use std::cell::Cell;
use std::io;
use std::sync::{Arc, Mutex, Weak};
use std::thread::{self, JoinHandle};

thread_local! {
    static IN_RENDER_WORKER: Cell<bool> = const { Cell::new(false) };
}

static WORKERS: SharedRenderWorkers = SharedRenderWorkers::new();

pub(crate) fn shared_render_workers(count: usize) -> io::Result<Arc<rayon::ThreadPool>> {
    WORKERS.acquire(count)
}

struct Generation {
    pool: Weak<rayon::ThreadPool>,
    workers: Vec<JoinHandle<()>>,
}

struct SharedRenderWorkers {
    current: Mutex<Option<Generation>>,
    #[cfg(test)]
    events: Option<Arc<Events>>,
}

impl SharedRenderWorkers {
    const fn new() -> Self {
        Self {
            current: Mutex::new(None),
            #[cfg(test)]
            events: None,
        }
    }

    fn acquire(&self, count: usize) -> io::Result<Arc<rayon::ThreadPool>> {
        // Before the lock: the embedder may hold it while joining this worker.
        // Painter construction belongs to the embedder, never its Rayon jobs.
        if IN_RENDER_WORKER.with(Cell::get) {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "a render worker cannot acquire its renderer pool",
            ));
        }
        let mut current = self
            .current
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(pool) = current.as_ref().and_then(|entry| entry.pool.upgrade()) {
            return Ok(pool);
        }
        if let Some(previous) = current.take() {
            join_workers(previous.workers)?;
        }

        let mut workers = Vec::new();
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(count.max(1))
            .thread_name(|index| format!("WRWorker#{index}"))
            .spawn_handler(|worker| {
                let mut builder = thread::Builder::new();
                if let Some(name) = worker.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = worker.stack_size() {
                    builder = builder.stack_size(size);
                }
                #[cfg(test)]
                let events = self.events.clone();
                workers.push(builder.spawn(move || {
                    IN_RENDER_WORKER.with(|state| state.set(true));
                    #[cfg(test)]
                    if let Some(events) = events.as_ref() {
                        events.started();
                    }
                    worker.run();
                    #[cfg(test)]
                    if let Some(events) = events.as_ref() {
                        events.exited();
                    }
                    IN_RENDER_WORKER.with(|state| state.set(false));
                })?);
                Ok(())
            })
            .build();
        match built {
            Ok(pool) => {
                let pool = Arc::new(pool);
                *current = Some(Generation {
                    pool: Arc::downgrade(&pool),
                    workers,
                });
                Ok(pool)
            }
            Err(error) => {
                // Rayon terminates any partially built registry on error.
                // Keep a failed attempt out of the cache and drain its threads.
                join_workers(workers)?;
                Err(io::Error::other(error.to_string()))
            }
        }
    }
}

fn join_workers(workers: Vec<JoinHandle<()>>) -> io::Result<()> {
    let mut failed = false;
    for worker in workers {
        failed |= worker.join().is_err();
    }
    if failed {
        return Err(io::Error::other("a retired render worker panicked"));
    }
    Ok(())
}

impl Drop for SharedRenderWorkers {
    fn drop(&mut self) {
        let Some(generation) = self
            .current
            .get_mut()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        else {
            return;
        };
        // A local registry can be dropped before an externally owned pool.
        // Do not wait for that live pool or attempt to join from its worker.
        // Dropping the join handles detaches; Rayon's last owner still governs
        // termination. The production registry is process-static.
        if generation.pool.strong_count() == 0 && !IN_RENDER_WORKER.with(Cell::get) {
            let _ = join_workers(generation.workers);
        }
    }
}

#[cfg(test)]
#[derive(Default)]
struct Events {
    starts: std::sync::atomic::AtomicUsize,
    exits: std::sync::atomic::AtomicUsize,
    live: std::sync::atomic::AtomicUsize,
    peak: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl Events {
    fn started(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        self.starts.fetch_add(1, SeqCst);
        let live = self.live.fetch_add(1, SeqCst) + 1;
        self.peak.fetch_max(live, SeqCst);
    }

    fn exited(&self) {
        use std::sync::atomic::Ordering::SeqCst;
        self.exits.fetch_add(1, SeqCst);
        self.live.fetch_sub(1, SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::sync::{Barrier, mpsc};
    use std::time::{Duration, Instant};

    fn fixture() -> (Arc<SharedRenderWorkers>, Arc<Events>) {
        let events = Arc::new(Events::default());
        let registry = Arc::new(SharedRenderWorkers {
            current: Mutex::new(None),
            events: Some(events.clone()),
        });
        (registry, events)
    }

    #[test]
    fn overlapping_renderers_share_real_workers_and_release_the_last_owner() {
        let (registry, events) = fixture();
        let gate = Arc::new(Barrier::new(5));
        let clients: Vec<_> = (0..4)
            .map(|owner| {
                let registry = registry.clone();
                let gate = gate.clone();
                thread::spawn(move || {
                    gate.wait();
                    let pool = registry.acquire(3).unwrap();
                    let result = pool.install(|| {
                        (0..4096_u64)
                            .into_par_iter()
                            .map(|value| value + owner * 4096)
                            .sum::<u64>()
                    });
                    let threads = pool.broadcast(|_| thread::current().id());
                    (pool, result, threads)
                })
            })
            .collect();
        gate.wait();
        let owners: Vec<_> = clients
            .into_iter()
            .map(|client| client.join().unwrap())
            .collect();
        for (owner, (_, actual, _)) in owners.iter().enumerate() {
            let expected: u64 = (0..4096_u64).map(|value| value + owner as u64 * 4096).sum();
            assert_eq!(*actual, expected, "shared render pool lost an owner's work");
        }
        let ids: HashSet<_> = owners
            .iter()
            .flat_map(|(_, _, ids)| ids.iter().copied())
            .collect();
        assert_eq!(
            ids.len(),
            3,
            "overlapping renderers started duplicate worker threads"
        );
        assert!(
            owners
                .iter()
                .all(|(pool, _, _)| Arc::ptr_eq(pool, &owners[0].0))
        );
        assert_eq!(events.starts.load(SeqCst), 3);
        assert_eq!(events.peak.load(SeqCst), 3);
        let weak = Arc::downgrade(&owners[0].0);
        drop(owners);
        assert!(
            weak.upgrade().is_none(),
            "renderer pool survived its last owner"
        );

        let resized = registry.acquire(1).unwrap();
        assert_eq!(resized.current_num_threads(), 1);
        assert_eq!(
            events.exits.load(SeqCst),
            3,
            "retired workers were not joined before restart"
        );
        drop(resized);
        drop(registry);
        assert_eq!(events.starts.load(SeqCst), 4);
        assert_eq!(events.exits.load(SeqCst), 4);
        assert_eq!(events.live.load(SeqCst), 0);
        println!(
            "render pool: four overlapping owners, degree3, unique workers3; full work preserved; all workers exited; restart degree1"
        );
    }

    #[test]
    fn restart_drains_running_render_work_before_starting_new_workers() {
        let (registry, events) = fixture();
        let pool = registry.acquire(2).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let completed = Arc::new(AtomicUsize::new(0));
        let completed_job = completed.clone();
        pool.spawn(move || {
            started_tx.send(()).unwrap();
            finish_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            completed_job.store(4096, SeqCst);
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(pool);
        let next_registry = registry.clone();
        let (enter_tx, enter_rx) = mpsc::channel();
        let (result_tx, result_rx) = mpsc::channel();
        let restart = thread::spawn(move || {
            enter_tx.send(()).unwrap();
            result_tx.send(next_registry.acquire(2).unwrap()).unwrap();
        });
        enter_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let early = result_rx.recv_timeout(Duration::from_millis(250)).ok();
        finish_tx.send(()).unwrap();
        let next = early.unwrap_or_else(|| result_rx.recv_timeout(Duration::from_secs(5)).unwrap());
        restart.join().unwrap();
        assert_eq!(
            completed.load(SeqCst),
            4096,
            "restart discarded pending renderer work"
        );
        assert_eq!(
            events.peak.load(SeqCst),
            2,
            "new render workers overlapped retiring generation"
        );
        assert_eq!(events.exits.load(SeqCst), 2);
        drop(next);
        drop(registry);
        assert_eq!(events.starts.load(SeqCst), events.exits.load(SeqCst));
        assert_eq!(events.live.load(SeqCst), 0);
    }

    #[test]
    fn retiring_worker_refuses_registry_reentry_before_the_embedder_lock() {
        let (registry, events) = fixture();
        let pool = registry.acquire(2).unwrap();
        let worker_registry = registry.clone();
        let (started_tx, started_rx) = mpsc::channel();
        let (allow_tx, allow_rx) = mpsc::channel();
        let (refused_tx, refused_rx) = mpsc::channel();
        pool.spawn(move || {
            started_tx.send(()).unwrap();
            allow_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            let refused = matches!(worker_registry.acquire(1), Err(error) if error.kind() == io::ErrorKind::WouldBlock);
            refused_tx.send(refused).unwrap();
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(pool);
        let embedder_registry = registry.clone();
        let restart = thread::spawn(move || embedder_registry.acquire(1).unwrap());
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if matches!(
                registry.current.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "embedder did not begin retiring worker join"
            );
            thread::yield_now();
        }
        allow_tx.send(()).unwrap();
        assert_eq!(
            refused_rx.recv_timeout(Duration::from_secs(2)),
            Ok(true),
            "renderer worker blocked on the retiring pool registry"
        );
        let next = restart.join().unwrap();
        assert_eq!(events.exits.load(SeqCst), 2);
        drop(next);
        drop(registry);
        assert_eq!(events.live.load(SeqCst), 0);
    }

    #[test]
    fn registry_drop_does_not_join_an_externally_owned_pool() {
        let (registry, events) = fixture();
        let pool = registry.acquire(2).unwrap();
        drop(registry);
        assert_eq!(pool.install(|| 7 * 9), 63);
        let (tx, rx) = mpsc::channel();
        pool.spawn(move || {
            tx.send(()).unwrap();
        });
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        drop(pool);
        let deadline = Instant::now() + Duration::from_secs(5);
        while events.live.load(SeqCst) != 0 {
            assert!(
                Instant::now() < deadline,
                "externally owned render workers did not exit"
            );
            thread::yield_now();
        }
    }
}
