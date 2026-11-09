// SPDX-License-Identifier: MPL-2.0
//! Isolated storage actors share execution capacity, not databases or state.
//! Start one worker per service class. Grow only after sustained ready work on
//! several ports, moving the whole Send actor between workers at a message
//! boundary. IndexedDB and ClientStorage stay on different service pools.
use std::cell::RefCell;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Select, SelectedOperation, Sender};
use serde::{Deserialize, Serialize};
use servo_base::generic_channel::{GenericReceiver, RoutedReceiver};

trait Port: Send {
    fn register<'a>(&'a self, select: &mut Select<'a>);
    fn dispatch(&self, operation: SelectedOperation<'_>) -> bool;
    fn pending(&self) -> bool;
}

struct StoragePort<T, F> {
    receiver: RoutedReceiver<T>,
    handler: RefCell<F>,
}

impl<T, F> Port for StoragePort<T, F>
where
    T: Serialize + for<'de> Deserialize<'de> + Send + 'static,
    F: FnMut(T) -> bool + Send + 'static,
{
    fn register<'a>(&'a self, select: &mut Select<'a>) { select.recv(&self.receiver); }
    fn pending(&self) -> bool { !self.receiver.is_empty() }
    fn dispatch(&self, operation: SelectedOperation<'_>) -> bool {
        match operation.recv(&self.receiver) {
            Ok(Ok(message)) => (self.handler.borrow_mut())(message),
            Ok(Err(_)) | Err(_) => false,
        }
    }
}

struct Registration {
    port: Box<dyn Port>,
    owner: Arc<AtomicUsize>,
}

impl Registration {
    fn reassign(&mut self, owner: Arc<AtomicUsize>) {
        self.owner.fetch_sub(1, Ordering::Relaxed);
        owner.fetch_add(1, Ordering::Relaxed);
        self.owner = owner;
    }
}

impl Drop for Registration {
    fn drop(&mut self) { self.owner.fetch_sub(1, Ordering::Relaxed); }
}

struct Worker {
    registrations: Sender<Registration>,
    active_ports: Arc<AtomicUsize>,
}

struct Pool {
    name: &'static str,
    capacity: AtomicUsize,
    workers: Mutex<Vec<Worker>>,
}

impl Pool {
    fn new(name: &'static str, capacity: usize) -> Self {
        Self { name, capacity: AtomicUsize::new(capacity.max(1)), workers: Mutex::new(Vec::new()) }
    }

    fn start(self: &Arc<Self>, workers: &mut Vec<Worker>) -> io::Result<usize> {
        let index = workers.len();
        let (sender, receiver) = crossbeam_channel::unbounded();
        let pool = Arc::downgrade(self);
        thread::Builder::new().name(format!("{}#{index}", self.name))
            .spawn(move || run(receiver, pool, index))?;
        workers.push(Worker {
            registrations: sender,
            active_ports: Arc::new(AtomicUsize::new(0)),
        });
        Ok(index)
    }

    fn register(self: &Arc<Self>, port: Box<dyn Port>) -> io::Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(|error| error.into_inner());
        if workers.is_empty() { self.start(&mut workers)?; }
        let worker = workers.iter().min_by_key(|w| w.active_ports.load(Ordering::Relaxed)).unwrap();
        worker.active_ports.fetch_add(1, Ordering::Relaxed);
        let job = Registration { port, owner: Arc::clone(&worker.active_ports) };
        worker.registrations.send(job).map_err(|_| io::Error::other("storage worker exited"))
    }

    fn redistribute(self: &Arc<Self>, mut job: Registration, source: usize)
        -> Result<(), Registration>
    {
        let mut workers = self.workers.lock().unwrap_or_else(|error| error.into_inner());
        let source_load = workers[source].active_ports.load(Ordering::Relaxed);
        let existing = workers.iter().enumerate()
            .filter(|(id, worker)| *id != source
                && worker.active_ports.load(Ordering::Relaxed).saturating_add(1) < source_load)
            .min_by_key(|(_, worker)| worker.active_ports.load(Ordering::Relaxed))
            .map(|(id, _)| id);
        let destination = match existing {
            Some(index) => index,
            None if workers.len() < self.capacity.load(Ordering::Relaxed) => {
                match self.start(&mut workers) {
                    Ok(index) => index,
                    // Another lane is optional. Keep using the existing actor
                    // worker if the kernel cannot allocate an extra thread.
                    Err(_) => return Err(job),
                }
            }
            None => return Err(job),
        };
        let previous = Arc::clone(&job.owner);
        job.reassign(Arc::clone(&workers[destination].active_ports));
        match workers[destination].registrations.send(job) {
            Ok(()) => Ok(()),
            Err(error) => {
                let mut job = error.0;
                job.reassign(previous);
                Err(job)
            }
        }
    }
}

pub(crate) struct StorageWorker {
    name: &'static str,
    pool: OnceLock<Arc<Pool>>,
}

impl StorageWorker {
    pub(crate) const fn new(name: &'static str) -> Self { Self { name, pool: OnceLock::new() } }

    pub(crate) fn register<T, F>(&self, receiver: GenericReceiver<T>, handler: F) -> io::Result<()>
    where
        T: Serialize + for<'de> Deserialize<'de> + Send + 'static,
        F: FnMut(T) -> bool + Send + 'static,
    {
        let capacity = thread::available_parallelism().map_or(1, |n| n.get());
        let pool = self.pool.get_or_init(|| Arc::new(Pool::new(self.name, capacity)));
        pool.capacity.store(capacity, Ordering::Relaxed);
        pool.register(Box::new(StoragePort {
            receiver: receiver.route_preserving_errors(), handler: RefCell::new(handler),
        }))
    }
}

fn run(registrations: Receiver<Registration>, pool: Weak<Pool>, id: usize) {
    let mut ports: Vec<Registration> = Vec::new();
    let mut busy_since = None;
    loop {
        while let Ok(port) = registrations.try_recv() { ports.push(port); }
        if !ports.iter().any(|p| p.port.pending()) { busy_since = None; }
        enum Event { Register(Option<Registration>), Retire(usize), Keep }
        let event = {
            let mut select = Select::new();
            select.recv(&registrations);
            for job in &ports { job.port.register(&mut select); }
            let operation = select.select();
            let index = operation.index();
            if index == 0 {
                Event::Register(operation.recv(&registrations).ok())
            } else {
                busy_since.get_or_insert_with(Instant::now);
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(||
                    ports[index - 1].port.dispatch(operation))).unwrap_or(false)
                { Event::Keep } else { Event::Retire(index - 1) }
            }
        };
        match event {
            Event::Register(Some(port)) => ports.push(port),
            Event::Register(None) => return,
            Event::Retire(index) => { ports.swap_remove(index); }
            Event::Keep => {}
        }
        let backlogged = ports.iter().filter(|p| p.port.pending()).take(2).count() == 2;
        if backlogged && busy_since.is_some_and(|start| start.elapsed() >= Duration::from_millis(2)) {
            if let Some(pool) = pool.upgrade() {
                let index = ports.iter().position(|p| p.port.pending()).unwrap();
                let job = ports.swap_remove(index);
                if let Err(job) = pool.redistribute(job, id) { ports.push(job); }
            }
            busy_since = None;
        }
    }
}

pub(crate) static CLIENT_STORAGE: StorageWorker = StorageWorker::new("ClientStorage");
pub(crate) static INDEXED_DB: StorageWorker = StorageWorker::new("IndexedDB");
pub(crate) static WEB_STORAGE: StorageWorker = StorageWorker::new("WebStorage");
pub(crate) static CACHE_STORAGE: StorageWorker = StorageWorker::new("CacheStorage");

#[cfg(test)]
mod tests {
    use super::*;
    use servo_base::generic_channel;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn backlog_uses_available_capacity_without_reordering_or_sharing_actor_state() {
        servo_config::opts::initialize_options(Default::default());
        let pool = Arc::new(Pool::new("StorageCapacityContract", 2));
        let ready = Arc::new(AtomicBool::new(false));
        let (reply, replies) = std::sync::mpsc::channel();
        let mut senders = Vec::new();
        for actor in 0..2 {
            let (sender, receiver) = generic_channel::channel::<u64>().unwrap();
            let reply = reply.clone(); let ready = Arc::clone(&ready);
            let mut sum = 0_u64;
            pool.register(Box::new(StoragePort {
                receiver: receiver.route_preserving_errors(),
                handler: RefCell::new(move |message| {
                    if message == u64::MAX { return false; }
                    while !ready.load(Ordering::Acquire) { thread::yield_now(); }
                    sum += message;
                    let mut work = 0_u64;
                    for n in 0..2_000_000_u64 { work = std::hint::black_box(work.wrapping_add(n)); }
                    reply.send((actor, message, sum, work, thread::current().id())).unwrap();
                    true
                }),
            })).unwrap();
            senders.push(sender);
        }
        for message in 0..12 { for sender in &senders { sender.send(message).unwrap(); } }
        ready.store(true, Ordering::Release);
        let mut next = [0_u64; 2]; let mut threads = HashSet::new();
        for _ in 0..24 {
            let (actor, message, sum, work, thread) = replies.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(message, next[actor], "migration reordered one actor");
            assert_eq!(sum, message * (message + 1) / 2, "actor state crossed context boundaries");
            assert_eq!(work, 2_000_000_u64 * 1_999_999 / 2, "work was discarded");
            next[actor] += 1; threads.insert(thread);
        }
        assert_eq!(threads.len(), 2, "backlogged actors left declared execution capacity unused");
        for sender in senders { sender.send(u64::MAX).unwrap(); }
        drop(pool);
    }
}
