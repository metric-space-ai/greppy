// SPDX-License-Identifier: MPL-2.0
//! Share service capacity, never storage state. Each registered port owns its
//! own manager, database connection, origin maps and profiler registration.
//! Different service classes have different workers: IndexedDB may synchronously
//! call ClientStorage, so those services must not occupy the same worker.
//! Workers are process scoped; Exit removes and drops exactly its port's state.
use std::cell::RefCell;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crossbeam_channel::{Receiver, Select, SelectedOperation, Sender};
use serde::{Deserialize, Serialize};
use servo_base::generic_channel::{GenericReceiver, RoutedReceiver};

trait Port: Send {
    fn register<'a>(&'a self, select: &mut Select<'a>);
    fn dispatch(&self, operation: SelectedOperation<'_>) -> bool;
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
    fn register<'a>(&'a self, select: &mut Select<'a>) {
        select.recv(&self.receiver);
    }

    fn dispatch(&self, operation: SelectedOperation<'_>) -> bool {
        match operation.recv(&self.receiver) {
            Ok(Ok(message)) => (self.handler.borrow_mut())(message),
            // Closing a port releases only this manager. Do not retain a closed
            // receiver in Select (it would otherwise spin forever).
            Ok(Err(_)) | Err(_) => false,
        }
    }
}

type Registration = Box<dyn Port>;

struct Worker {
    registrations: Sender<Registration>,
    active_ports: Arc<AtomicUsize>,
}

struct CountedPort {
    inner: Registration,
    active_ports: Arc<AtomicUsize>,
}

impl Drop for CountedPort {
    fn drop(&mut self) {
        self.active_ports.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Port for CountedPort {
    fn register<'a>(&'a self, select: &mut Select<'a>) {
        self.inner.register(select);
    }
    fn dispatch(&self, operation: SelectedOperation<'_>) -> bool {
        self.inner.dispatch(operation)
    }
}

pub(crate) struct StorageWorker {
    name: &'static str,
    workers: Mutex<Vec<Worker>>,
}

impl StorageWorker {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self {
            name,
            workers: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn register<T, F>(&self, receiver: GenericReceiver<T>, handler: F) -> io::Result<()>
    where
        T: Serialize + for<'de> Deserialize<'de> + Send + 'static,
        F: FnMut(T) -> bool + Send + 'static,
    {
        let port: Registration = Box::new(StoragePort {
            receiver: receiver.route_preserving_errors(),
            handler: RefCell::new(handler),
        });
        let capacity = thread::available_parallelism().map_or(1, |n| n.get());
        let mut workers = self
            .workers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let least = workers
            .iter()
            .enumerate()
            .min_by_key(|(_, worker)| worker.active_ports.load(Ordering::Relaxed))
            .map(|(index, _)| index);
        let busy =
            least.is_none_or(|index| workers[index].active_ports.load(Ordering::Relaxed) > 0);
        if workers.len() < capacity && busy {
            let (sender, receiver) = crossbeam_channel::unbounded();
            match thread::Builder::new()
                .name(format!("{}#{}", self.name, workers.len()))
                .spawn(move || run(receiver))
            {
                Ok(_) => workers.push(Worker {
                    registrations: sender,
                    active_ports: Arc::new(AtomicUsize::new(0)),
                }),
                // Another shard is an optimization. Preserve an existing live
                // shard when the kernel cannot allocate another thread.
                Err(error) if workers.is_empty() => return Err(error),
                Err(_) => {}
            }
        }
        let worker = workers
            .iter()
            .min_by_key(|worker| worker.active_ports.load(Ordering::Relaxed))
            .expect("storage worker");
        worker.active_ports.fetch_add(1, Ordering::Relaxed);
        let port: Registration = Box::new(CountedPort {
            inner: port,
            active_ports: Arc::clone(&worker.active_ports),
        });
        worker
            .registrations
            .send(port)
            .map_err(|_| io::Error::other("storage worker exited"))
    }
}

fn run(registrations: Receiver<Registration>) {
    let mut ports: Vec<Registration> = Vec::new();
    loop {
        // Select borrows the port receivers. Finish that borrow before adding or
        // removing a manager; no state is shared across registered channels.
        enum Event {
            Register(Option<Registration>),
            Retire(usize),
            Keep,
        }
        let event = {
            let mut select = Select::new();
            select.recv(&registrations);
            for port in &ports {
                port.register(&mut select);
            }
            let operation = select.select();
            let index = operation.index();
            if index == 0 {
                Event::Register(operation.recv(&registrations).ok())
            } else if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                ports[index - 1].dispatch(operation)
            }))
            .unwrap_or(false)
            {
                Event::Keep
            } else {
                Event::Retire(index - 1)
            }
        };
        match event {
            Event::Register(Some(port)) => ports.push(port),
            Event::Register(None) => return,
            Event::Retire(index) => {
                ports.swap_remove(index);
            }
            Event::Keep => {}
        }
    }
}

pub(crate) static CLIENT_STORAGE: StorageWorker = StorageWorker::new("ClientStorage");
pub(crate) static INDEXED_DB: StorageWorker = StorageWorker::new("IndexedDB");
pub(crate) static WEB_STORAGE: StorageWorker = StorageWorker::new("WebStorage");
pub(crate) static CACHE_STORAGE: StorageWorker = StorageWorker::new("CacheStorage");
