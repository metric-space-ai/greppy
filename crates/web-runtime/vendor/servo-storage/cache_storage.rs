/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::fmt::Debug;
use std::path::PathBuf;

use log::error;
use servo_base::generic_channel::{self, GenericSender};
use storage_traits::cache_storage::{
    CacheStorageError, CacheStorageThreadHandle, CacheStorageThreadMessage,
    CacheStorageThreadResponse,
};

trait CacheStorageEngine {
    type Error: Debug;

    /// <https://w3c.github.io/ServiceWorker/#cache-storage-has>
    fn has_cache(&mut self, cache_name: &str) -> Result<bool, CacheStorageError<Self::Error>>;
}

pub struct DummyCacheStorageEngine;

impl CacheStorageEngine for DummyCacheStorageEngine {
    type Error = ();

    /// <https://w3c.github.io/ServiceWorker/#cache-storage-has>
    /// The parallel steps.
    fn has_cache(&mut self, _cache_name: &str) -> Result<bool, CacheStorageError<Self::Error>> {
        // TODO: implement.
        // Step 2.1:For each key → value of the relevant name to cache map:
        // Step 2.1.1: If cacheName matches key, resolve promise with true and abort these steps.
        // Step 2.2: Resolve promise with false.
        // Note: promise resolved in the callback in CacheStorage.
        Ok(false)
    }
}

pub trait CacheStorageThreadFactory {
    fn new(config_dir: Option<PathBuf>, temporary_storage: bool) -> Self;
}

impl CacheStorageThreadFactory for CacheStorageThreadHandle {
    fn new(config_dir: Option<PathBuf>, temporary_storage: bool) -> CacheStorageThreadHandle {
        let (generic_sender, generic_receiver) = generic_channel::channel().unwrap();
        let mut temp_dir: Option<tempfile::TempDir> = None;
        let base_dir = config_dir
            .unwrap_or_else(|| {
                let tmp_dir = tempfile::tempdir().unwrap();
                let path = tmp_dir.path().to_path_buf();
                temp_dir = Some(tmp_dir);
                path
            })
            .join("cachestorage");
        let storage_dir = if temporary_storage {
            let unique_id = uuid::Uuid::new_v4().to_string();
            base_dir.join("temporary").join(unique_id)
        } else {
            base_dir.join("default_v1")
        };
        std::fs::create_dir_all(&storage_dir)
            .expect("Failed to create CacheStorage storage directory");
        let sender_clone = generic_sender.clone();
        let mut manager = None;
        crate::shared_workers::CACHE_STORAGE
            .register(generic_receiver, move |message| {
                let _keep_temp_dir = &temp_dir;
                if let CacheStorageThreadMessage::Exit(sender) = message {
                    let _ = sender.send(());
                    return false;
                }
                manager
                    .get_or_insert_with(|| {
                        CacheStorageThread::new(sender_clone.clone(), DummyCacheStorageEngine)
                    })
                    .handle_message(message)
            })
            .expect("CacheStorage worker spawning failed");

        CacheStorageThreadHandle::new(generic_sender)
    }
}

struct CacheStorageThread<E: CacheStorageEngine> {
    // Note: a sender to self might be required later for the storage engine.
    _sender: GenericSender<CacheStorageThreadMessage>,
    engine: E,
}

impl<E> CacheStorageThread<E>
where
    E: CacheStorageEngine,
{
    pub fn new(
        _sender: GenericSender<CacheStorageThreadMessage>,

        engine: E,
    ) -> CacheStorageThread<E> {
        CacheStorageThread { _sender, engine }
    }

    fn handle_message(&mut self, message: CacheStorageThreadMessage) -> bool {
        match message {
            CacheStorageThreadMessage::HasCache {
                cache_name,
                callback,
                proxy: _,
                origin: _,
            } => {
                let result = self.engine.has_cache(&cache_name);
                if callback
                    .send(CacheStorageThreadResponse::HasCacheResult(
                        result.map(|_| false).map_err(|e| format!("{:?}", e)),
                    ))
                    .is_err()
                {
                    error!("Failed to send response to script for HasCache message.");
                }
            }
            CacheStorageThreadMessage::Exit(sender) => {
                let _ = sender.send(());
                return false;
            }
        }
        true
    }
}
