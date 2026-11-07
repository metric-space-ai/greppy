// SPDX-License-Identifier: MPL-2.0

/// Own a debug copy only when the response event can actually be delivered.
/// The network response keeps its original bytes regardless of observation.
#[inline]
pub(crate) fn response_body_for_devtools(
    body: &[u8],
    channel_present: bool,
    pipeline_present: bool,
    webview_present: bool,
) -> Option<Vec<u8>> {
    if !channel_present || !pipeline_present || !webview_present {
        return None;
    }
    Some(body.to_vec())
}

#[cfg(test)]
mod tests {
    use super::response_body_for_devtools;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static ALLOCATIONS: Cell<(bool, usize, usize)> = const { Cell::new((false, 0, 0)) };
    }

    struct TrackedAllocator;

    fn record(bytes: usize) {
        let _ = ALLOCATIONS.try_with(|state| {
            let (enabled, calls, allocated) = state.get();
            if enabled {
                state.set((true, calls + 1, allocated + bytes));
            }
        });
    }

    // Test-only accounting delegates unchanged to the OS allocator. Tracking
    // is local to the test thread and never allocates or locks inside alloc.
    unsafe impl GlobalAlloc for TrackedAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            // SAFETY: forward the caller's valid allocation layout unchanged.
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() {
                record(layout.size());
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: forward the original allocation and layout unchanged.
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            // SAFETY: the allocator caller supplies the valid old layout and size.
            let resized = unsafe { System.realloc(pointer, layout, size) };
            if !resized.is_null() {
                record(size);
            }
            resized
        }
    }

    #[global_allocator]
    static ALLOCATOR: TrackedAllocator = TrackedAllocator;

    fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
        ALLOCATIONS.with(|state| state.set((true, 0, 0)));
        let value = operation();
        let (calls, bytes) = ALLOCATIONS.with(|state| {
            let (_, calls, bytes) = state.get();
            state.set((false, 0, 0));
            (calls, bytes)
        });
        (value, calls, bytes)
    }

    #[test]
    fn absent_devtools_recipient_allocates_no_body_and_active_recipient_keeps_every_byte() {
        let body = vec![0xa5; 1024 * 1024];
        for channel in [false, true] {
            for pipeline in [false, true] {
                for webview in [false, true] {
                    let (copied, calls, bytes) = measured(|| {
                        response_body_for_devtools(
                            std::hint::black_box(&body),
                            channel,
                            pipeline,
                            webview,
                        )
                    });
                    if channel && pipeline && webview {
                        assert_eq!(
                            copied.as_deref(),
                            Some(body.as_slice()),
                            "active DevTools recipient lost response bytes"
                        );
                        assert_eq!(
                            (calls, bytes),
                            (1, body.len()),
                            "active DevTools response copied its body more than once"
                        );
                    } else {
                        assert_eq!(
                            (calls, bytes),
                            (0, 0),
                            "unobserved HTTP body allocated a DevTools copy"
                        );
                        assert!(copied.is_none(), "an undeliverable debug body was retained");
                    }
                }
            }
        }
        println!(
            "allocation contract: missing recipient = 0 extra bytes; active recipient = {} byte-for-byte bytes",
            body.len()
        );
    }
}
