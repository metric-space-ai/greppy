//! JavaScript shims for Web APIs this engine does not implement.
//!
//! A page that touches a missing global dies on a `ReferenceError`, and that
//! one exception takes down everything queued behind it: `react.dev` rendered
//! zero links and a zero-height body purely because `IntersectionObserver` was
//! undefined. The rule this module exists to enforce is that a missing API
//! ends as a *refused call*, never as a `ReferenceError`.
//!
//! Only APIs Servo has no implementation for belong here. Anything Servo ships
//! behind a preference is switched on in [`crate::content_worker`]'s
//! preference setup instead, so pages get the real thing.

/// Source of the shim bundle, injected into every page before its own scripts
/// run.
///
/// Every shim guards on the feature being absent, so enabling a real
/// implementation later silently retires the corresponding stub.
pub fn shim_source() -> &'static str {
    SHIM_JS
}

const SHIM_JS: &str = r#"(function () {
  'use strict';

  // requestIdleCallback: Servo has no implementation at all. Deferred work is
  // better run late than not at all, so this maps onto setTimeout. The
  // deadline reports a plausible budget rather than pretending to measure one.
  if (typeof globalThis.requestIdleCallback !== 'function') {
    var IDLE_BUDGET_MS = 50;
    var nextIdleHandle = 1;
    var idleTimers = new Map();

    globalThis.requestIdleCallback = function requestIdleCallback(callback, options) {
      if (typeof callback !== 'function') {
        throw new TypeError('requestIdleCallback: callback is not a function');
      }
      var handle = nextIdleHandle++;
      var timeout = options && typeof options.timeout === 'number' ? options.timeout : 0;
      var delay = timeout > 0 ? Math.min(timeout, IDLE_BUDGET_MS) : 1;
      var start = Date.now();
      var timer = setTimeout(function () {
        idleTimers.delete(handle);
        callback({
          didTimeout: false,
          timeRemaining: function timeRemaining() {
            return Math.max(0, IDLE_BUDGET_MS - (Date.now() - start));
          },
        });
      }, delay);
      idleTimers.set(handle, timer);
      return handle;
    };

    globalThis.cancelIdleCallback = function cancelIdleCallback(handle) {
      var timer = idleTimers.get(handle);
      if (timer !== undefined) {
        clearTimeout(timer);
        idleTimers.delete(handle);
      }
    };
  }

  // navigator.serviceWorker: an agent has no use for service workers, but a
  // page must not die looking for the container. Registration refuses; the
  // lookups answer empty, which is what a page with no worker installed sees.
  if (globalThis.navigator && !('serviceWorker' in globalThis.navigator)) {
    var unsupported = function unsupported() {
      return Promise.reject(
        new Error('serviceWorker is not supported in this runtime')
      );
    };
    var container = {
      controller: null,
      // Never settles: that is what `ready` does until a worker becomes
      // active, and no worker ever will here.
      ready: new Promise(function () {}),
      register: unsupported,
      getRegistration: function getRegistration() {
        return Promise.resolve(undefined);
      },
      getRegistrations: function getRegistrations() {
        return Promise.resolve([]);
      },
      startMessages: function startMessages() {},
      addEventListener: function addEventListener() {},
      removeEventListener: function removeEventListener() {},
      dispatchEvent: function dispatchEvent() {
        return false;
      },
      oncontrollerchange: null,
      onmessage: null,
      onmessageerror: null,
    };
    Object.defineProperty(globalThis.navigator, 'serviceWorker', {
      value: container,
      configurable: true,
      enumerable: true,
    });
  }
})();
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shim_defines_every_api_the_bench_found_missing() {
        let source = shim_source();
        for symbol in [
            "requestIdleCallback",
            "cancelIdleCallback",
            "serviceWorker",
        ] {
            assert!(
                source.contains(symbol),
                "shim bundle does not mention {symbol}"
            );
        }
    }

    #[test]
    fn shim_yields_to_a_real_implementation() {
        // Each shim must be guarded, so switching on a real implementation
        // retires the stub instead of being shadowed by it.
        let source = shim_source();
        assert!(source.contains("typeof globalThis.requestIdleCallback !== 'function'"));
        assert!(source.contains("!('serviceWorker' in globalThis.navigator)"));
    }

    #[test]
    fn shim_does_not_stub_what_servo_implements_behind_a_pref() {
        // IntersectionObserver is real in Servo and switched on by preference.
        // A stub here would shadow the real implementation and silently render
        // nothing observable.
        assert!(!shim_source().contains("IntersectionObserver"));
    }
}
