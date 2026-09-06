function greppyWaitForFunction(source, token, timeoutMs, strictBoolean) {
  var key = "__greppyWait_" + token;
  var finished = false;
  var signaled = false;
  var observer = null;
  var deadlineId = 0;
  var rafId = 0;
  var expirationId = 0;
  var expiresAt = Date.now() + timeoutMs;
  var slot = {
    done: 0,
    status: "",
    value: undefined,
    cleanup: cleanup,
  };
  window[key] = slot;

  function signal(status) {
    if (signaled) {
      return;
    }
    signaled = true;
    try {
      console.debug("__greppyWaitDone:" + token + ":" + status);
    } catch (_e) {}
  }

  function cleanup() {
    try {
      if (observer) observer.disconnect();
    } catch (_e) {}
    observer = null;
    try {
      if (deadlineId) window.clearTimeout(deadlineId);
    } catch (_e) {}
    deadlineId = 0;
    try {
      if (rafId) window.cancelAnimationFrame(rafId);
    } catch (_e) {}
    rafId = 0;
    try {
      if (expirationId) window.clearTimeout(expirationId);
    } catch (_e) {}
    expirationId = 0;
    try {
      window.removeEventListener("hashchange", onEvent);
    } catch (_e) {}
    try {
      window.removeEventListener("popstate", onEvent);
    } catch (_e) {}
    try {
      window.removeEventListener("load", onEvent);
    } catch (_e) {}
  }

  function finish(status, value) {
    if (finished) {
      return;
    }
    finished = true;
    slot.done = 1;
    slot.status = status;
    slot.value = value;
    try {
      window.__waitFinishCount = (window.__waitFinishCount || 0) + 1;
    } catch (_e) {}
    cleanup();
    signal(status);
    if (strictBoolean) {
      // The caller may run out of budget before its final slot-read. Do not
      // retain completed predicate values forever in that case. This expiry
      // only drops bookkeeping; it never reevaluates the page or adds time.
      if (status === "timeout") {
        try { delete window[key]; } catch (_e) {}
      } else {
        expirationId = window.setTimeout(function () {
          expirationId = 0;
          try { delete window[key]; } catch (_e) {}
        }, Math.max(0, expiresAt - Date.now()));
      }
    }
  }

  function onEvent() {
    check();
  }

  function scheduleRaf() {
    // MutationObserver may deliver repeatedly before the next paint. Keep a
    // single outstanding frame; otherwise each delivery starts another
    // recurring chain and cleanup can cancel only the last recorded handle.
    if (finished || rafId) {
      return;
    }
    rafId = window.requestAnimationFrame(function () {
      rafId = 0;
      check();
      if (!finished) {
        scheduleRaf();
      }
    });
  }

  function check() {
    if (finished) {
      return;
    }
    var result;
    try {
      result = eval(source);
    } catch (err) {
      const message = strictBoolean && err && err.name === "SyntaxError"
        ? "INVALID_WAIT_SOURCE: invalid JavaScript predicate or regular expression"
        : String(err && err.message ? err.message : err);
      finish("error", message);
      return;
    }
    if (strictBoolean && typeof result !== "boolean") {
      finish("error", "INVALID_WAIT_PREDICATE: expected a Boolean condition");
      return;
    }
    if (result) {
      finish("ok", result);
    }
  }

  try {
    observer = new MutationObserver(function () {
      check();
      if (!finished) {
        scheduleRaf();
      }
    });
    observer.observe(document.documentElement || document, {
      subtree: true,
      childList: true,
      attributes: true,
      characterData: true,
    });
  } catch (_e) {}

  try {
    window.addEventListener("hashchange", onEvent);
    window.addEventListener("popstate", onEvent);
    window.addEventListener("load", onEvent);
  } catch (_e) {}

  deadlineId = window.setTimeout(function () {
    finish("timeout", "timeout: waitForFunction");
  }, timeoutMs);

  check();
  if (!finished) {
    scheduleRaf();
  }
  if (finished && slot.status === "ok") {
    return slot.value;
  }
  return false;
}
