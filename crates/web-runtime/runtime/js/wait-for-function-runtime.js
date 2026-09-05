function greppyWaitForFunction(source, token, timeoutMs) {
  var key = "__greppyWait_" + token;
  var finished = false;
  var signaled = false;
  var observer = null;
  var deadlineId = 0;
  var rafId = 0;
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
      finish("error", String(err && err.message ? err.message : err));
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
