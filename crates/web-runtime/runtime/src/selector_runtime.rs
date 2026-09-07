//! Selector runtime and observed-ref condition source shared by the daemon
//! and the Servo content worker. This module has no engine dependency so the
//! daemon builds without the optional `content-runtime` feature; the JavaScript
//! it emits is unchanged.

/// Use exactly the native locator's node-identity check for condition refs.
/// The daemon supplies a selector bound to one session/page snapshot, never a
/// page-provided CSS recipe. The closure runs again for each wait sample.
pub(crate) fn observed_ref_condition_source(source: &str, selector: &serde_json::Value) -> String {
    let guard = include_str!("observed-ref-condition.js");
    format!("(function(){{ {SELECTOR_RUNTIME} return ({guard})({selector}, function(__greppyConditionNodes){{ return ({source}); }}); }})()")
}

pub(crate) const SELECTOR_RUNTIME: &str = concat!(
    include_str!("native-label-text.js"),
    r#"
function greppyAccessibleName(el) {
  const labelled = el.getAttribute('aria-label');
  if (labelled) return labelled.trim();
  const labels = Array.from(el.labels || []);
  if (labels.length) {
    return labels.map((label) => greppyNativeLabelText(label, el)).join(' ').trim();
  }
  return ((el.innerText || el.textContent || el.value || '') + '').trim();
}
function greppyRoleOf(el) {
  const explicit = el.getAttribute('role');
  if (explicit) return explicit;
  const tag = el.tagName.toLowerCase();
  if (tag === 'button') return 'button';
  if (tag === 'input' && (el.type === 'button' || el.type === 'submit' || el.type === 'reset')) return 'button';
  if (tag === 'a' && el.hasAttribute('href')) return 'link';
  if (tag === 'input' || tag === 'textarea') return 'textbox';
  return tag;
}
function greppyIsDisplayed(el) {
  var n = el;
  while (n && n.nodeType === 1) {
    var style = getComputedStyle(n);
    if (style && (style.display === "none" || style.visibility === "hidden" || style.visibility === "collapse")) {
      return false;
    }
    n = n.parentElement;
  }
  var rect = el.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}
function greppyQueryAll(root, sel) {
  var visible = null;
  var css = String(sel);
  if (css.indexOf(":visible") !== -1) {
    visible = true;
    css = css.split(":visible").join("");
  }
  if (css.indexOf(":hidden") !== -1) {
    visible = false;
    css = css.split(":hidden").join("");
  }
  css = css.replace(/\s{2,}/g, " ").trim();
  try {
    var ctx = root === document ? document : root;
    var nodes = css ? Array.from(ctx.querySelectorAll(css)) : [];
    if (visible === null) return nodes;
    return nodes.filter(function (el) {
      var shown = greppyIsDisplayed(el);
      return visible ? shown : !shown;
    });
  } catch (error) { return []; }
}
function greppyCandidates(root) {
  return greppyQueryAll(root === document ? document : root, '*');
}
function greppyResolveIn(root, selector) {
  if (selector.type === 'css') {
    return greppyQueryAll(root === document ? document : root, selector.value);
  }
  if (selector.type === 'xpath') {
    try {
      const ctx = root === document ? document : root;
      const result = document.evaluate(
        selector.value,
        ctx,
        null,
        XPathResult.ORDERED_NODE_SNAPSHOT_TYPE,
        null
      );
      const nodes = [];
      for (let i = 0; i < result.snapshotLength; i++) {
        nodes.push(result.snapshotItem(i));
      }
      return nodes;
    } catch (error) {
      return [];
    }
  }
  if (selector.type === 'label') {
    const labels = greppyQueryAll(root === document ? document : root, 'label');
    const match = labels.find((label) => greppyNativeLabelText(label, greppyControlForLabel(label)).trim() === selector.name);
    if (!match) return [];
    if (match.control) return [match.control];
    if (match.htmlFor) {
      const el = document.getElementById(match.htmlFor);
      return el ? [el] : [];
    }
    const nested = match.querySelector("input, textarea, select, button");
    return nested ? [nested] : [];
  }
  const pool = greppyCandidates(root);
  if (selector.type === 'role') {
    return pool.filter((el) => {
      if (greppyRoleOf(el) !== selector.role) return false;
      if (selector.name == null) return true;
      return greppyAccessibleName(el) === selector.name;
    });
  }
  if (selector.type === 'text') {
    const wanted = selector.value;
    return pool.filter((el) => ((el.innerText || el.textContent || '') + '').trim() === wanted);
  }
  if (selector.type === 'placeholder') {
    return pool.filter((el) => (el.getAttribute('placeholder') || '') === selector.name);
  }
  if (selector.type === 'alt') {
    return pool.filter((el) => (el.getAttribute('alt') || '') === selector.name);
  }
  if (selector.type === 'title') {
    return pool.filter((el) => (el.getAttribute('title') || '') === selector.name);
  }
  if (selector.type === 'testid') {
    const attr = selector.attr || 'data-testid';
    return pool.filter((el) => (el.getAttribute(attr) || '') === selector.name);
  }
  if (selector.type === 'filter') {
    return pool;
  }
  if (selector.type === 'framecss' || selector.type === 'frametext' || selector.type === 'framerole') {
    const framesAll = greppyQueryAll(root === document ? document : root, selector.frame || 'iframe');
    let frames = framesAll;
    if (selector.frameIndex != null) {
      const idx = selector.frameIndex < 0 ? framesAll.length + selector.frameIndex : selector.frameIndex;
      const frame = framesAll[idx];
      frames = frame ? [frame] : [];
    }
    let nodes = [];
    for (let i = 0; i < frames.length; i++) {
      try {
        const doc = frames[i].contentDocument;
        if (!doc) continue;
        if (selector.type === 'framecss') {
          nodes = nodes.concat(Array.from(doc.querySelectorAll(selector.value)));
        } else if (selector.type === 'frametext') {
          const wanted = selector.value;
          nodes = nodes.concat(Array.from(doc.querySelectorAll('body *')).filter((el) => {
            return ((el.innerText || el.textContent || '') + '').trim() === wanted;
          }));
        } else {
          const role = selector.role;
          const name = selector.name;
          nodes = nodes.concat(Array.from(doc.querySelectorAll('body *')).filter((el) => {
            if (greppyRoleOf(el) !== role) return false;
            if (name == null) return true;
            return greppyAccessibleName(el) === name;
          }));
        }
      } catch (error) {}
    }
    return nodes;
  }
  return [];
}
function greppyObservedRefMatches(selector, nodes) {
  if (selector.snapshot == null) return true;
  const registry = window.__greppyObservedRefs;
  return !!(registry && registry.snapshot === selector.snapshot &&
    document.documentElement && document.documentElement.getAttribute('data-greppy-ref-snapshot') === selector.snapshot &&
    nodes.length === 1 && registry.matches(nodes[0], selector.observed_ref) &&
    nodes[0].ownerDocument === document && nodes[0].isConnected);
}
function greppyResolveNodes(selector) {
  if (selector.type === 'filter') {
    let nodes = greppyResolveNodes(selector.scope);
    if (selector.hasText) {
      const wanted = String(selector.hasText);
      nodes = nodes.filter((el) => ((el.innerText || el.textContent || '') + '').indexOf(wanted) !== -1);
    }
    if (selector.has) {
      nodes = nodes.filter((el) => greppyResolveIn(el, selector.has).length > 0);
    }
    if (selector.hasNot) {
      nodes = nodes.filter((el) => greppyResolveIn(el, selector.hasNot).length === 0);
    }
    if (selector.nth != null) {
      const idx = selector.nth < 0 ? nodes.length + selector.nth : selector.nth;
      const el = nodes[idx];
      return el ? [el] : [];
    }
    return nodes;
  }
  let roots = [document];
  if (selector.scope) {
    roots = greppyResolveNodes(selector.scope);
    if (!roots.length) return [];
  }
  let nodes = [];
  for (let i = 0; i < roots.length; i++) {
    nodes = nodes.concat(greppyResolveIn(roots[i], selector));
  }
  if (selector.nth != null) {
    const idx = selector.nth < 0 ? nodes.length + selector.nth : selector.nth;
    const el = nodes[idx];
    return el ? [el] : [];
  }
  return nodes;
}
"#
);
