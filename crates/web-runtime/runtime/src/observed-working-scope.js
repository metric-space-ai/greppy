// Read-only context selection. ARIA is a page declaration, not proof that the
// engine makes the background inert. An open non-modal dialog is not a modal.
function greppyWorkingScopePlan(document, visible) {
  const focus = document.activeElement;
  const focused = focus && focus !== document.body && focus !== document.documentElement
    ? focus : null;
  let native = [];
  let nativeSupported = true;
  try { native = Array.from(document.querySelectorAll(':modal')).filter(visible); }
  catch (_) { nativeSupported = false; }
  const declared = Array.from(document.querySelectorAll('[aria-modal="true"]')).filter(visible);
  const candidates = native.length ? native : declared;
  const containingFocus = focused ? candidates.filter(node => node.contains(focused)) : [];
  // Prefer the innermost focused modal, not DOM order masquerading as top-layer
  // order. Multiple unrelated unfocused modals are reported as ambiguous.
  const innermost = containingFocus.filter(node =>
    !containingFocus.some(other => other !== node && node.contains(other)));
  const scope = innermost.length === 1 ? innermost[0]
    : candidates.length === 1 ? candidates[0] : null;
  const ancestry = [];
  for (let node = focused && focused.parentElement; node; node = node.parentElement) {
    const role = node.getAttribute('role');
    if (node === scope || node.tagName === 'FORM' || node.tagName === 'DIALOG' ||
        role === 'form' || role === 'dialog' || role === 'alertdialog') ancestry.push(node);
  }
  const retainedAncestry = ancestry.slice(0, 8).reverse();
  const extraNodes = Array.from(new Set([scope, focused, ...retainedAncestry].filter(Boolean)));
  return {
    scope, focused, ancestry: retainedAncestry, extraNodes,
    ancestryTruncated: ancestry.length > retainedAncestry.length,
    nativeSupported,
    candidateCount: candidates.length,
    provenance: scope ? (native.length ? 'native_modal' : 'declared_aria_modal') : null,
    ambiguous: !scope && candidates.length > 1
  };
}

function greppyWorkingScopeSnapshot(plan, candidates, capped, actionables, describe) {
  const root = plan.scope;
  const isForeground = node => !root || root.contains(node);
  const descriptor = root ? describe(root) : null;
  const text = root ? String(root.innerText || '') : '';
  return {
    schema: 'greppy.web.working-scope.v1',
    kind: root ? 'modal' : plan.ambiguous ? 'ambiguous' : 'page',
    scope_ref: descriptor ? descriptor.ref : null,
    role: descriptor ? descriptor.role : null,
    name: descriptor ? descriptor.name : null,
    provenance: plan.provenance,
    native_modal_detection: plan.nativeSupported ? 'supported' : 'unavailable',
    modal_candidates: plan.candidateCount,
    focus_ref: plan.focused ? describe(plan.focused).ref : null,
    focus_source: 'document.activeElement',
    ancestry: plan.ancestry.map(describe),
    ancestry_truncated: plan.ancestryTruncated,
    actionable_refs: actionables.filter((_, i) => isForeground(capped[i])).map(item => item.ref),
    background_count: candidates.filter(node => !isForeground(node)).length,
    background_returned: capped.filter(node => !isForeground(node)).length,
    background_location: 'snapshot.actionables',
    text: root ? text.slice(0, 4000) : null,
    text_truncated: text.length > 4000
  };
}
