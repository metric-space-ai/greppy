// Invoked inside SELECTOR_RUNTIME's lexical scope on every condition sample.
// Ref validity is not a presence predicate: an invalid ref throws before the
// caller can invert a false value for --absent.
(function(selector, evaluate) {
  const nodes = greppyResolveNodes(selector);
  if (!greppyObservedRefMatches(selector, nodes)) {
    throw new Error('STALE_REF: observed node no longer belongs to the active document');
  }
  return evaluate(nodes);
})
