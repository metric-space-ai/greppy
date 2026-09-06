// DOM scope selection for explicit observation queries. Never substitute the
// document when the query has no visible match. All serialized data lives in
// metadata; nodes and predicates stay private to this observation operation.
function greppyObservationScope(document, query, resolved, visible, rootLimit) {
  if (typeof query !== 'string' || !query.trim()) {
    throw new Error('observation query must not be empty');
  }
  if (!Number.isSafeInteger(rootLimit) || rootLimit < 1) {
    throw new Error('observation root limit must be a positive integer');
  }
  const unique = new Set();
  for (const node of resolved) {
    if (!node || node.nodeType !== 1 || node.ownerDocument !== document) {
      throw new Error('observation query must select elements in the current document');
    }
    unique.add(node);
  }
  const visibleMatches = Array.from(unique).filter(visible);
  const matches = new Set(visibleMatches);
  // A matched ancestor already contains a descendant match. Deduplicate before
  // the root limit, so nested matches neither repeat text nor consume the cap.
  const outermost = visibleMatches.filter(function(node) {
    for (let parent = node.parentElement; parent; parent = parent.parentElement) {
      if (matches.has(parent)) return false;
    }
    return true;
  });
  const roots = outermost.slice(0, rootLimit);
  const includes = function(node) {
    return roots.some(function(root) { return root === node || root.contains(node); });
  };
  return {
    roots: roots,
    includes: includes,
    metadata: {
      schema: 'greppy.web.observation-scope.v1',
      query: query,
      matched_elements: unique.size,
      visible_matches: visibleMatches.length,
      roots_total: outermost.length,
      roots_returned: roots.length,
      roots_truncated: outermost.length > roots.length
    },
    collect: function(nodes, limit) {
      if (!Number.isSafeInteger(limit) || limit < 0) {
        throw new Error('observation item limit must be a nonnegative integer');
      }
      const selected = Array.from(nodes).filter(function(node) {
        return includes(node) && visible(node);
      });
      return { nodes: selected.slice(0, limit), total: selected.length,
        truncated: selected.length > limit };
    }
  };
}
