// Expression embedded into the observation script, not a page-global API.
// Only WeakMap keys retain node identity: detached nodes are not kept alive by
// the registry. DOM attributes are lookup hints and never prove identity.
(function(document, previous, snapshot, first, last, expectedSnapshot) {
  if (!Number.isSafeInteger(first) || !Number.isSafeInteger(last) || first < 1 || last < first) {
    throw new Error('invalid observed reference allocation');
  }
  // History restoration can bring back a live Document with an old registry.
  // Only retain identity in the scope the supervisor still recognizes. A
  // restored/expired scope gets fresh references rather than resurrecting old
  // handles or making every later observation fail scope validation.
  const reuse = previous && previous.document === document &&
    previous.snapshot === expectedSnapshot && previous.identities instanceof WeakMap;
  if (reuse && first <= previous.lastReserved) {
    throw new Error('observed reference allocation would recycle identifiers');
  }
  const identities = reuse ? previous.identities : new WeakMap();
  const token = reuse ? previous.snapshot : snapshot;
  let next = first;
  return {
    document: document,
    snapshot: token,
    identities: identities,
    lastReserved: last,
    refFor: function(node) {
      if (!node || node.ownerDocument !== document || !node.isConnected) {
        throw new Error('cannot observe a detached or foreign document node');
      }
      const existing = identities.get(node);
      if (existing !== undefined) return existing;
      if (next > last) throw new Error('observed reference allocation exhausted');
      const reference = next++;
      identities.set(node, reference);
      return reference;
    },
    matches: function(node, reference) {
      return Number.isSafeInteger(reference) && reference > 0 && !!node &&
        node.ownerDocument === document && node.isConnected && identities.get(node) === reference;
    }
  };
})
