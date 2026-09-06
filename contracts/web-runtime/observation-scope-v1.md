# Explicit observation query (v1)

Implementation in progress; native browser acceptance is still required.

`greppy web observe [QUERY]` accepts one optional node query, using the same
CSS, XPath, text, text-regex, role, id and tag grammar as `web find`. Quote a
query containing spaces as one shell argument. The CLI normalizes quoted CSS
values before sending the query; the runtime resolves it against the current
document before allocating or modifying observation references.

No query retains the existing whole-page observation. A query selects visible
elements as region roots, including their descendants. Duplicate and nested
roots are coalesced before the 20-root cap. Disjoint roots are retained in
query order up to that cap; this does not silently select the first match.

Text, headings, links and actionables are drawn from the retained regions,
not filtered from an already serialized whole-page observation. Actionable
state retains credential redaction, the 200-reference bound and the existing
node-identity checks. Page URL and title remain page-level context. An explicit
scope omits automatic whole-page working-scope diagnostics.

The result includes `observation_scope` with schema
`greppy.web.observation-scope.v1`, normalized `query`, `matched_elements`,
`visible_matches`, `roots_total`, `roots_returned`, `roots_truncated`,
`text_truncated`, `headings_truncated` and `links_truncated`. `refs_truncated`
continues to describe the actionable cap. Text is capped at 8,000 characters,
headings and links at 20 each. HTML is available only when explicitly requested
and contains the retained roots, never a fallback to `page.content`.
Text and HTML artifact responses retain the scope metadata.

No visible region yields `NO_MATCH`, nonzero exit and an empty scoped
observation. Invalid syntax or a non-element XPath result yields an error.
Neither case may widen the request to the whole page or discard its operand.
Invalid syntax must not replace the current document's reference registry.
The supervisor rejects a scoped response that lacks matching scope evidence.
The CLI applies the same guard before printing a response: an older runtime
which ignores QUERY cannot return a successful whole-page observation. Such
content is discarded and the caller is told to use a matching artifact set.
An existing error retains its code while any unscoped page result is discarded.

Required acceptance: scoped DOM text/links/headings/refs, unfiltered parity,
closed/no-match regions, CSS/XPath parity, text/HTML isolation, invalid query
followed by successful ref use, and node replacement rejecting the old ref.
Component tests with a controlled DOM do not establish native-engine success.
