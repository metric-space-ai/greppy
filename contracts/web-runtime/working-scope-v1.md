# Observed working scope v1

Native observations add `working_scope` with schema
`greppy.web.working-scope.v1`. It is page-derived, untrusted observed context,
not authorization, proof of actionability, or an action-completion result.

- `kind` is `modal`, `page`, or `ambiguous`. `native_modal` provenance comes
  from the engine's `:modal` matching. `declared_aria_modal` comes only from
  visible `aria-modal="true"` declarations and does not prove inertness.
  A dialog opened with `show()` or `open` alone is not classified as modal.
- A focused innermost modal is preferred. Multiple unrelated modal candidates
  without a uniquely contained focus remain ambiguous; DOM order is not
  represented as top-layer order. Unsupported `:modal` matching is disclosed.
- `scope_ref`, `role`, `name`, `focus_ref` and bounded `ancestry` describe the
  current document context. Focus provenance is `document.activeElement`:
  this version does not claim deep cross-frame/shadow-root focus discovery.
- Scope/focus/ancestor refs use the same identity registry and bounded range
  as control refs. Up to ten slots are reserved for this context before
  capping controls. Existing truncation reporting remains in force.
- `actionable_refs` selects controls inside the scope. Background controls
  remain in `snapshot.actionables`, with `background_count` and
  `background_returned`; original page text, headings and links also remain.
  A compact renderer may foreground the scope but must preserve access to
  the complete returned snapshot. Modal text is bounded to 4000 characters
  and reports truncation.

The native integration regression distinguishes non-modal, native-modal and
ARIA-declared states, validates focus/form ancestry and container ref lookup,
preserves background controls and checks closing the modal. Pure policy tests
also cover ambiguity, nested focus, unsupported selectors and allocation bounds.
Neither test set substitutes for a complete package or three-platform gate.
