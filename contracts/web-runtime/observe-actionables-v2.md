# Observed actionable form state (v2)

`web.observe` adds `actionable_schema: "greppy.web.actionable.v2"` to its
result. The outer runtime protocol remains v1. Existing `actionables`, refs,
page text, links, headings and truncation indicators remain available. This
is an additive DOM observation contract, not a complete accessibility tree
or evidence that an application's business operation succeeded.

Each actionable retains `ref`, `tag`, `role`, `name`, `text`, `href` and
`disabled`. Native controls now have implicit roles where defined. Names
use `aria-labelledby` references first, then `aria-label`, associated HTML
labels, applicable element content/button values, then `title`.
`name_source` identifies that source. This implements those naming sources,
not the entire accessible-name computation for every custom widget.

New state fields come from the current DOM properties, not initial markup:

- `type`: native input/button type, otherwise null.
- `value`: current input/textarea/select value; null for non-value elements
  or a redacted value. In particular, a checkbox value of `on` does **not**
  mean it is checked.
- `checked`: checkbox/radio property or ARIA state, including `"mixed"` for
  an indeterminate checkbox/ARIA mixed state; null when not applicable.
- `selected`: explicit ARIA selection state, otherwise null.
- `selected_options`: selected native select options (`value`, `label`),
  otherwise null. This handles multiple selections separately from `value`.
- `expanded`: explicit ARIA expansion state, otherwise null.
- `invalid`: native validity or explicit ARIA invalid state; null when
  neither source applies. Reading validity does not fire validation events.

`disabled` includes native disabled controls/fieldsets and `aria-disabled`.
Refs retain their existing snapshot/document lifetime: observing state does
not permit an old ref to target a replacement node.

Names and values are capped at 160 characters and selected options at 20.
`name_truncated`, `value_truncated` and `selected_options_truncated` explicitly
report those limits. Existing text/ref/page limits still apply; this is not
a claim that the observation contains the whole document.

Password/file controls and controls with credential/payment autocomplete
tokens (`current-password`, `new-password`, `one-time-code`, `cc-number`,
`cc-csc`) have `value: null`, `value_redacted: true` and empty legacy `text`.
Other controls have `value_redacted: false`. This protects recognized form
values; it cannot redact arbitrary secrets a page deliberately copies into
ordinary visible text. Page-derived names and states remain untrusted page
content, inside the existing untrusted-content boundary.
