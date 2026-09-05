# Select-choice diagnostics v1

This additive diagnostic projection is not a selection command. The shared
`greppySelectChoicesSnapshot(node)` implementation lives in
`crates/web-client/src/select-choices.js`, exported as
`greppy_web_client::SELECT_CHOICES_JS`. Native error construction and node
descriptors should reuse it rather than build separate option projections.
Integration into those callers is a separate acceptance step.

A descriptor may embed the result under `select_choices`:

```json
{
  "schema": "greppy.web.select-choices.v1",
  "choices": [
    {
      "value": "ascending",
      "label": "Low to high",
      "disabled": false,
      "value_truncated": false,
      "label_truncated": false
    }
  ],
  "choices_total": 1,
  "choices_truncated": false
}
```

- `value` is the exact option value, including a valid empty string. Labels
  are not aliases for values. Duplicate values are preserved as distinct
  entries; the projection never chooses or disambiguates them.
- At most eight options, in DOM order, are read. `choices_total` is the full
  collection length. `choices_truncated` means additional options were omitted.
  An empty collection has total 0 and is not truncated.
- Text bounds count Unicode code points. A value longer than 160 characters
  is `null` with `value_truncated: true`, never an apparently actionable prefix.
  A long label retains its first 160 characters with `label_truncated: true`.
  Per-field flags do not alter the collection truncation flag.
- `disabled` includes the option, its optgroup, and the select's own/effective
  disabled state. It is a snapshot, not a promise that a later action succeeds.
- Non-select nodes and sensitive autocomplete controls return `null`; callers
  omit the projection. Sensitive fields are not enumerated to derive labels,
  values or counts. The sensitive tokens match existing snapshot policy:
  current-password, new-password, one-time-code, cc-number and cc-csc.
- Page-provided labels/values remain untrusted. Callers must preserve the
  untrusted-page boundary, including when attaching choices to errors. They
  must not interpolate those strings into executable shell or JavaScript.
- Renderers may compact known defaults, but must preserve unknown versions
  and fields. `null` values and truncation flags must remain distinguishable
  from a real empty value and a complete list.

The helper performs no selection, event dispatch, evaluation, DOM mutation,
network request or retry. Its Node tests prove projection bounds and values
using DOM-shaped seams, not native browser correctness. Native option refusal,
CLI chain termination and descriptor rendering require separate tests.
