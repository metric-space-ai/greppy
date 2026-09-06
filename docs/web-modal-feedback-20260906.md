# Modal scope and asynchronous action feedback: trace-backed work

The public S08 A1 and C5 tool results expose two separate problems. This is a
concrete development comparison, not causal attribution of provider-token totals.
The evidence contains only public tool outputs, no private reasoning.

After opening Reserve Ember, A1 receives the active reservation-dialog container,
reservation-form, heading, unit price, Quantity, Confirm and Cancel, with focus on
Quantity. C5 receives nine flat controls: the region/capacity/sort controls, three
background Reserve buttons and three dialog controls. It repeats background table
text and option values, without a dialog/form container or focus relationship.
This establishes a missing working-view distinction, not missing native DOM data.

C5's subsequent `web do fill @1001 3 :: click @1002` returns an observation with the
same open dialog and Confirm control, revision 3 and no reservation. The trace then
contains another Confirm, a timeout and an observation showing revision 4 and the
correct reservation. The independent oracle passes. Delivered input and eventual
application outcome are different milestones; returning this intermediate state
encourages another mutation when an observation or expected-condition wait is
needed. C5 obeys the transport/poll instructions, so transport misuse alone cannot
explain this sequence. It does not prove the model's internal motive.

The next implementation must expose native-confirmed active modal scope, focus
and form relationships. Prioritize controls that are actually available; summarize
inert background content with explicit expansion. An open nonmodal dialog must not
hide the rest of the page. Preserve explicit scopes, frame/document identities,
ambiguity failures and reference invalidation. Do not infer modality from a title,
CSS naming convention or merely `dialog[open]`.

For asynchronous completion, preserve the input-delivery receipt and use an
explicit expected condition to await the application result through native events.
Do not invent a successful reservation, auto-retry a mutation, add blind sleeps or
wait for all network traffic. Without an expectation, report the observed state
and its limits; a generic delivered receipt is not a business-success assertion.
The CLI's proposed action `--expect` is not implemented in this checkpoint.

Acceptance before a fresh Luna series: modal and nonmodal controls; nested dialogs;
focus changes; replaced/stale nodes; background ambiguity; delayed submit success;
validation failure; no-change actions; timeout without duplicate mutation; frames
and continuous network activity. Then measure actual input/output tokens with
identical Luna settings, retaining failures. Output bytes alone cannot close this
performance bug. Do not change frozen S08 inputs or outputs.

Public evidence:
`/Users/michaelwelsch/.local/state/greppy-web-study/modal-feedback-20260906/public-evidence.json`.
A1 call IDs `call_Tz5lLIYXW4qa5mQub24PO4wv` and `call_Llwtf4g178IAW0qrzuHhqW8P`;
C5 `call_dFReeXuNKRLqcgwYTNqDjHGr` and `call_0fGjNXI8F1skzUwWqwJWempb`.
Original source records and paths are retained in the artifact. Both findings were
reported to the designated existing fix task; no frozen executable was modified.
