// Runs beside greppy_web_client::SELECT_CHOICES_JS in the page evaluation scope.
// Validate before mutation; a missing value must not silently clear a select.
function greppySelectOption(node, value) {
  if (!node || String(node.tagName || '').toLowerCase() !== 'select') {
    throw new Error('INVALID_SELECT_TARGET: select requires a select element');
  }
  if (typeof value !== 'string') {
    throw new Error('INVALID_OPTION_VALUE: pass an exact string option value');
  }
  function refuse(code, explanation) {
    const choices = greppySelectChoicesSnapshot(node);
    const detail = choices === null ? '' : '\nUNTRUSTED_PAGE_CONTENT_BEGIN\n'
      + JSON.stringify({ select_choices: choices }) + '\nUNTRUSTED_PAGE_CONTENT_END';
    // Do not echo the requested value: it may be a secret, even when none of
    // the page's autocomplete annotations identifies the field as sensitive.
    throw new Error(code + ': ' + explanation
      + ' Use an exact option value, not its label: greppy web select TARGET VALUE.' + detail);
  }
  let matching = null;
  for (let index = 0; index < node.options.length; index++) {
    if (node.options[index].value === value) {
      matching = node.options[index];
      break;
    }
  }
  if (matching === null) {
    refuse('OPTION_NOT_FOUND', 'no option has the requested value; selection was not changed');
  }
  const group = matching.parentElement;
  const groupDisabled = group && String(group.tagName).toLowerCase() === 'optgroup'
    && (group.disabled || group.getAttribute('aria-disabled') === 'true');
  if (node.disabled || node.matches(':disabled') || node.getAttribute('aria-disabled') === 'true'
      || matching.disabled || groupDisabled) {
    refuse('OPTION_DISABLED', 'the matching option is disabled; selection was not changed');
  }
  function applied() {
    return node.value === value && node.selectedOptions.length === 1
      && node.selectedOptions[0].value === value;
  }
  if (applied()) return true;
  node.value = value;
  if (!applied()) {
    throw new Error('SELECTION_NOT_APPLIED: the select did not retain the requested option; no input/change event was dispatched');
  }
  node.dispatchEvent(new Event('input', { bubbles: true }));
  node.dispatchEvent(new Event('change', { bubbles: true }));
  if (!applied()) {
    // Event handlers own their side effects: do not replay the action or roll
    // the page back. Surface the mismatch instead of certifying false success.
    throw new Error('SELECTION_CHANGED: page event handlers changed the requested selection; inspect the current page before retrying');
  }
  return true;
}
