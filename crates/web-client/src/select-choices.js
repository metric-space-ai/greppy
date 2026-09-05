// Pure, bounded projection for option diagnostics. Callers still own the
// untrusted-page boundary and must omit a null result from node descriptors.
// This helper neither selects an option nor dispatches page events.
function greppySelectChoicesSnapshot(node) {
  if (!node || String(node.tagName || '').toLowerCase() !== 'select') return null;
  const autocomplete = String(node.getAttribute('autocomplete') || '').toLowerCase().split(/\s+/);
  if (autocomplete.some(token => [
    'current-password', 'new-password', 'one-time-code', 'cc-number', 'cc-csc',
  ].includes(token))) return null;

  const options = node.options;
  const total = options.length;
  if (!Number.isSafeInteger(total) || total < 0) {
    throw new Error('select options are unavailable');
  }
  const selectDisabled = !!node.disabled || node.getAttribute('aria-disabled') === 'true'
    || node.matches(':disabled');
  const choices = [];
  // Count Unicode code points without allocating an array for the entire
  // value. A truncated value is NOT an actionable value: never publish it.
  function bounded(text, redactIfTruncated) {
    let prefix = '';
    let count = 0;
    for (const character of String(text)) {
      if (count === 160) {
        return { text: redactIfTruncated ? null : prefix, truncated: true };
      }
      prefix += character;
      count++;
    }
    return { text: prefix, truncated: false };
  }
  for (let index = 0; index < Math.min(total, 8); index++) {
    const option = options[index];
    const group = option.parentElement;
    const groupDisabled = group && String(group.tagName).toLowerCase() === 'optgroup'
      && (group.disabled || group.getAttribute('aria-disabled') === 'true');
    const value = bounded(option.value, true);
    const label = bounded(option.label, false);
    choices.push({
      value: value.text,
      label: label.text,
      disabled: !!(selectDisabled || option.disabled || groupDisabled),
      value_truncated: value.truncated,
      label_truncated: label.truncated,
    });
  }
  return {
    schema: 'greppy.web.select-choices.v1',
    choices,
    choices_total: total,
    choices_truncated: total > choices.length,
  };
}
