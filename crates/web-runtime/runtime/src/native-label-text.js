// Label text must not contain the labelled control's own subtree (e.g. all
// options of a wrapping select). Keep the live DOM unchanged and keep option
// values in the separate form-state fields, not in the control's name.
function greppyNativeLabelText(label, control) {
  if (!control) return label.textContent || '';
  const walker = label.ownerDocument.createTreeWalker(label, NodeFilter.SHOW_TEXT);
  let text = '';
  while (walker.nextNode()) {
    if (!control.contains(walker.currentNode)) {
      text += walker.currentNode.nodeValue || '';
    }
  }
  return text;
}

function greppyControlForLabel(label) {
  if (label.control) return label.control;
  if (label.htmlFor) return label.ownerDocument.getElementById(label.htmlFor);
  return label.querySelector('input, textarea, select, button');
}
