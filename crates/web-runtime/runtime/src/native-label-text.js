// Label text must not contain the labelled control's own subtree (e.g. all
// options of a wrapping select). Keep the live DOM unchanged and keep option
// values in the separate form-state fields, not in the control's name.
function greppyNativeLabelText(label, control) {
  if (!control) return label.textContent || '';
  // Traverse only descendants of this label. Native TreeWalker can escape
  // an empty root and keep visiting unrelated page text in the engine.
  const pending = Array.from(label.childNodes).reverse();
  let text = '';
  while (pending.length) {
    const node = pending.pop();
    if (node === control) continue;
    if (node.nodeType === 3) {
      text += node.nodeValue || '';
    } else {
      for (let child = node.lastChild; child; child = child.previousSibling) {
        pending.push(child);
      }
    }
  }
  return text;
}

function greppyControlForLabel(label) {
  if (label.control) return label.control;
  if (label.htmlFor) return label.ownerDocument.getElementById(label.htmlFor);
  return label.querySelector('input, textarea, select, button');
}
