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

function greppyIsLabelable(control) {
  return !!control && (['BUTTON', 'METER', 'OUTPUT', 'PROGRESS', 'SELECT', 'TEXTAREA'].includes(control.tagName)
    || (control.tagName === 'INPUT' && control.type !== 'hidden'));
}

function greppyControlForLabel(label) {
  if (label.hasAttribute('for')) {
    const control = label.ownerDocument.getElementById(label.getAttribute('for'));
    return greppyIsLabelable(control) ? control : null;
  }
  return Array.from(label.querySelectorAll('button,input,meter,output,progress,select,textarea'))
    .find(greppyIsLabelable) || null;
}

function greppyNativeLabels(control) {
  if (!greppyIsLabelable(control)) return [];
  // The engine-backed control.labels collection can stall on Magento grid
  // checkboxes. Resolve HTML label associations from the finite label list.
  return Array.from(control.ownerDocument.querySelectorAll('label'))
    .filter(label => greppyControlForLabel(label) === control);
}
