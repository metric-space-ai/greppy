(function(e, withAttrs) {
  var r = e.getBoundingClientRect();
  var out = {
    tag: e.tagName.toLowerCase(),
    id: e.id || null,
    text: String(e.textContent == null ? "" : e.textContent).replace(/\s+/g, " ").trim().slice(0, 120),
    visible: !!(r.width || r.height) && getComputedStyle(e).visibility !== "hidden" && getComputedStyle(e).display !== "none",
    box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) }
  };
  if (e.value !== undefined) out.value = e.value;
  if (e.checked !== undefined) out.checked = e.checked;
  if (e.disabled !== undefined) out.disabled = e.disabled;
  if (e.href) out.href = e.href;
  if (withAttrs) {
    out.attrs = {};
    for (var i = 0; i < e.attributes.length; i++) out.attrs[e.attributes[i].name] = e.attributes[i].value;
  }
  return out;
})
