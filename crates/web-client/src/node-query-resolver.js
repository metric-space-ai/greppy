(function(q) {
  function esc(s) { return String(s).replace(/"/g, '\\"'); }
  var m = /^([a-z]+)(=|~)([\s\S]*)$/.exec(q);
  // A bare CSS sibling combinator such as div~span is not a query prefix.
  if (m && m[2] === "~" && ["css", "xpath", "text", "role", "id", "tag"].indexOf(m[1]) < 0) m = null;
  var kind = m ? m[1] : "css";
  if (["css", "xpath", "text", "role", "id", "tag"].indexOf(kind) < 0) throw new Error("unsupported node query kind: " + kind);
  var op = m ? m[2] : "=";
  if (op === "~" && kind !== "text") throw new Error("regex operator requires a text query");
  var val = m ? m[3] : q;
  function norm(s) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim(); }
  function reOf(v) {
    var r = /^\/([\s\S]*)\/([imsu]*)$/.exec(v);
    return r ? new RegExp(r[1], r[2]) : new RegExp(v);
  }
  if (kind === "css") return Array.prototype.slice.call(document.querySelectorAll(val));
  if (kind === "xpath") {
    var out = [], it = document.evaluate(val, document, null, 5, null), n;
    while ((n = it.iterateNext())) out.push(n);
    return out;
  }
  if (kind === "id") return Array.prototype.slice.call(document.querySelectorAll(String.fromCharCode(35) + val));
  if (kind === "tag") return Array.prototype.slice.call(document.getElementsByTagName(val));
  var all = Array.prototype.slice.call(document.querySelectorAll("*"));
  if (kind === "role") {
    return all.filter(function (e) {
      var r = e.getAttribute("role");
      if (r) return r === val;
      var t = e.tagName.toLowerCase();
      if (val === "button") return t === "button" || (t === "input" && /^(button|submit|reset)$/.test(e.type || ""));
      if (val === "link") return t === "a" && e.hasAttribute("href");
      if (val === "textbox") return t === "textarea" || (t === "input" && !/^(button|submit|reset|checkbox|radio|file)$/.test(e.type || ""));
      if (val === "checkbox") return t === "input" && e.type === "checkbox";
      if (val === "dialog") return t === "dialog";
      if (val === "heading") return /^h[1-6]$/.test(t);
      return false;
    });
  }
  if (kind === "text") {
    if (op === "~") { var re = reOf(val); return all.filter(function (e) { return re.test(norm(e.textContent)); }); }
    return all.filter(function (e) { return norm(e.textContent) === norm(val); });
  }
  return [];
})
