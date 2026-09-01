function greppyKbState() {
  if (!window.__greppyKb) {
    window.__greppyKb = {
      mods: { Shift: false, Control: false, Alt: false, Meta: false },
      pressed: Object.create(null),
    };
  }
  return window.__greppyKb;
}

function greppySmartKey(name) {
  if (name === "ControlOrMeta") {
    return /Mac|iPhone|iPad/.test(String(navigator.platform || "")) ? "Meta" : "Control";
  }
  return name;
}

function greppyLookupKey(name) {
  name = greppySmartKey(String(name));
  var special = {
    Enter: { key: "Enter", code: "Enter", keyCode: 13, location: 0, text: "\r", inputType: "insertLineBreak" },
    "\n": { key: "Enter", code: "Enter", keyCode: 13, location: 0, text: "\r", inputType: "insertLineBreak" },
    "\r": { key: "Enter", code: "Enter", keyCode: 13, location: 0, text: "\r", inputType: "insertLineBreak" },
    Shift: { key: "Shift", code: "ShiftLeft", keyCode: 16, location: 1 },
    ShiftLeft: { key: "Shift", code: "ShiftLeft", keyCode: 16, location: 1 },
    ShiftRight: { key: "Shift", code: "ShiftRight", keyCode: 16, location: 2 },
    Control: { key: "Control", code: "ControlLeft", keyCode: 17, location: 1 },
    ControlLeft: { key: "Control", code: "ControlLeft", keyCode: 17, location: 1 },
    ControlRight: { key: "Control", code: "ControlRight", keyCode: 17, location: 2 },
    Alt: { key: "Alt", code: "AltLeft", keyCode: 18, location: 1 },
    AltLeft: { key: "Alt", code: "AltLeft", keyCode: 18, location: 1 },
    AltRight: { key: "Alt", code: "AltRight", keyCode: 18, location: 2 },
    Meta: { key: "Meta", code: "MetaLeft", keyCode: 91, location: 1 },
    MetaLeft: { key: "Meta", code: "MetaLeft", keyCode: 91, location: 1 },
    MetaRight: { key: "Meta", code: "MetaRight", keyCode: 93, location: 2 },
    Backspace: { key: "Backspace", code: "Backspace", keyCode: 8, location: 0, inputType: "deleteContentBackward" },
    Delete: { key: "Delete", code: "Delete", keyCode: 46, location: 0, inputType: "deleteContentForward" },
    Tab: { key: "Tab", code: "Tab", keyCode: 9, location: 0 },
    Escape: { key: "Escape", code: "Escape", keyCode: 27, location: 0 },
    Space: { key: " ", code: "Space", keyCode: 32, location: 0, text: " " },
    " ": { key: " ", code: "Space", keyCode: 32, location: 0, text: " " },
    ArrowLeft: { key: "ArrowLeft", code: "ArrowLeft", keyCode: 37, location: 0 },
    ArrowUp: { key: "ArrowUp", code: "ArrowUp", keyCode: 38, location: 0 },
    ArrowRight: { key: "ArrowRight", code: "ArrowRight", keyCode: 39, location: 0 },
    ArrowDown: { key: "ArrowDown", code: "ArrowDown", keyCode: 40, location: 0 },
    Home: { key: "Home", code: "Home", keyCode: 36, location: 0 },
    End: { key: "End", code: "End", keyCode: 35, location: 0 },
    PageUp: { key: "PageUp", code: "PageUp", keyCode: 33, location: 0 },
    PageDown: { key: "PageDown", code: "PageDown", keyCode: 34, location: 0 },
    Insert: { key: "Insert", code: "Insert", keyCode: 45, location: 0 },
    "!": { key: "!", code: "Digit1", keyCode: 49, location: 0, text: "!" },
    "@": { key: "@", code: "Digit2", keyCode: 50, location: 0, text: "@" },
    "#": { key: "#", code: "Digit3", keyCode: 51, location: 0, text: "#" },
    $: { key: "$", code: "Digit4", keyCode: 52, location: 0, text: "$" },
    "%": { key: "%", code: "Digit5", keyCode: 53, location: 0, text: "%" },
    "^": { key: "^", code: "Digit6", keyCode: 54, location: 0, text: "^" },
    "&": { key: "&", code: "Digit7", keyCode: 55, location: 0, text: "&" },
    "*": { key: "*", code: "Digit8", keyCode: 56, location: 0, text: "*" },
    "(": { key: "(", code: "Digit9", keyCode: 57, location: 0, text: "(" },
    ")": { key: ")", code: "Digit0", keyCode: 48, location: 0, text: ")" },
    "-": { key: "-", code: "Minus", keyCode: 189, location: 0, text: "-" },
    _: { key: "_", code: "Minus", keyCode: 189, location: 0, text: "_" },
    "=": { key: "=", code: "Equal", keyCode: 187, location: 0, text: "=" },
    "+": { key: "+", code: "Equal", keyCode: 187, location: 0, text: "+" },
    "`": { key: "`", code: "Backquote", keyCode: 192, location: 0, text: "`" },
    "~": { key: "~", code: "Backquote", keyCode: 192, location: 0, text: "~" },
    "[": { key: "[", code: "BracketLeft", keyCode: 219, location: 0, text: "[" },
    "{": { key: "{", code: "BracketLeft", keyCode: 219, location: 0, text: "{" },
    "]": { key: "]", code: "BracketRight", keyCode: 221, location: 0, text: "]" },
    "}": { key: "}", code: "BracketRight", keyCode: 221, location: 0, text: "}" },
    "\\": { key: "\\", code: "Backslash", keyCode: 220, location: 0, text: "\\" },
    "|": { key: "|", code: "Backslash", keyCode: 220, location: 0, text: "|" },
    ";": { key: ";", code: "Semicolon", keyCode: 186, location: 0, text: ";" },
    ":": { key: ":", code: "Semicolon", keyCode: 186, location: 0, text: ":" },
    "'": { key: "'", code: "Quote", keyCode: 222, location: 0, text: "'" },
    '"': { key: '"', code: "Quote", keyCode: 222, location: 0, text: '"' },
    ",": { key: ",", code: "Comma", keyCode: 188, location: 0, text: "," },
    "<": { key: "<", code: "Comma", keyCode: 188, location: 0, text: "<" },
    ".": { key: ".", code: "Period", keyCode: 190, location: 0, text: "." },
    ">": { key: ">", code: "Period", keyCode: 190, location: 0, text: ">" },
    "/": { key: "/", code: "Slash", keyCode: 191, location: 0, text: "/" },
    "?": { key: "?", code: "Slash", keyCode: 191, location: 0, text: "?" },
  };
  if (special[name]) return Object.assign({}, special[name]);
  if (/^Key[A-Z]$/.test(name)) {
    var letter = name.charAt(3).toLowerCase();
    return {
      key: letter,
      code: name,
      keyCode: name.charCodeAt(3),
      location: 0,
      text: letter,
    };
  }
  if (/^Digit[0-9]$/.test(name)) {
    var digit = name.charAt(5);
    return { key: digit, code: name, keyCode: 48 + Number(digit), location: 0, text: digit };
  }
  if (name.length === 1) {
    var ch = name;
    if (ch >= "a" && ch <= "z") {
      return { key: ch, code: "Key" + ch.toUpperCase(), keyCode: ch.toUpperCase().charCodeAt(0), location: 0, text: ch };
    }
    if (ch >= "A" && ch <= "Z") {
      return { key: ch, code: "Key" + ch, keyCode: ch.charCodeAt(0), location: 0, text: ch };
    }
    if (ch >= "0" && ch <= "9") {
      return { key: ch, code: "Digit" + ch, keyCode: ch.charCodeAt(0), location: 0, text: ch };
    }
  }
  return null;
}

function greppyApplyShift(desc, shift) {
  if (!shift) return desc;
  if (desc.key.length === 1 && desc.key >= "a" && desc.key <= "z") {
    var up = desc.key.toUpperCase();
    return Object.assign({}, desc, { key: up, text: up });
  }
  return desc;
}

function greppyHasLayout(ch) {
  return greppyLookupKey(ch) !== null && (ch.length === 1 || ch === "Enter" || ch === "\n" || ch === "\r");
}

function greppyStamp(event, props) {
  var names = Object.keys(props);
  for (var i = 0; i < names.length; i++) {
    (function (name, value) {
      if (value === undefined) return;
      try {
        if (event[name] !== value) {
          Object.defineProperty(event, name, {
            configurable: true,
            get: function () { return value; },
          });
        }
      } catch (_err) {}
    })(names[i], props[names[i]]);
  }
}

function greppyMakeInputEvent(type, cancelable, inputType, data) {
  var event = null;
  var inits = [
    { bubbles: true, cancelable: cancelable, inputType: inputType, data: data },
    { bubbles: true, cancelable: cancelable, data: data },
    { bubbles: true, cancelable: cancelable },
  ];
  if (typeof InputEvent === "function") {
    for (var i = 0; i < inits.length && !event; i++) {
      try {
        event = new InputEvent(type, inits[i]);
      } catch (_err) {}
    }
  }
  if (!event) {
    event = new Event(type, { bubbles: true, cancelable: cancelable });
  }
  var stamp = { cancelable: cancelable };
  if (inputType != null) stamp.inputType = inputType;
  if (data !== undefined) stamp.data = data;
  greppyStamp(event, stamp);
  return event;
}

function greppyKeyEvent(el, type, desc, extra) {
  extra = extra || {};
  var state = greppyKbState();
  var cancelable = extra.cancelable !== false;
  var init = {
    key: desc.key,
    code: desc.code || "",
    keyCode: desc.keyCode || 0,
    which: desc.keyCode || 0,
    bubbles: true,
    cancelable: cancelable,
    shiftKey: !!state.mods.Shift,
    ctrlKey: !!state.mods.Control,
    altKey: !!state.mods.Alt,
    metaKey: !!state.mods.Meta,
    repeat: !!extra.repeat,
    location: desc.location || 0,
  };
  var event = new KeyboardEvent(type, init);
  greppyStamp(event, {
    key: init.key,
    code: init.code,
    keyCode: init.keyCode,
    which: init.which,
    cancelable: cancelable,
    shiftKey: init.shiftKey,
    ctrlKey: init.ctrlKey,
    altKey: init.altKey,
    metaKey: init.metaKey,
    location: init.location,
    repeat: init.repeat,
  });
  return el.dispatchEvent(event);
}

function greppyBeforeInput(el, inputType, data) {
  return el.dispatchEvent(greppyMakeInputEvent("beforeinput", true, inputType, data));
}

function greppyTextInput(el, data) {
  return el.dispatchEvent(greppyMakeInputEvent("textInput", true, null, data));
}

function greppyInput(el, inputType, data) {
  el.dispatchEvent(greppyMakeInputEvent("input", false, inputType, data));
}

function greppyEditable(el) {
  return el && ("value" in el || el.isContentEditable);
}

function greppyInsertChars(el, text) {
  if (!el || !("value" in el)) {
    if (el && el.isContentEditable) {
      el.textContent = String(el.textContent || "") + text;
    }
    return;
  }
  var value = String(el.value || "");
  var start = typeof el.selectionStart === "number" ? el.selectionStart : value.length;
  var end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
  el.value = value.slice(0, start) + text + value.slice(end);
  var pos = start + text.length;
  try {
    el.setSelectionRange(pos, pos);
  } catch (_err) {}
}

function greppyDeleteBackward(el) {
  if (!el || !("value" in el)) return;
  var value = String(el.value || "");
  var start = typeof el.selectionStart === "number" ? el.selectionStart : value.length;
  var end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
  if (start !== end) {
    el.value = value.slice(0, start) + value.slice(end);
  } else if (start > 0) {
    el.value = value.slice(0, start - 1) + value.slice(start);
    start -= 1;
  } else {
    return;
  }
  try {
    el.setSelectionRange(start, start);
  } catch (_err) {}
}

function greppyDeleteForward(el) {
  if (!el || !("value" in el)) return;
  var value = String(el.value || "");
  var start = typeof el.selectionStart === "number" ? el.selectionStart : value.length;
  var end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
  if (start !== end) {
    el.value = value.slice(0, start) + value.slice(end);
  } else {
    el.value = value.slice(0, start) + value.slice(start + 1);
  }
  try {
    el.setSelectionRange(start, start);
  } catch (_err) {}
}

function greppyMoveCaret(el, delta) {
  if (!el || typeof el.selectionStart !== "number") return;
  var start = el.selectionStart + delta;
  if (start < 0) start = 0;
  if (start > String(el.value || "").length) start = String(el.value || "").length;
  try {
    el.setSelectionRange(start, start);
  } catch (_err) {}
}

function greppyCommitInsert(el, text, inputType) {
  inputType = inputType || "insertText";
  if (!greppyBeforeInput(el, inputType, inputType === "insertText" ? text : null)) {
    return false;
  }
  if (text != null && text !== "") {
    if (!greppyTextInput(el, text)) return false;
  }
  greppyInsertChars(el, text || "");
  greppyInput(el, inputType, inputType === "insertText" ? text : null);
  return true;
}

function greppyImplicitSubmit(el) {
  if (!el) return;
  var form = el.form;
  if (!form && el.closest) form = el.closest("form");
  if (!form) return;
  if (typeof form.requestSubmit === "function") {
    try { form.requestSubmit(); return; } catch (_err) {}
  }
  if (typeof form.submit === "function") {
    try { form.submit(); } catch (_err2) {}
  }
}

function greppyDefaultAction(el, desc) {
  if (desc.inputType === "deleteContentBackward") {
    if (!greppyBeforeInput(el, "deleteContentBackward", null)) return;
    greppyDeleteBackward(el);
    greppyInput(el, "deleteContentBackward", null);
    return;
  }
  if (desc.inputType === "deleteContentForward") {
    if (!greppyBeforeInput(el, "deleteContentForward", null)) return;
    greppyDeleteForward(el);
    greppyInput(el, "deleteContentForward", null);
    return;
  }
  if (desc.inputType === "insertLineBreak") {
    var isTextArea = el && String(el.tagName || "").toLowerCase() === "textarea";
    if (!greppyBeforeInput(el, "insertLineBreak", null)) return;
    if (isTextArea) {
      if (!greppyTextInput(el, "\n")) return;
      greppyInsertChars(el, "\n");
      greppyInput(el, "insertLineBreak", null);
      return;
    }
    // Script-dispatched Enter has no default action. Implicit form
    // submission is the one agents actually need (finding 038).
    greppyImplicitSubmit(el);
    return;
  }
  if (desc.key === "ArrowLeft") {
    greppyMoveCaret(el, -1);
    return;
  }
  if (desc.key === "ArrowRight") {
    greppyMoveCaret(el, 1);
    return;
  }
  if (desc.key === "Home" && el && typeof el.selectionStart === "number") {
    try { el.setSelectionRange(0, 0); } catch (_err) {}
    return;
  }
  if (desc.key === "End" && el && typeof el.selectionStart === "number") {
    var n = String(el.value || "").length;
    try { el.setSelectionRange(n, n); } catch (_err) {}
    return;
  }
  if (desc.text && greppyEditable(el)) {
    var mods = greppyKbState().mods;
    if (mods.Control || mods.Alt || mods.Meta) return;
    greppyCommitInsert(el, desc.text, "insertText");
  }
}

function greppySplitChord(key) {
  var keys = [];
  var building = "";
  for (var i = 0; i < key.length; i++) {
    var ch = key.charAt(i);
    if (ch === "+" && building) {
      keys.push(building);
      building = "";
    } else {
      building += ch;
    }
  }
  keys.push(building);
  return keys;
}

function greppyDown(el, name) {
  var desc = greppyLookupKey(name);
  if (!desc) throw new Error('Unknown key: "' + name + '"');
  var state = greppyKbState();
  desc = greppyApplyShift(desc, state.mods.Shift);
  var repeat = !!state.pressed[desc.code];
  state.pressed[desc.code] = true;
  if (desc.key === "Shift" || desc.key === "Control" || desc.key === "Alt" || desc.key === "Meta") {
    state.mods[desc.key] = true;
  }
  if (!greppyKeyEvent(el, "keydown", desc, { repeat: repeat })) {
    return desc;
  }
  if (desc.text || desc.inputType === "insertLineBreak") {
    var kp = Object.assign({}, desc, { keyCode: desc.text ? desc.text.charCodeAt(0) : desc.keyCode });
    if (desc.key === "Enter") kp.keyCode = 13;
    if (desc.key.length === 1 && desc.key >= "A" && desc.key <= "Z") kp.keyCode = desc.keyCode;
    if (!greppyKeyEvent(el, "keypress", kp, { repeat: repeat })) {
      return desc;
    }
  }
  greppyDefaultAction(el, desc);
  return desc;
}

function greppyUp(el, name) {
  var desc = greppyLookupKey(name);
  if (!desc) throw new Error('Unknown key: "' + name + '"');
  var state = greppyKbState();
  desc = greppyApplyShift(desc, state.mods.Shift);
  if (desc.key === "Shift" || desc.key === "Control" || desc.key === "Alt" || desc.key === "Meta") {
    state.mods[desc.key] = false;
  }
  delete state.pressed[desc.code];
  greppyKeyEvent(el, "keyup", desc, {});
  return desc;
}

function greppyPress(el, chord) {
  var tokens = greppySplitChord(String(chord));
  var key = tokens[tokens.length - 1];
  for (var i = 0; i < tokens.length - 1; i++) greppyDown(el, tokens[i]);
  greppyDown(el, key);
  greppyUp(el, key);
  for (var j = tokens.length - 2; j >= 0; j--) greppyUp(el, tokens[j]);
}

function greppyInsertText(el, text) {
  greppyCommitInsert(el, String(text), "insertText");
}

function greppyType(el, text) {
  var chars = String(text);
  for (var i = 0; i < chars.length; i++) {
    var ch = chars.charAt(i);
    if (greppyLookupKey(ch) && ch.length === 1) {
      greppyPress(el, ch);
    } else {
      greppyInsertText(el, ch);
    }
  }
}

function greppyActive() {
  return document.activeElement || document.body;
}
