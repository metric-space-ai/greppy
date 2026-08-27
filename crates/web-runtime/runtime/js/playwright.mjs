const ops = Deno.core.ops;

class TimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = "TimeoutError";
  }
}

function engineCall(method, params) {
  const result = ops.op_engine_call(method, params ?? {});
  if (result && typeof result.then === "function") {
    return result.then(
      (value) => value,
      (error) => {
        const message = String(error && error.message ? error.message : error);
        if (message.includes("timed out") || message.includes("timeout")) {
          throw new TimeoutError(message);
        }
        throw error;
      },
    );
  }
  return result;
}

function locatorParams(locator, extra) {
  const params = {
    page: locator._page._id,
    selector: locator._selector,
    timeout: locator._page._timeout || 30_000,
  };
  if (extra) {
    for (const key of Object.keys(extra)) {
      params[key] = extra[key];
    }
  }
  return params;
}

function unsupported(symbol) {
  return async function unsupportedOperation() {
    const error = new Error(`unsupported_playwright_operation: ${symbol}`);
    error.code = "unsupported_playwright_operation";
    throw error;
  };
}

function throwUnsupported(symbol) {
  const error = new Error(`unsupported_playwright_operation: ${symbol}`);
  error.code = "unsupported_playwright_operation";
  throw error;
}

function refuseLocatorOptions(prefix, options, allowed) {
  if (options == null) {
    return;
  }
  const keys = Object.keys(options).filter((key) => options[key] !== undefined);
  for (const key of keys) {
    if (allowed.indexOf(key) === -1) {
      throwUnsupported(`${prefix}.${key}`);
    }
  }
}

function withUnsupported(target, prefix) {
  return new Proxy(target, {
    get(obj, prop) {
      if (typeof prop === "symbol") {
        return obj[prop];
      }
      if (prop === "then" || prop === "catch" || prop === "finally") {
        return undefined;
      }
      if (String(prop).startsWith("_")) {
        return obj[prop];
      }
      if (prop in obj) {
        return obj[prop];
      }
      return unsupported(`${prefix}.${String(prop)}`);
    },
  });
}

function decodeBase64(binary) {
  if (typeof atob === "function") {
    return Uint8Array.from(atob(binary), (c) => c.charCodeAt(0));
  }
  const alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const clean = String(binary).replace(/[^A-Za-z0-9+/]/g, "");
  const out = [];
  for (let i = 0; i < clean.length; i += 4) {
    const a = alphabet.indexOf(clean[i] || "A");
    const b = alphabet.indexOf(clean[i + 1] || "A");
    const c = alphabet.indexOf(clean[i + 2] || "A");
    const d = alphabet.indexOf(clean[i + 3] || "A");
    const n = (a << 18) | (b << 12) | (c << 6) | d;
    out.push((n >> 16) & 255);
    if (clean[i + 2] && clean[i + 2] !== "=") out.push((n >> 8) & 255);
    if (clean[i + 3] && clean[i + 3] !== "=") out.push(n & 255);
  }
  return Uint8Array.from(out);
}

function hexEncode(bytes) {
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = "";
  for (let i = 0; i < arr.length; i++) {
    out += (arr[i] + 256).toString(16).slice(-2);
  }
  return out;
}

function encodeBase64(bytes) {
  const alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = "";
  for (let i = 0; i < arr.length; i += 3) {
    const a = arr[i];
    const b = i + 1 < arr.length ? arr[i + 1] : 0;
    const c = i + 2 < arr.length ? arr[i + 2] : 0;
    const n = (a << 16) | (b << 8) | c;
    out += alphabet[(n >> 18) & 63];
    out += alphabet[(n >> 12) & 63];
    out += i + 1 < arr.length ? alphabet[(n >> 6) & 63] : "=";
    out += i + 2 < arr.length ? alphabet[n & 63] : "=";
  }
  return out;
}

function encodeUtf8(str) {
  const s = String(str);
  const out = [];
  for (let i = 0; i < s.length; i++) {
    let cp = s.charCodeAt(i);
    if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < s.length) {
      const low = s.charCodeAt(i + 1);
      if (low >= 0xdc00 && low <= 0xdfff) {
        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
        i += 1;
      }
    }
    if (cp < 0x80) {
      out.push(cp);
    } else if (cp < 0x800) {
      out.push(0xc0 | (cp >> 6), 0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
      out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3f), 0x80 | (cp & 0x3f));
    } else {
      out.push(
        0xf0 | (cp >> 18),
        0x80 | ((cp >> 12) & 0x3f),
        0x80 | ((cp >> 6) & 0x3f),
        0x80 | (cp & 0x3f),
      );
    }
  }
  return Uint8Array.from(out);
}

function decodeUtf8(bytes) {
  const arr = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  let out = "";
  for (let i = 0; i < arr.length; ) {
    const b0 = arr[i];
    let cp;
    let n = 1;
    if (b0 < 0x80) {
      cp = b0;
    } else if ((b0 & 0xe0) === 0xc0 && i + 1 < arr.length) {
      cp = ((b0 & 0x1f) << 6) | (arr[i + 1] & 0x3f);
      n = 2;
      if (cp < 0x80 || (arr[i + 1] & 0xc0) !== 0x80) cp = 0xfffd;
    } else if ((b0 & 0xf0) === 0xe0 && i + 2 < arr.length) {
      cp = ((b0 & 0x0f) << 12) | ((arr[i + 1] & 0x3f) << 6) | (arr[i + 2] & 0x3f);
      n = 3;
      if (cp < 0x800 || (arr[i + 1] & 0xc0) !== 0x80 || (arr[i + 2] & 0xc0) !== 0x80) cp = 0xfffd;
    } else if ((b0 & 0xf8) === 0xf0 && i + 3 < arr.length) {
      cp =
        ((b0 & 0x07) << 18) |
        ((arr[i + 1] & 0x3f) << 12) |
        ((arr[i + 2] & 0x3f) << 6) |
        (arr[i + 3] & 0x3f);
      n = 4;
      if (
        cp < 0x10000 ||
        cp > 0x10ffff ||
        (arr[i + 1] & 0xc0) !== 0x80 ||
        (arr[i + 2] & 0xc0) !== 0x80 ||
        (arr[i + 3] & 0xc0) !== 0x80
      ) {
        cp = 0xfffd;
      }
    } else {
      cp = 0xfffd;
    }
    out += String.fromCodePoint(cp);
    i += n;
  }
  return out;
}

function bytesFromFulfillBody(body) {
  if (body == null) {
    return new Uint8Array();
  }
  if (typeof body === "string") {
    return encodeUtf8(body);
  }
  if (body instanceof ArrayBuffer) {
    return new Uint8Array(body.slice(0));
  }
  if (ArrayBuffer.isView(body)) {
    return new Uint8Array(body.buffer, body.byteOffset, body.byteLength);
  }
  if (Array.isArray(body)) {
    return Uint8Array.from(body);
  }
  return encodeUtf8(String(body));
}

function serializeEvaluate(pageFunction, arg) {
  if (typeof pageFunction === "function") {
    const source = pageFunction.toString();
    if (arg === undefined) {
      return `(${source})()`;
    }
    return `(${source})(${JSON.stringify(arg)})`;
  }
  return String(pageFunction);
}

class FileChooser {
  constructor(page, record) {
    this._page = page;
    this._multiple = !!(record && record.multiple);
  }

  isMultiple() {
    return this._multiple;
  }

  page() {
    return this._page;
  }

  async setFiles(files) {
    const selector = this._page._lastFileSelector || 'input[type="file"]';
    return this._page.setInputFiles(selector, files);
  }

  element() {
    const selector = this._page._lastFileSelector || 'input[type="file"]';
    return this._page.locator(selector);
  }
}

class Dialog {
  constructor(page, record) {
    this._page = page;
    this._record = record || { type: "alert", message: "", defaultValue: "" };
  }

  type() {
    return this._record.type || "alert";
  }

  message() {
    return this._record.message || "";
  }

  defaultValue() {
    return this._record.defaultValue || "";
  }

  page() {
    return this._page;
  }

  async accept(prompt) {
    return engineCall("page.setDialogPolicy", {
      page: this._page._id,
      action: "accept",
      prompt: prompt ?? null,
    });
  }

  async dismiss() {
    return engineCall("page.setDialogPolicy", {
      page: this._page._id,
      action: "dismiss",
    });
  }
}

class Locator {
  constructor(page, selector) {
    this._page = page;
    this._selector = selector;
    this._description = undefined;
    return withUnsupported(this, "Locator");
  }

  async click(options) {
    if (options != null) {
      return unsupported("Locator.click.options")();
    }
    if (this._selector && this._selector.type === "css" && this._selector.value) {
      this._page._lastFileSelector = this._selector.value;
    }
    await engineCall("locator.click", {
      ...locatorParams(this),
    });
    await this._page._flushPopups();
  }

  async tap(options) {
    if (options != null) {
      return unsupported("Locator.tap.options")();
    }
    await engineCall("locator.tap", {
      ...locatorParams(this),
    });
    await this._page._flushPopups();
  }

  async fill(value, options) {
    if (options != null) {
      return unsupported("Locator.fill.options")();
    }
    await engineCall("locator.fill", {
      ...locatorParams(this),
      value: String(value),
      editable: true,
    });
  }

  async hover() {
    await engineCall("locator.hover", {
      ...locatorParams(this),
    });
  }

  async innerText() {
    const result = await engineCall("locator.innerText", {
      ...locatorParams(this),
    });
    return result.text;
  }

  async textContent() {
    return this.innerText();
  }

  async count() {
    const result = await engineCall("locator.count", {
      ...locatorParams(this),
    });
    return result.count;
  }

  async isVisible() {
    const result = await engineCall("locator.isVisible", {
      ...locatorParams(this),
    });
    return !!result.visible;
  }

  async waitFor(options) {
    if (options != null) {
      const keys = Object.keys(options).filter((key) => options[key] !== undefined);
      if (keys.some((key) => key !== "state" && key !== "timeout")) {
        return unsupported("Locator.waitFor.options")();
      }
      if (
        options.state != null &&
        options.state !== "visible" &&
        options.state !== "attached"
      ) {
        return unsupported("Locator.waitFor.state")();
      }
    }
    await engineCall("locator.waitFor", locatorParams(this, {
      timeout: (options && options.timeout) || this._page._timeout || 30_000,
    }));
  }

  async waitForFunction(pageFunction, arg, options) {
    if (options != null) {
      return unsupported("Locator.waitForFunction.options")();
    }
    const deadline = Date.now() + (this._page._timeout || 30_000);
    while (Date.now() < deadline) {
      const value = await this.evaluate(pageFunction, arg);
      if (value) {
        return value;
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: Locator.waitForFunction");
  }

  async check() {
    await engineCall("locator.check", {
      ...locatorParams(this),
    });
  }

  async uncheck() {
    await engineCall("locator.uncheck", {
      ...locatorParams(this),
    });
  }

  async selectOption(value) {
    await engineCall("locator.selectOption", {
      ...locatorParams(this),
      value: Array.isArray(value) ? value[0] : value,
    });
  }

  async inputValue() {
    const result = await engineCall("locator.inputValue", {
      ...locatorParams(this),
    });
    return result.value;
  }

  async getAttribute(name) {
    const result = await engineCall("locator.getAttribute", {
      ...locatorParams(this),
      name: String(name),
    });
    return result.value;
  }

  async isChecked() {
    const result = await engineCall("locator.isChecked", {
      ...locatorParams(this),
    });
    return !!result.checked;
  }

  async isEnabled() {
    const result = await engineCall("locator.isEnabled", {
      ...locatorParams(this),
    });
    return !!result.enabled;
  }

  async isDisabled() {
    const result = await engineCall("locator.isDisabled", {
      ...locatorParams(this),
    });
    return !!result.disabled;
  }

  async isHidden() {
    const result = await engineCall("locator.isHidden", {
      ...locatorParams(this),
    });
    return !!result.hidden;
  }

  async innerHTML() {
    const result = await engineCall("locator.innerHTML", {
      ...locatorParams(this),
    });
    return result.html;
  }

  async focus() {
    await engineCall("locator.focus", {
      ...locatorParams(this),
    });
  }

  async blur() {
    await engineCall("locator.blur", {
      ...locatorParams(this),
    });
  }

  async boundingBox() {
    return engineCall("locator.boundingBox", {
      ...locatorParams(this),
    });
  }

  async screenshot() {
    const result = await engineCall("locator.screenshot", {
      ...locatorParams(this),
    });
    return decodeBase64(result.png_base64 || "").buffer;
  }

  async allTextContents() {
    const result = await engineCall("locator.allTextContents", {
      ...locatorParams(this),
    });
    return result.values || [];
  }

  async allInnerTexts() {
    return this.allTextContents();
  }

  async evaluate(pageFunction, arg) {
    const fn =
      typeof pageFunction === "function" ? pageFunction.toString() : String(pageFunction);
    const source =
      arg === undefined
        ? fn
        : "function(el) { return (" + fn + ")(el, " + JSON.stringify(arg) + "); }";
    const result = await engineCall("locator.evaluate", {
      ...locatorParams(this),
      source,
    });
    return result.value;
  }

  async evaluateAll(pageFunction, arg) {
    const fn =
      typeof pageFunction === "function" ? pageFunction.toString() : String(pageFunction);
    const source =
      arg === undefined
        ? fn
        : "function(els) { return (" + fn + ")(els, " + JSON.stringify(arg) + "); }";
    const result = await engineCall("locator.evaluateAll", {
      ...locatorParams(this),
      source,
    });
    return result.value;
  }

  async dblclick(options) {
    if (options != null) {
      return unsupported("Locator.dblclick.options")();
    }
    await engineCall("locator.dblclick", {
      ...locatorParams(this),
    });
  }

  async dispatchEvent(type) {
    await engineCall("locator.dispatchEvent", {
      ...locatorParams(this),
      event: String(type),
    });
  }

  async clear() {
    await this.fill("");
  }

  async isEditable() {
    const result = await engineCall("locator.isEditable", {
      ...locatorParams(this),
    });
    return !!result.editable;
  }

  first() {
    return new Locator(this._page, { ...this._selector, nth: 0 });
  }

  nth(index) {
    return new Locator(this._page, { ...this._selector, nth: index });
  }

  last() {
    return new Locator(this._page, { ...this._selector, nth: -1 });
  }

  locator(selector) {
    return new Locator(this._page, { type: "css", value: selector, scope: this._selector });
  }

  getByRole(role, options = {}) {
    refuseLocatorOptions("Locator.getByRole", options, ["name", "exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByRole.exact");
    }
    return new Locator(this._page, {
      type: "role",
      role,
      name: options.name ?? null,
      scope: this._selector,
    });
  }

  getByText(text, options) {
    refuseLocatorOptions("Locator.getByText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByText.exact");
    }
    return new Locator(this._page, { type: "text", value: String(text), scope: this._selector });
  }

  getByLabel(name, options) {
    refuseLocatorOptions("Locator.getByLabel", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByLabel.exact");
    }
    return new Locator(this._page, { type: "label", name, scope: this._selector });
  }

  getByPlaceholder(name, options) {
    refuseLocatorOptions("Locator.getByPlaceholder", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByPlaceholder.exact");
    }
    return new Locator(this._page, { type: "placeholder", name: String(name), scope: this._selector });
  }

  getByAltText(name, options) {
    refuseLocatorOptions("Locator.getByAltText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByAltText.exact");
    }
    return new Locator(this._page, { type: "alt", name: String(name), scope: this._selector });
  }

  getByTitle(name, options) {
    refuseLocatorOptions("Locator.getByTitle", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Locator.getByTitle.exact");
    }
    return new Locator(this._page, { type: "title", name: String(name), scope: this._selector });
  }

  getByTestId(name, options) {
    refuseLocatorOptions("Locator.getByTestId", options, []);
    return new Locator(this._page, {
      type: "testid",
      name: String(name),
      attr: testIdAttribute,
      scope: this._selector,
    });
  }

  filter(options = {}) {
    refuseLocatorOptions("Locator.filter", options, ["has", "hasNot", "hasText"]);
    if (options.hasText != null && typeof options.hasText !== "string") {
      throwUnsupported("Locator.filter.hasText");
    }
    const has = options.has && options.has._selector ? options.has._selector : null;
    const hasNot = options.hasNot && options.hasNot._selector ? options.hasNot._selector : null;
    if (options.has && !has) {
      throwUnsupported("Locator.filter.has");
    }
    if (options.hasNot && !hasNot) {
      throwUnsupported("Locator.filter.hasNot");
    }
    return new Locator(this._page, {
      type: "filter",
      hasText: options.hasText != null ? String(options.hasText) : null,
      has,
      hasNot,
      scope: this._selector,
    });
  }

  async scrollIntoViewIfNeeded() {
    await engineCall("locator.scrollIntoViewIfNeeded", {
      ...locatorParams(this),
    });
  }

  async selectText() {
    await engineCall("locator.selectText", {
      ...locatorParams(this),
    });
  }

  async all() {
    const n = await this.count();
    const locators = [];
    for (let i = 0; i < n; i++) {
      locators.push(this.nth(i));
    }
    return locators;
  }

  async setInputFiles(files) {
    if (this._selector.type !== "css") {
      return unsupported("Locator.setInputFiles.nonCss")();
    }
    return this._page.setInputFiles(this._selector.value, files);
  }

  async type(text) {
    await this.focus();
    await this._page.keyboard.type(text);
  }

  async press(key) {
    await this.focus();
    await this._page.keyboard.press(key);
  }

  page() {
    return this._page;
  }

  async setChecked(checked) {
    return checked ? this.check() : this.uncheck();
  }

  async pressSequentially(text) {
    return this.type(text);
  }

  describe(description) {
    this._description = String(description);
    return this;
  }

  description() {
    return this._description || null;
  }

  async ariaSnapshot(options) {
    if (options != null) {
      return unsupported("Locator.ariaSnapshot.options")();
    }
    return this.evaluate((el) => {
      function roleOf(node) {
        const explicit = node.getAttribute && node.getAttribute("role");
        if (explicit) return explicit;
        const tag = (node.tagName || "").toLowerCase();
        if (tag === "button") return "button";
        if (tag === "a" && node.hasAttribute("href")) return "link";
        if (tag === "input" || tag === "textarea") return "textbox";
        if (tag === "img") return "img";
        if (tag === "h1" || tag === "h2" || tag === "h3" || tag === "h4" || tag === "h5" || tag === "h6") {
          return "heading";
        }
        return tag || "generic";
      }
      function nameOf(node) {
        const labelled = node.getAttribute && node.getAttribute("aria-label");
        if (labelled) return labelled.trim();
        if (node.getAttribute && node.getAttribute("alt")) return node.getAttribute("alt").trim();
        return ((node.innerText || node.textContent || node.value || "") + "").trim().split("\n")[0];
      }
      function walk(node, indent) {
        if (!node || node.nodeType !== 1) return "";
        const role = roleOf(node);
        const name = nameOf(node);
        let line = indent + "- " + role;
        if (name) line += ' "' + name.replace(/"/g, '\\"') + '"';
        const kids = [];
        const children = node.children || [];
        for (let i = 0; i < children.length; i++) {
          const child = walk(children[i], indent + "  ");
          if (child) kids.push(child);
        }
        return [line].concat(kids).join("\n");
      }
      return walk(el, "");
    });
  }

  toString() {
    return this._description || JSON.stringify(this._selector);
  }

  contentFrame() {
    if (!this._selector || this._selector.type !== "css" || !this._selector.value) {
      throwUnsupported("Locator.contentFrame.nonCss");
    }
    const index = this._selector.nth == null ? null : this._selector.nth;
    return new FrameLocator(this._page, this._selector.value, index);
  }

  frameLocator(selector) {
    if (!this._selector || this._selector.type !== "css" || !this._selector.value) {
      throwUnsupported("Locator.frameLocator.nonCss");
    }
    const index = this._selector.nth == null ? null : this._selector.nth;
    return new FrameLocator(this._page, this._selector.value + " " + selector, index);
  }

  async dragTo(target) {
    const from = await this.boundingBox();
    const to = await target.boundingBox();
    if (!from || !to) {
      throw new Error("dragTo requires visible bounding boxes");
    }
    await this._page.mouse.move(from.x + from.width / 2, from.y + from.height / 2);
    await this._page.mouse.down();
    await this._page.mouse.move(to.x + to.width / 2, to.y + to.height / 2);
    await this._page.mouse.up();
  }
}

class FrameLocator {
  constructor(page, frameSelector, index = null) {
    this._page = page;
    this._frame = frameSelector;
    this._index = index;
    return withUnsupported(this, "FrameLocator");
  }

  _frameScope() {
    return {
      type: "framecss",
      frame: this._frame,
      value: "html",
      frameIndex: this._index,
    };
  }

  _inner(type, extra) {
    return new Locator(this._page, {
      type,
      frame: this._frame,
      frameIndex: this._index,
      ...extra,
    });
  }

  getByLabel(name, options) {
    refuseLocatorOptions("FrameLocator.getByLabel", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByLabel.exact");
    }
    return new Locator(this._page, {
      type: "label",
      name,
      scope: this._frameScope(),
    });
  }

  getByPlaceholder(name, options) {
    refuseLocatorOptions("FrameLocator.getByPlaceholder", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByPlaceholder.exact");
    }
    return new Locator(this._page, {
      type: "placeholder",
      name: String(name),
      scope: this._frameScope(),
    });
  }

  getByAltText(name, options) {
    refuseLocatorOptions("FrameLocator.getByAltText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByAltText.exact");
    }
    return new Locator(this._page, {
      type: "alt",
      name: String(name),
      scope: this._frameScope(),
    });
  }

  getByTitle(name, options) {
    refuseLocatorOptions("FrameLocator.getByTitle", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByTitle.exact");
    }
    return new Locator(this._page, {
      type: "title",
      name: String(name),
      scope: this._frameScope(),
    });
  }

  getByTestId(name) {
    return new Locator(this._page, {
      type: "testid",
      name: String(name),
      attr: testIdAttribute,
      scope: this._frameScope(),
    });
  }

  locator(selector) {
    return this._inner("framecss", { value: selector });
  }

  getByText(text, options) {
    refuseLocatorOptions("FrameLocator.getByText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByText.exact");
    }
    return this._inner("frametext", { value: String(text) });
  }

  getByRole(role, options = {}) {
    refuseLocatorOptions("FrameLocator.getByRole", options, ["name", "exact"]);
    if (options && options.exact === false) {
      throwUnsupported("FrameLocator.getByRole.exact");
    }
    return this._inner("framerole", { role, name: options.name ?? null });
  }

  nth(index) {
    return new FrameLocator(this._page, this._frame, index);
  }

  first() {
    return this.nth(0);
  }

  last() {
    return this.nth(-1);
  }

  owner() {
    const loc = new Locator(this._page, { type: "css", value: this._frame });
    return this._index == null ? loc : loc.nth(this._index);
  }

  frameLocator() {
    throwUnsupported("FrameLocator.frameLocator.nested");
  }
}

class Frame {
  constructor(page, info) {
    this._page = page;
    this._id = info.id;
    this._name = info.name || "";
    this._url = info.url || "";
    return withUnsupported(this, "Frame");
  }

  async evaluate(pageFunction, arg) {
    if (this._id === "main") {
      return this._page.evaluate(pageFunction, arg);
    }
    return engineCall("page.frameEvaluate", {
      page: this._page._id,
      index: Number(this._id),
      source: serializeEvaluate(pageFunction, arg),
    }).then((result) => result.value);
  }

  url() {
    return this._url;
  }

  name() {
    return this._name;
  }

  async content() {
    return this.evaluate(() => document.documentElement.outerHTML);
  }

  _isMain() {
    return this._id === "main";
  }

  _childFrameLocator() {
    return this._page.frameLocator("iframe:nth-of-type(" + (Number(this._id) + 1) + ")");
  }

  locator(selector) {
    if (this._isMain()) {
      return this._page.locator(selector);
    }
    return this._childFrameLocator().locator(selector);
  }

  getByRole(role, options) {
    if (this._isMain()) {
      return this._page.getByRole(role, options);
    }
    return this._childFrameLocator().getByRole(role, options);
  }

  getByText(text) {
    if (this._isMain()) {
      return this._page.getByText(text);
    }
    return this._childFrameLocator().getByText(text);
  }

  getByLabel(name) {
    if (this._isMain()) {
      return this._page.getByLabel(name);
    }
    throwUnsupported("Frame.getByLabel.child");
  }

  click(selector, options) {
    return this.locator(selector).click(options);
  }

  tap(selector, options) {
    return this.locator(selector).tap(options);
  }

  fill(selector, value, options) {
    return this.locator(selector).fill(value, options);
  }

  async type(selector, text) {
    return this.locator(selector).type(text);
  }

  async innerText(selector) {
    return this.locator(selector).innerText();
  }

  async goto(url, options) {
    if (this._id !== "main") {
      if (options != null) {
        return unsupported("Frame.goto.options")();
      }
      const result = await engineCall("page.frameGoto", {
        page: this._page._id,
        index: Number(this._id),
        url: String(url),
      });
      this._url = result.url || String(url);
      await this._page._dispatchFrames();
      return result;
    }
    return this._page.goto(url, options);
  }

  async setContent(html, options) {
    if (options != null) {
      return unsupported("Frame.setContent.options")();
    }
    if (this._id !== "main") {
      await this.evaluate((markup) => {
        document.open();
        document.write(String(markup));
        document.close();
        return true;
      }, html);
      await this._page._dispatchFrames();
      return;
    }
    return this._page.setContent(html);
  }

  async waitForSelector(selector) {
    await this.locator(selector).waitFor();
  }

  async waitForLoadState(state) {
    if (state != null && state !== "load" && state !== "domcontentloaded") {
      return unsupported("Frame.waitForLoadState.state")();
    }
    if (this._id !== "main") {
      const wantComplete = state == null || state === "load";
      const deadline = Date.now() + (this._page._timeout || 30_000);
      while (Date.now() < deadline) {
        const ready = await this.evaluate(() => document.readyState);
        if (ready === "complete" || (!wantComplete && ready === "interactive")) {
          return;
        }
        ops.op_sleep_ms(20);
      }
      throw new TimeoutError("timeout: Frame.waitForLoadState");
    }
    return this._page.waitForLoadState(state);
  }

  async addScriptTag(options) {
    if (options && (options.url || options.path || options.type)) {
      return unsupported("Frame.addScriptTag.url")();
    }
    if (this._id !== "main") {
      await this.evaluate((source) => {
        const script = document.createElement("script");
        script.textContent = String(source || "");
        document.documentElement.appendChild(script);
        return true;
      }, (options && options.content) || "");
      return;
    }
    return this._page.addScriptTag(options || {});
  }

  async addStyleTag(options) {
    if (options && (options.url || options.path)) {
      return unsupported("Frame.addStyleTag.url")();
    }
    if (this._id !== "main") {
      await this.evaluate((css) => {
        const style = document.createElement("style");
        style.textContent = String(css || "");
        document.documentElement.appendChild(style);
        return true;
      }, (options && options.content) || "");
      return;
    }
    return this._page.addStyleTag(options || {});
  }

  hover(selector) {
    return this.locator(selector).hover();
  }

  check(selector) {
    return this.locator(selector).check();
  }

  isVisible(selector) {
    return this.locator(selector).isVisible();
  }

  innerHTML(selector) {
    return this.locator(selector).innerHTML();
  }

  inputValue(selector) {
    return this.locator(selector).inputValue();
  }

  isEditable(selector) {
    return this.locator(selector).isEditable();
  }

  page() {
    return this._page;
  }

  isDetached() {
    if (this._isMain()) {
      return false;
    }
    throwUnsupported("Frame.isDetached.child");
  }

  getByPlaceholder(name) {
    if (this._isMain()) {
      return this._page.getByPlaceholder(name);
    }
    throwUnsupported("Frame.getByPlaceholder.child");
  }

  getByAltText(name) {
    if (this._isMain()) {
      return this._page.getByAltText(name);
    }
    throwUnsupported("Frame.getByAltText.child");
  }

  getByTitle(name) {
    if (this._isMain()) {
      return this._page.getByTitle(name);
    }
    throwUnsupported("Frame.getByTitle.child");
  }

  getByTestId(name) {
    if (this._isMain()) {
      return this._page.getByTestId(name);
    }
    throwUnsupported("Frame.getByTestId.child");
  }

  getAttribute(selector, name) {
    return this.locator(selector).getAttribute(name);
  }

  textContent(selector) {
    return this.locator(selector).textContent();
  }

  title() {
    return this._id === "main" ? this._page.title() : this.evaluate(() => document.title);
  }

  isChecked(selector) {
    return this.locator(selector).isChecked();
  }

  isEnabled(selector) {
    return this.locator(selector).isEnabled();
  }

  isDisabled(selector) {
    return this.locator(selector).isDisabled();
  }

  isHidden(selector) {
    return this.locator(selector).isHidden();
  }

  press(selector, key) {
    return this.locator(selector).press(key);
  }

  selectOption(selector, value) {
    return this.locator(selector).selectOption(value);
  }

  uncheck(selector) {
    return this.locator(selector).uncheck();
  }

  setChecked(selector, checked) {
    return checked ? this.check(selector) : this.uncheck(selector);
  }

  dblclick(selector, options) {
    return this.locator(selector).dblclick(options);
  }

  dispatchEvent(selector, type) {
    return this.locator(selector).dispatchEvent(type);
  }

  focus(selector) {
    return this.locator(selector).focus();
  }

  waitForTimeout(ms) {
    return this._page.waitForTimeout(ms);
  }

  async waitForFunction(pageFunction, arg, options) {
    if (options != null) {
      return unsupported("Frame.waitForFunction.options")();
    }
    const deadline = Date.now() + (this._page._timeout || 30_000);
    while (Date.now() < deadline) {
      const value = await this.evaluate(pageFunction, arg);
      if (value) {
        return value;
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: Frame.waitForFunction");
  }

  async waitForURL(pattern, options) {
    if (options != null) {
      return unsupported("Frame.waitForURL.options")();
    }
    if (pattern instanceof RegExp || typeof pattern === "function") {
      return unsupported("Frame.waitForURL.pattern")();
    }
    const needle = String(pattern);
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      const url = this._isMain()
        ? await this._page.url()
        : await this.evaluate(() => String(location.href));
      if (String(url).includes(needle)) {
        return url;
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: Frame.waitForURL " + needle);
  }

  waitForNavigation() {
    if (!this._isMain()) {
      throwUnsupported("Frame.waitForNavigation.child");
    }
    return this._page.waitForNavigation();
  }

  frameLocator(selector) {
    if (this._isMain()) {
      return this._page.frameLocator(selector);
    }
    throwUnsupported("Frame.frameLocator.child");
  }

  parentFrame() {
    if (this._isMain()) {
      return null;
    }
    return this._page.mainFrame();
  }

  async childFrames() {
    if (!this._isMain()) {
      throwUnsupported("Frame.childFrames.nested");
    }
    const frames = await this._page.frames();
    return frames.filter((frame) => !frame._isMain());
  }
}

class Page {
  constructor(id) {
    this._id = id;
    this._closed = false;
    this._timeout = 30_000;
    this._context = null;
    this._handlers = {};
    this._seenFrames = {};
    this._emittedNetwork = new Set();
    this._consoleSeen = 0;
    this._dialogSeen = 0;
    this._popupWaiters = [];
    this._pendingPopups = [];
    this._openerId = null;
    this._navWaiters = [];
    this._consoleWaiters = [];
    this._pendingConsole = [];
    this._dialogWaiters = [];
    this._pendingDialogs = [];
    this._mouseX = 0;
    this._mouseY = 0;
    this.coverage = {
      startJSCoverage: unsupported("Coverage.startJSCoverage"),
      stopJSCoverage: unsupported("Coverage.stopJSCoverage"),
      startCSSCoverage: unsupported("Coverage.startCSSCoverage"),
      stopCSSCoverage: unsupported("Coverage.stopCSSCoverage"),
    };
    this.request = withUnsupported({}, "APIRequestContext");
    this.touchscreen = {
      tap: async (x, y) => {
        await engineCall("page.touch.tap", { page: this._id, x, y });
      },
    };
    this.mouse = {
      click: async (x, y) => {
        this._mouseX = Number(x) || 0;
        this._mouseY = Number(y) || 0;
        await engineCall("page.mouse.click", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
      },
      move: async (x, y) => {
        this._mouseX = Number(x) || 0;
        this._mouseY = Number(y) || 0;
        await engineCall("page.mouse.move", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
      },
      down: async () => {
        await engineCall("page.mouse.down", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
      },
      up: async () => {
        await engineCall("page.mouse.up", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
      },
      wheel: async (deltaX, deltaY) => {
        await engineCall("page.mouse.wheel", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
          deltaX: Number(deltaX) || 0,
          deltaY: Number(deltaY) || 0,
        });
      },
      dblclick: async (x, y) => {
        this._mouseX = Number(x) || 0;
        this._mouseY = Number(y) || 0;
        await engineCall("page.mouse.click", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
        await engineCall("page.mouse.click", {
          page: this._id,
          x: this._mouseX,
          y: this._mouseY,
        });
      },
    };
    return withUnsupported(this, "Page");
  }

  context() {
    return this._context;
  }

  setDefaultTimeout(ms) {
    this._timeout = Math.max(0, Number(ms) || 0);
  }

  setDefaultNavigationTimeout(ms) {
    this.setDefaultTimeout(ms);
  }

  getByRole(role, options = {}) {
    refuseLocatorOptions("Page.getByRole", options, ["name", "exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByRole.exact");
    }
    return new Locator(this, {
      type: "role",
      role,
      name: options.name ?? null,
    });
  }

  getByLabel(name, options) {
    refuseLocatorOptions("Page.getByLabel", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByLabel.exact");
    }
    return new Locator(this, { type: "label", name });
  }

  getByText(text, options) {
    refuseLocatorOptions("Page.getByText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByText.exact");
    }
    return new Locator(this, { type: "text", value: String(text) });
  }

  getByPlaceholder(name, options) {
    refuseLocatorOptions("Page.getByPlaceholder", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByPlaceholder.exact");
    }
    return new Locator(this, { type: "placeholder", name: String(name) });
  }

  getByAltText(name, options) {
    refuseLocatorOptions("Page.getByAltText", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByAltText.exact");
    }
    return new Locator(this, { type: "alt", name: String(name) });
  }

  getByTitle(name, options) {
    refuseLocatorOptions("Page.getByTitle", options, ["exact"]);
    if (options && options.exact === false) {
      throwUnsupported("Page.getByTitle.exact");
    }
    return new Locator(this, { type: "title", name: String(name) });
  }

  getByTestId(name, options) {
    refuseLocatorOptions("Page.getByTestId", options, []);
    return new Locator(this, { type: "testid", name: String(name), attr: testIdAttribute });
  }

  locator(selector) {
    return new Locator(this, { type: "css", value: selector });
  }

  frameLocator(selector) {
    return new FrameLocator(this, selector);
  }

  click(selector, options) {
    return this.locator(selector).click(options);
  }

  fill(selector, value, options) {
    return this.locator(selector).fill(value, options);
  }

  hover(selector) {
    return this.locator(selector).hover();
  }

  check(selector) {
    return this.locator(selector).check();
  }

  uncheck(selector) {
    return this.locator(selector).uncheck();
  }

  async setChecked(selector, checked) {
    if (checked) {
      return this.locator(selector).check();
    }
    return this.locator(selector).uncheck();
  }

  selectOption(selector, value) {
    return this.locator(selector).selectOption(value);
  }

  innerText(selector) {
    return this.locator(selector).innerText();
  }

  innerHTML(selector) {
    return this.locator(selector).innerHTML();
  }

  textContent(selector) {
    return this.locator(selector).textContent();
  }

  inputValue(selector) {
    return this.locator(selector).inputValue();
  }

  getAttribute(selector, name) {
    return this.locator(selector).getAttribute(name);
  }

  isVisible(selector) {
    return this.locator(selector).isVisible();
  }

  isHidden(selector) {
    return this.locator(selector).isHidden();
  }

  isChecked(selector) {
    return this.locator(selector).isChecked();
  }

  isEnabled(selector) {
    return this.locator(selector).isEnabled();
  }

  isDisabled(selector) {
    return this.locator(selector).isDisabled();
  }

  focus(selector) {
    return this.locator(selector).focus();
  }

  async type(selector, text) {
    await this.focus(selector);
    await this.keyboard.type(text);
  }

  async press(selector, key) {
    await this.focus(selector);
    await this.keyboard.press(key);
  }

  async tap(selector, options) {
    return this.locator(selector).tap(options);
  }

  dblclick(selector, options) {
    return this.locator(selector).dblclick(options);
  }

  dispatchEvent(selector, type) {
    return this.locator(selector).dispatchEvent(type);
  }

  clear(selector) {
    return this.locator(selector).clear();
  }

  isEditable(selector) {
    return this.locator(selector).isEditable();
  }

  async bringToFront() {
    return unsupported("Page.bringToFront")();
  }

  async addScriptTag(options = {}) {
    await engineCall("page.addScriptTag", {
      page: this._id,
      content: options.content || "",
      url: options.url || "",
    });
  }

  async addStyleTag(options = {}) {
    await engineCall("page.addStyleTag", {
      page: this._id,
      content: options.content || "",
    });
  }

  async waitForFunction(pageFunction, arg) {
    const deadline = Date.now() + (this._timeout || 30_000);
    while (Date.now() < deadline) {
      const value = await this.evaluate(pageFunction, arg);
      if (value) {
        return value;
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: waitForFunction");
  }

  async waitForURL(pattern) {
    const needle = String(pattern);
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      const url = await this.url();
      if (String(url).includes(needle)) {
        return url;
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: waitForURL " + needle);
  }

  async waitForRequest(pattern) {
    const needle = String(pattern);
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      const result = await engineCall("page.requests", { page: this._id });
      const records = result.requests || [];
      const hit = records.find((rec) => String(rec.url).includes(needle));
      if (hit) {
        return this._requestFromRecord(hit, records);
      }
      ops.op_sleep_ms(20);
    }
    throw new TimeoutError("timeout: waitForRequest " + needle);
  }

  async waitForResponse(pattern) {
    const needle = String(pattern);
    const deadline = Date.now() + (this._timeout || 30_000);
    while (Date.now() < deadline) {
      const result = await engineCall("page.responses", { page: this._id });
      const hit = (result.responses || []).find((rec) => String(rec.url).includes(needle));
      if (hit) {
        return this._responseFromRecord(hit);
      }
      const request = (await engineCall("page.requests", { page: this._id })).requests || [];
      const reqHit = request.find((rec) => String(rec.url).includes(needle));
      if (reqHit) {
        // Non-intercepted loads have no recorded Servo response body/status.
        break;
      }
      ops.op_sleep_ms(20);
    }
    return unsupported("Page.waitForResponse.unintercepted")();
  }

  async goto(url, options) {
    if (options != null) {
      return unsupported("Page.goto.options")();
    }
    try {
      const result = await engineCall("page.goto", { page: this._id, url });
      await this._flushPopups();
      await this._flushNavigation();
      await this._dispatchNetworkUntilSettled();
      await this._dispatchFrames();
      this._emitLoad();
      return result;
    } catch (error) {
      await this._dispatchNetworkUntilSettled();
      throw error;
    }
  }

  _requestFromRecord(rec, all) {
    const headerList = rec.headers || [];
    const records = all || [];
    const index = records.indexOf(rec);
    const headerMap = () => {
      const out = {};
      headerList.forEach((h) => {
        out[String(h.name).toLowerCase()] = h.value;
      });
      return out;
    };
    return withUnsupported({
      url: () => rec.url,
      method: () => rec.method || "GET",
      headers: headerMap,
      headerValue: (name) => headerMap()[String(name).toLowerCase()] || null,
      headersArray: () =>
        headerList.map((h) => ({ name: String(h.name), value: h.value })),
      allHeaders: async () => headerMap(),
      resourceType: () => (rec.main_frame ? "document" : "other"),
      isNavigationRequest: () => !!rec.main_frame,
      failure: () => rec.failure || null,
      redirectedFrom: () => throwUnsupported("Request.redirectedFrom"),
      redirectedTo: () => throwUnsupported("Request.redirectedTo"),
      postData: () => {
        const method = rec.method || "GET";
        if (method === "GET" || method === "HEAD") return null;
        throwUnsupported("Request.postData");
      },
      postDataJSON: () => {
        const method = rec.method || "GET";
        if (method === "GET" || method === "HEAD") return null;
        throwUnsupported("Request.postDataJSON");
      },
      postDataBuffer: () => {
        const method = rec.method || "GET";
        if (method === "GET" || method === "HEAD") return null;
        throwUnsupported("Request.postDataBuffer");
      },
      timing: () => throwUnsupported("Request.timing"),
      sizes: () => throwUnsupported("Request.sizes"),
      frame: () => this.mainFrame(),
      response: async () => {
        const result = await engineCall("page.responses", { page: this._id });
        const hit = (result.responses || []).find((row) => row.url === rec.url);
        if (!hit) return null;
        return this._responseFromRecord(hit);
      },
    }, "Request");
  }

  _responseFromRecord(rec, request) {
    const headers = rec.headers || {};
    const status = Number(rec.status) || 200;
    const bytes = () => decodeBase64(rec.bodyBase64 || "");
    return withUnsupported({
      url: () => rec.url,
      status: () => status,
      statusText: () => rec.statusText || (status < 400 ? "OK" : ""),
      ok: () => (rec.ok == null ? status < 400 : !!rec.ok),
      headers: () => headers,
      headerValue: (name) => headers[String(name).toLowerCase()] || null,
      headersArray: () =>
        Object.keys(headers).map((name) => ({ name, value: headers[name] })),
      allHeaders: async () => headers,
      headerValues: async (name) => {
        const value = headers[String(name).toLowerCase()];
        return value == null ? [] : [value];
      },
      body: async () => bytes(),
      text: async () => decodeUtf8(bytes()),
      json: async () => JSON.parse(decodeUtf8(bytes())),
      request: () =>
        request || this._requestFromRecord({ url: rec.url, method: "GET" }),
    }, "Response");
  }

  _downloadFromRecord(rec) {
    const page = this;
    return withUnsupported({
      url: () => rec.url,
      suggestedFilename: () => rec.suggestedFilename || "download",
      page: () => page,
      failure: async () => null,
      path: async () => rec.path || null,
      cancel: async () => unsupported("Download.cancel")(),
      saveAs: async (path) => {
        const result = await engineCall("page.saveDownload", {
          page: page._id,
          url: rec.url,
          path: String(path),
        });
        if (Number(result.bytes) !== Number(rec.byteLength || 0)) {
          throw new Error(
            "saveAs wrote " + result.bytes + " bytes, expected " + rec.byteLength,
          );
        }
        const expectedHex = hexEncode(decodeBase64(rec.bodyBase64 || ""));
        if (String(result.hex) !== expectedHex) {
          throw new Error("saveAs hex " + result.hex + " expected " + expectedHex);
        }
        rec.path = String(path);
      },
    }, "Download");
  }

  async _dispatchNetwork(settle) {
    const result = await engineCall("page.requests", { page: this._id });
    const responses = ((await engineCall("page.responses", { page: this._id })).responses || []);
    const requests = result.requests || [];
    this._emittedNetwork = this._emittedNetwork || new Set();
    for (let index = 0; index < requests.length; index++) {
      const rec = requests[index];
      const key = String(rec.method || "GET") + " " + String(rec.url) + " " + index;
      const request = this._requestFromRecord(rec, requests);
      const hit = responses.find((row) => row.url === rec.url);
      if (!this._emittedNetwork.has(key + " req")) {
        this._emittedNetwork.add(key + " req");
        this._emit("request", request);
      }
      if (hit) {
        if (!this._emittedNetwork.has(key + " fin")) {
          this._emittedNetwork.add(key + " fin");
          this._emit("response", this._responseFromRecord(hit, request));
          this._emit("requestfinished", request);
        }
      } else if (request.failure()) {
        if (!this._emittedNetwork.has(key + " fail")) {
          this._emittedNetwork.add(key + " fail");
          this._emit("requestfailed", request);
        }
      } else if (settle) {
        if (!this._emittedNetwork.has(key + " fin")) {
          this._emittedNetwork.add(key + " fin");
          this._emit("requestfinished", request);
        }
      }
    }
    return requests.length;
  }

  async _dispatchNetworkUntilSettled() {
    const deadline = Date.now() + 2000;
    while (Date.now() < deadline) {
      const count = await this._dispatchNetwork(false);
      if (count > 0) {
        await this._dispatchNetwork(true);
        return;
      }
      ops.op_sleep_ms(20);
    }
    await this._dispatchNetwork(true);
  }

  async _dispatchFrames() {
    const frames = await this.frames();
    this._seenFrames = this._seenFrames || {};
    const current = {};
    for (const frame of frames) {
      const id = String(frame._id);
      const url = String(frame.url() || "");
      const prev = this._seenFrames[id];
      current[id] = { name: frame.name(), url, frame };
      if (!prev) {
        this._emit("frameattached", frame);
        this._emit("framenavigated", frame);
      } else if (prev.url !== url) {
        this._emit("framenavigated", frame);
      }
    }
    for (const id of Object.keys(this._seenFrames)) {
      if (id !== "main" && !current[id]) {
        this._emit("framedetached", this._seenFrames[id].frame);
      }
    }
    this._seenFrames = current;
  }

  _emit(event, payload) {
    const list = (this._handlers && this._handlers[event]) || [];
    for (const handler of list) {
      const result = handler(payload);
      if (result && typeof result.then === "function") {
        result.catch(() => {});
      }
    }
    if (
      this._context &&
      (event === "console" ||
        event === "dialog" ||
        event === "download" ||
        event === "request" ||
        event === "response" ||
        event === "requestfailed" ||
        event === "requestfinished" ||
        event === "frameattached" ||
        event === "framedetached" ||
        event === "framenavigated")
    ) {
      this._context._emit(event, payload);
    }
  }

  async evaluate(pageFunction, arg) {
    const result = await engineCall("page.evaluate", {
      page: this._id,
      source: serializeEvaluate(pageFunction, arg),
    });
    await this._flushPopups();
    await this._dispatchConsole();
    await this._dispatchDialogs();
    await this._dispatchFrames();
    return result.value;
  }

  async _dispatchConsole() {
    this._consoleWaiters = this._consoleWaiters || [];
    this._pendingConsole = this._pendingConsole || [];
    const result = await engineCall("page.consoleMessages", { page: this._id });
    const messages = result.messages || [];
    for (let i = this._consoleSeen; i < messages.length; i++) {
      const rec = messages[i];
      const payload = withUnsupported({
        type: () => rec.type || "log",
        text: () => rec.text || "",
        page: () => this,
      }, "ConsoleMessage");
      const waiter = this._consoleWaiters.shift();
      if (waiter) {
        waiter(payload);
      } else {
        this._pendingConsole.push(payload);
      }
      this._emit("console", payload);
      if ((rec.type || "log") === "error") {
        const error = new Error(rec.text || "");
        this._emit("pageerror", error);
      }
    }
    this._consoleSeen = messages.length;
  }

  _waitForConsole() {
    this._pendingConsole = this._pendingConsole || [];
    this._consoleWaiters = this._consoleWaiters || [];
    if (this._pendingConsole.length) {
      return Promise.resolve(this._pendingConsole.shift());
    }
    return new Promise((resolve) => {
      this._consoleWaiters.push(resolve);
    });
  }

  async _dispatchDialogs() {
    this._dialogWaiters = this._dialogWaiters || [];
    this._pendingDialogs = this._pendingDialogs || [];
    const result = await engineCall("page.dialogs", { page: this._id });
    const dialogs = result.dialogs || [];
    for (let i = this._dialogSeen; i < dialogs.length; i++) {
      const dialog = new Dialog(this, dialogs[i]);
      const waiter = this._dialogWaiters.shift();
      if (waiter) {
        waiter(dialog);
      } else {
        this._pendingDialogs.push(dialog);
      }
      this._emit("dialog", dialog);
    }
    this._dialogSeen = dialogs.length;
  }

  _waitForDialog() {
    this._pendingDialogs = this._pendingDialogs || [];
    this._dialogWaiters = this._dialogWaiters || [];
    if (this._pendingDialogs.length) {
      return Promise.resolve(this._pendingDialogs.shift());
    }
    return new Promise((resolve) => {
      this._dialogWaiters.push(resolve);
    });
  }

  _adoptPopup(rec) {
    const id = typeof rec === "string" ? rec : rec.page;
    const openerId = typeof rec === "string" ? this._id : rec.opener || this._id;
    let page = null;
    if (this._context && this._context._pages) {
      page = this._context._pages.find((item) => item._id === id) || null;
    }
    if (!page) {
      page = new Page(id);
      page._context = this._context;
      if (this._context) {
        this._context._pages = this._context._pages || [];
        this._context._pages.push(page);
      }
    }
    page._openerId = openerId;
    return page;
  }

  async _flushPopups() {
    this._pendingPopups = this._pendingPopups || [];
    this._popupWaiters = this._popupWaiters || [];
    const result = await engineCall("page.popups", { page: this._id });
    const recs = result.pages || [];
    for (const rec of recs) {
      const popup = this._adoptPopup(rec);
      const waiter = this._popupWaiters.shift();
      if (waiter) {
        waiter(popup);
      } else {
        this._pendingPopups.push(popup);
      }
      this._emit("popup", popup);
    }
  }

  _waitForPopup() {
    this._pendingPopups = this._pendingPopups || [];
    this._popupWaiters = this._popupWaiters || [];
    if (this._pendingPopups.length) {
      return Promise.resolve(this._pendingPopups.shift());
    }
    return new Promise((resolve) => {
      this._popupWaiters.push(resolve);
    });
  }

  async url() {
    const result = await engineCall("page.url", { page: this._id });
    return result.url;
  }

  async title() {
    const result = await engineCall("page.title", { page: this._id });
    return result.title;
  }

  async content() {
    const result = await engineCall("page.content", { page: this._id });
    return result.html;
  }

  async screenshot(options) {
    let clip = null;
    if (options != null) {
      const keys = Object.keys(options).filter((key) => options[key] !== undefined);
      if (keys.some((key) => key !== "clip") || !options.clip) {
        return unsupported("Page.screenshot.options")();
      }
      clip = {
        x: Number(options.clip.x) || 0,
        y: Number(options.clip.y) || 0,
        width: Number(options.clip.width) || 1,
        height: Number(options.clip.height) || 1,
      };
    }
    const result = await engineCall("page.screenshot", { page: this._id, clip });
    const binary = result.png_base64 || "";
    return decodeBase64(binary).buffer;
  }

  async setContent(html) {
    await engineCall("page.setContent", { page: this._id, html: String(html) });
    await this._dispatchFrames();
    this._emitLoad();
  }

  async reload() {
    await engineCall("page.reload", { page: this._id });
    await this._flushNavigation();
    await this._dispatchFrames();
    this._emitLoad();
  }

  _emitLoad() {
    this._emit("domcontentloaded", this);
    this._emit("load", this);
  }

  async waitForTimeout(ms) {
    ops.op_sleep_ms(Math.max(0, Number(ms) || 0));
  }

  async waitForLoadState(state) {
    if (state != null && state !== "load" && state !== "domcontentloaded") {
      return unsupported("Page.waitForLoadState.state")();
    }
    await engineCall("page.waitForLoadState", { page: this._id });
  }

  waitForNavigation() {
    this._navWaiters = this._navWaiters || [];
    return new Promise((resolve) => {
      this._navWaiters.push(resolve);
    });
  }

  async _flushNavigation() {
    const result = await engineCall("page.url", { page: this._id });
    while (this._navWaiters.length) {
      this._navWaiters.shift()({ url: () => result.url });
    }
  }

  async waitForSelector(selector) {
    await this.locator(selector).waitFor();
  }

  async frames() {
    const result = await engineCall("page.frames", { page: this._id });
    const children = (result.frames || []).map((info) => new Frame(this, info));
    const main = this.mainFrame();
    try {
      main._url = await this.url();
    } catch (_error) {}
    return [main, ...children];
  }

  async frame(options = {}) {
    const frames = await this.frames();
    if (options.name) {
      return frames.find((frame) => frame.name() === options.name) || null;
    }
    if (options.url) {
      return frames.find((frame) => String(frame.url()).includes(String(options.url))) || null;
    }
    return frames[0] || null;
  }

  mainFrame() {
    return new Frame(this, { id: "main", name: "", url: "" });
  }

  async goBack() {
    const result = await engineCall("page.goBack", { page: this._id });
    await this._flushNavigation();
    await this._dispatchNetwork();
    return result.ok ? result : null;
  }

  async goForward() {
    const result = await engineCall("page.goForward", { page: this._id });
    await this._flushNavigation();
    await this._dispatchNetwork();
    return result.ok ? result : null;
  }

  async close() {
    await engineCall("page.close", { page: this._id });
    this._closed = true;
    this._emit("close", this);
  }

  async isClosed() {
    if (this._closed) return true;
    const result = await engineCall("page.isClosed", { page: this._id });
    return !!result.closed;
  }

  async waitForEvent(event) {
    if (event === "dialog") {
      return this._waitForDialog();
    }
    if (event === "filechooser") {
      const result = await engineCall("page.fileChoosers", { page: this._id, consume: true });
      const rec = (result.choosers || [])[0];
      if (!rec) {
        return unsupported("Page.waitForEvent.filechooser.empty")();
      }
      const chooser = new FileChooser(this, rec);
      this._emit("filechooser", chooser);
      return chooser;
    }
    if (event === "popup" || event === "page") {
      return this._waitForPopup();
    }
    if (event === "console") {
      return this._waitForConsole();
    }
    if (event === "download") {
      const result = await engineCall("page.downloads", { page: this._id });
      const rec = (result.downloads || [])[0];
      if (!rec) {
        return unsupported("Page.waitForEvent.download.empty")();
      }
      const download = this._downloadFromRecord(rec);
      this._emit("download", download);
      return download;
    }
    if (event === "request") {
      return this.waitForRequest("");
    }
    if (event === "response") {
      return this.waitForResponse("");
    }
    if (event === "pageerror") {
      const errors = await this.pageErrors();
      if (errors.length) return errors[0];
      const deadline = Date.now() + (this._timeout || 30_000);
      while (Date.now() < deadline) {
        const next = await this.pageErrors();
        if (next.length) return next[0];
        ops.op_sleep_ms(20);
      }
      return unsupported("Page.waitForEvent.pageerror.empty")();
    }
    if (event === "close") {
      if (this._closed) return this;
      return new Promise((resolve) => {
        this.once("close", () => resolve(this));
      });
    }
    if (event === "load" || event === "domcontentloaded") {
      return new Promise((resolve) => {
        this.once(event, () => resolve(this));
      });
    }
    if (
      event === "frameattached" ||
      event === "framedetached" ||
      event === "framenavigated" ||
      event === "requestfailed" ||
      event === "requestfinished"
    ) {
      return new Promise((resolve) => {
        this.once(event, (payload) => resolve(payload));
      });
    }
    return unsupported(`Page.waitForEvent.${event}`)();
  }

  async setInputFiles(selector, files) {
    const list = Array.isArray(files) ? files : [files];
    return engineCall("page.setInputFiles", {
      page: this._id,
      selector: String(selector),
      files: list.map(String),
    });
  }

  on(event, handler) {
    if (event === "dialog") {
      this._dialogHandler = handler;
      const dialog = new Dialog(this, { type: "alert", message: "", defaultValue: "" });
      const result = handler(dialog);
      if (result && typeof result.then === "function") {
        result.catch(() => {});
      }
      this._handlers[event] = this._handlers[event] || [];
      this._handlers[event].push(handler);
      return this;
    }
    if (
      event === "request" ||
      event === "response" ||
      event === "download" ||
      event === "popup" ||
      event === "console" ||
      event === "pageerror" ||
      event === "close" ||
      event === "load" ||
      event === "domcontentloaded" ||
      event === "frameattached" ||
      event === "framedetached" ||
      event === "framenavigated" ||
      event === "requestfailed" ||
      event === "requestfinished" ||
      event === "filechooser"
    ) {
      this._handlers[event] = this._handlers[event] || [];
      this._handlers[event].push(handler);
      return this;
    }
    return unsupported(`Page.on.${event}`)();
  }

  off(event, handler) {
    const list = this._handlers[event];
    if (!list) return this;
    this._handlers[event] = list.filter((item) => item !== handler);
    return this;
  }

  once(event, handler) {
    const wrap = (...args) => {
      this.off(event, wrap);
      return handler(...args);
    };
    return this.on(event, wrap);
  }

  addListener(event, handler) {
    return this.on(event, handler);
  }

  removeListener(event, handler) {
    return this.off(event, handler);
  }

  removeAllListeners(event) {
    if (event == null) {
      this._handlers = {};
    } else {
      this._handlers[event] = [];
    }
    return this;
  }

  prependListener(event, handler) {
    this._handlers[event] = this._handlers[event] || [];
    this._handlers[event].unshift(handler);
    return this;
  }

  async consoleMessages() {
    const result = await engineCall("page.consoleMessages", { page: this._id });
    return (result.messages || []).map((rec) => withUnsupported({
      type: () => rec.type || "log",
      text: () => rec.text || "",
      page: () => this,
    }, "ConsoleMessage"));
  }

  async pageErrors() {
    const messages = await this.consoleMessages();
    return messages
      .filter((msg) => msg.type() === "error")
      .map((msg) => {
        const error = new Error(msg.text());
        error.name = "Error";
        return error;
      });
  }

  async clearPageErrors() {
    await engineCall("page.clearPageErrors", { page: this._id });
  }

  async clearConsoleMessages() {
    await engineCall("page.clearConsoleMessages", { page: this._id });
    this._consoleSeen = 0;
  }

  async requests() {
    const result = await engineCall("page.requests", { page: this._id });
    const records = result.requests || [];
    return records.map((rec) => this._requestFromRecord(rec, records));
  }

  async opener() {
    if (this._openerId) {
      if (this._context && this._context._pages) {
        const found = this._context._pages.find((page) => page._id === this._openerId);
        if (found) return found;
      }
      const page = new Page(this._openerId);
      page._context = this._context;
      return page;
    }
    const result = await engineCall("page.opener", { page: this._id });
    if (!result.page) return null;
    if (this._context && this._context._pages) {
      const found = this._context._pages.find((page) => page._id === result.page);
      if (found) return found;
    }
    const page = new Page(result.page);
    page._context = this._context;
    return page;
  }

  async unroute(url) {
    await engineCall("page.unroute", { page: this._id, pattern: String(url) });
  }

  async unrouteAll() {
    await engineCall("page.unrouteAll", { page: this._id });
  }

  async routeFromHAR() {
    return unsupported("Page.routeFromHAR")();
  }

  async routeWebSocket() {
    return unsupported("Page.routeWebSocket")();
  }

  async emulateMedia() {
    return unsupported("Page.emulateMedia")();
  }

  async workers() {
    return unsupported("Page.workers")();
  }

  async pause() {
    return unsupported("Page.pause")();
  }

  async setExtraHTTPHeaders(headers) {
    await engineCall("page.setExtraHTTPHeaders", {
      page: this._id,
      headers: headers || {},
    });
  }

  async dragAndDrop(source, target) {
    await this.locator(source).dragTo(this.locator(target));
  }

  async route(url, handler) {
    const route = withUnsupported(
      {
        abort: () =>
          engineCall("page.addRoute", { page: this._id, pattern: String(url), action: "abort" }),
        continue: () =>
          engineCall("page.addRoute", { page: this._id, pattern: String(url), action: "continue" }),
        fulfill: (options = {}) =>
          engineCall("page.addRoute", {
            page: this._id,
            pattern: String(url),
            action: "fulfill",
            bodyBase64: encodeBase64(bytesFromFulfillBody(options.body)),
            byteLength: bytesFromFulfillBody(options.body).length,
            contentType: options.contentType || "text/html",
            status: options.status || 200,
          }),
      },
      "Route",
    );
    const result = handler(route);
    if (result && typeof result.then === "function") {
      await result;
    }
  }

  keyboard = {
    type: async (text) => {
      await engineCall("page.keyboard.type", { page: this._id, text: String(text) });
    },
    press: async (key) => {
      await engineCall("page.keyboard.press", { page: this._id, key: String(key) });
    },
    down: async (key) => {
      await engineCall("page.keyboard.down", { page: this._id, key: String(key) });
    },
    up: async (key) => {
      await engineCall("page.keyboard.up", { page: this._id, key: String(key) });
    },
    insertText: async (text) => {
      await engineCall("page.keyboard.insertText", { page: this._id, text: String(text) });
    },
  };

  async addInitScript(script) {
    const source =
      typeof script === "function" ? "(" + script.toString() + ")()" : String(script);
    await engineCall("page.addInitScript", { page: this._id, source });
  }

  async setViewportSize() {
    return unsupported("Page.setViewportSize")();
  }

  async viewportSize() {
    const result = await engineCall("page.viewportSize", { page: this._id });
    return { width: result.width, height: result.height };
  }
}

class BrowserContext {
  constructor(id) {
    this._id = id;
    this._pages = [];
    this._browser = null;
    this._pendingRoutes = [];
    this._extraHeaders = {};
    this._initScripts = [];
    this._closed = false;
    this._handlers = {};
    this.tracing = withUnsupported(
      {
        start: async () => unsupported("BrowserContext.tracing.start")(),
        stop: async () => unsupported("BrowserContext.tracing.stop")(),
      },
      "Tracing",
    );
    this.clock = withUnsupported(
      {
        install: unsupported("Clock.install"),
        fastForward: unsupported("Clock.fastForward"),
        pauseAt: unsupported("Clock.pauseAt"),
        resume: unsupported("Clock.resume"),
        runFor: unsupported("Clock.runFor"),
        setFixedTime: unsupported("Clock.setFixedTime"),
        setSystemTime: unsupported("Clock.setSystemTime"),
      },
      "Clock",
    );
    this.request = withUnsupported({}, "APIRequestContext");
    return withUnsupported(this, "BrowserContext");
  }

  browser() {
    return this._browser;
  }

  async newPage() {
    const result = await engineCall("context.newPage", { context: this._id });
    const page = new Page(result.page);
    page._context = this;
    this._lastPage = result.page;
    this._pages = this._pages || [];
    this._pages.push(page);
    for (const pending of this._pendingRoutes || []) {
      await page.route(pending.url, pending.handler);
    }
    if (this._extraHeaders && Object.keys(this._extraHeaders).length) {
      await page.setExtraHTTPHeaders(this._extraHeaders);
    }
    for (const source of this._initScripts || []) {
      await engineCall("page.addInitScript", { page: page._id, source });
    }
    this._emit("page", page);
    return page;
  }

  _emit(event, payload) {
    const list = (this._handlers && this._handlers[event]) || [];
    for (const handler of list) {
      const result = handler(payload);
      if (result && typeof result.then === "function") {
        result.catch(() => {});
      }
    }
  }

  on(event, handler) {
    if (
      event === "page" ||
      event === "close" ||
      event === "console" ||
      event === "dialog" ||
      event === "download" ||
      event === "request" ||
      event === "response" ||
      event === "requestfailed" ||
      event === "requestfinished" ||
      event === "frameattached" ||
      event === "framedetached" ||
      event === "framenavigated"
    ) {
      this._handlers = this._handlers || {};
      this._handlers[event] = this._handlers[event] || [];
      this._handlers[event].push(handler);
      return this;
    }
    throwUnsupported(`BrowserContext.on.${event}`);
  }

  off(event, handler) {
    const list = this._handlers && this._handlers[event];
    if (!list) return this;
    this._handlers[event] = list.filter((item) => item !== handler);
    return this;
  }

  once(event, handler) {
    const wrap = (...args) => {
      this.off(event, wrap);
      return handler(...args);
    };
    return this.on(event, wrap);
  }

  addListener(event, handler) {
    return this.on(event, handler);
  }

  removeListener(event, handler) {
    return this.off(event, handler);
  }

  removeAllListeners(event) {
    if (!this._handlers) return this;
    if (event == null) {
      this._handlers = {};
    } else {
      this._handlers[event] = [];
    }
    return this;
  }

  prependListener(event, handler) {
    this.on(event, handler);
    const list = this._handlers && this._handlers[event];
    if (list && list.length > 1) {
      const last = list.pop();
      list.unshift(last);
    }
    return this;
  }

  async routeWebSocket() {
    return unsupported("BrowserContext.routeWebSocket")();
  }

  async cookies() {
    const pages = this.pages();
    const out = [];
    const seen = new Set();
    for (const page of pages) {
      const result = await engineCall("page.cookies", { page: page._id });
      const raw = result.cookie || "";
      if (!raw) continue;
      for (const part of raw.split(";")) {
        const [name, ...rest] = part.trim().split("=");
        if (!name) continue;
        const value = rest.join("=");
        const key = name + "=" + value;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ name, value });
      }
    }
    return out;
  }

  async addCookies(cookies) {
    const pages = this.pages();
    if (!pages.length) {
      throwUnsupported("BrowserContext.addCookies.noPage");
    }
    for (const cookie of cookies || []) {
      if (cookie && cookie.httpOnly) {
        throwUnsupported("BrowserContext.addCookies.httpOnly");
      }
    }
    for (const page of pages) {
      await engineCall("page.addCookies", { page: page._id, cookies });
    }
  }

  async clearCookies() {
    for (const page of this.pages()) {
      await engineCall("page.clearCookies", { page: page._id });
    }
  }

  async setStorageState(state) {
    if (typeof state === "string") {
      throwUnsupported("BrowserContext.setStorageState.filePath");
    }
    return this._restoreStorageState(state);
  }

  async _restoreStorageState(state) {
    const cookies = (state && state.cookies) || [];
    const origins = (state && state.origins) || [];
    if (origins.some((origin) => origin && origin.indexedDB && origin.indexedDB.length)) {
      throwUnsupported("BrowserContext.storageState.indexedDB");
    }
    let page = this.pages()[0];
    if (!page) {
      page = await this.newPage();
    }
    for (const origin of origins) {
      if (!origin || !origin.origin) continue;
      const current = await page.url();
      if (!String(current).startsWith(origin.origin)) {
        await page.goto(origin.origin + "/");
      }
      await page.evaluate((items) => {
        for (const item of items || []) {
          if (item && item.name != null) {
            localStorage.setItem(String(item.name), String(item.value == null ? "" : item.value));
          }
        }
      }, origin.localStorage || []);
    }
    if (cookies.length) {
      await this.addCookies(cookies);
    }
  }

  async storageState() {
    const cookies = await this.cookies();
    const origins = [];
    const seen = new Set();
    for (const page of this.pages()) {
      const snapshot = await page.evaluate(() => {
        const origin = location.origin;
        if (!origin || origin === "null" || origin === "about:blank") {
          return null;
        }
        const localStorageItems = [];
        try {
          for (let i = 0; i < localStorage.length; i++) {
            const name = localStorage.key(i);
            localStorageItems.push({
              name,
              value: localStorage.getItem(name),
            });
          }
        } catch (_error) {
          return { origin, localStorage: [] };
        }
        return { origin, localStorage: localStorageItems };
      });
      if (snapshot && snapshot.origin && !seen.has(snapshot.origin)) {
        seen.add(snapshot.origin);
        origins.push(snapshot);
      }
    }
    return { cookies, origins };
  }

  async close() {
    for (const page of this.pages()) {
      await page.close();
    }
    this._pages = [];
    this._closed = true;
    if (this._browser && Array.isArray(this._browser._contexts)) {
      this._browser._contexts = this._browser._contexts.filter((context) => context !== this);
    }
    this._emit("close", this);
    await engineCall("context.close", { context: this._id });
  }

  isClosed() {
    return !!this._closed;
  }

  pages() {
    return this._pages || [];
  }

  setDefaultTimeout(ms) {
    this._timeout = Math.max(0, Number(ms) || 0);
    for (const page of this.pages()) {
      page.setDefaultTimeout(ms);
    }
  }

  setDefaultNavigationTimeout(ms) {
    this.setDefaultTimeout(ms);
  }

  async setExtraHTTPHeaders(headers) {
    this._extraHeaders = headers || {};
    for (const page of this.pages()) {
      await page.setExtraHTTPHeaders(this._extraHeaders);
    }
  }

  async addInitScript(script) {
    const source =
      typeof script === "function" ? "(" + script.toString() + ")()" : String(script);
    this._initScripts = this._initScripts || [];
    this._initScripts.push(source);
    for (const page of this.pages()) {
      await engineCall("page.addInitScript", { page: page._id, source });
    }
  }

  async unroute(url) {
    const pattern = String(url);
    this._pendingRoutes = (this._pendingRoutes || []).filter((item) => item.url !== pattern);
    for (const page of this.pages()) {
      await page.unroute(pattern);
    }
  }

  async unrouteAll() {
    this._pendingRoutes = [];
    for (const page of this.pages()) {
      await page.unrouteAll();
    }
  }

  async setOffline() {
    return unsupported("BrowserContext.setOffline")();
  }

  async grantPermissions() {
    return unsupported("BrowserContext.grantPermissions")();
  }

  async clearPermissions() {
    return unsupported("BrowserContext.clearPermissions")();
  }

  async setGeolocation() {
    return unsupported("BrowserContext.setGeolocation")();
  }

  async exposeFunction() {
    return unsupported("BrowserContext.exposeFunction")();
  }

  async exposeBinding() {
    return unsupported("BrowserContext.exposeBinding")();
  }

  async route(url, handler) {
    const pages = this.pages();
    if (pages.length) {
      for (const page of pages) {
        await page.route(url, handler);
      }
      return;
    }
    this._pendingRoutes = this._pendingRoutes || [];
    this._pendingRoutes.push({ url, handler });
  }
}

class Browser {
  constructor(id) {
    this._id = id;
    this._contexts = [];
    this._connected = true;
    this._handlers = {};
    return withUnsupported(this, "Browser");
  }

  async newContext(options) {
    const storageState = options && options.storageState;
    if (options != null) {
      const keys = Object.keys(options).filter((key) => options[key] !== undefined);
      if (keys.some((key) => key !== "storageState")) {
        return unsupported("Browser.newContext.options")();
      }
    }
    const result = await engineCall("browser.newContext", { browser: this._id });
    const context = new BrowserContext(result.context);
    context._browser = this;
    this._contexts.push(context);
    if (storageState) {
      await context._restoreStorageState(storageState);
    }
    return context;
  }

  contexts() {
    return this._contexts.slice();
  }

  isConnected() {
    return !!this._connected;
  }

  browserType() {
    return chromium;
  }

  async newPage(options) {
    if (options != null) {
      return unsupported("Browser.newPage.options")();
    }
    const context = await this.newContext();
    return context.newPage();
  }

  async close() {
    const contexts = this._contexts.slice();
    this._contexts = [];
    for (const context of contexts) {
      if (!context._closed) {
        await context.close();
      }
    }
    const wasConnected = this._connected;
    this._connected = false;
    if (wasConnected) {
      this._emit("disconnected");
    }
    await engineCall("browser.close", { browser: this._id });
  }

  _emit(event, payload) {
    const list = (this._handlers && this._handlers[event]) || [];
    for (const handler of list) {
      const result = handler(payload);
      if (result && typeof result.then === "function") {
        result.catch(() => {});
      }
    }
  }

  on(event, handler) {
    if (event === "disconnected") {
      this._handlers[event] = this._handlers[event] || [];
      this._handlers[event].push(handler);
      return this;
    }
    throwUnsupported(`Browser.on.${event}`);
  }

  off(event, handler) {
    const list = this._handlers[event];
    if (!list) return this;
    this._handlers[event] = list.filter((item) => item !== handler);
    return this;
  }

  once(event, handler) {
    const wrap = (...args) => {
      this.off(event, wrap);
      return handler(...args);
    };
    return this.on(event, wrap);
  }

  addListener(event, handler) {
    return this.on(event, handler);
  }

  removeListener(event, handler) {
    return this.off(event, handler);
  }

  removeAllListeners(event) {
    if (event == null) {
      this._handlers = {};
    } else {
      this._handlers[event] = [];
    }
    return this;
  }

  prependListener(event, handler) {
    if (event !== "disconnected") {
      throwUnsupported(`Browser.on.${event}`);
    }
    this._handlers[event] = this._handlers[event] || [];
    this._handlers[event].unshift(handler);
    return this;
  }

  version() {
    return "Servo 0.5.0";
  }
}

async function launchUnavailable(name) {
  const error = new Error(`browser_engine_not_available: ${name}`);
  error.code = "browser_engine_not_available";
  throw error;
}

const chromium = withUnsupported(
  {
    async launch(options) {
      if (options != null) {
        return unsupported("BrowserType.launch.options")();
      }
      const result = await engineCall("chromium.launch", {});
      return new Browser(result.browser);
    },
    name() {
      return "chromium";
    },
  },
  "BrowserType",
);

const firefox = withUnsupported(
  {
    async launch() {
      return launchUnavailable("firefox");
    },
    name() {
      return "firefox";
    },
  },
  "BrowserType",
);

const webkit = withUnsupported(
  {
    async launch() {
      return launchUnavailable("webkit");
    },
    name() {
      return "webkit";
    },
  },
  "BrowserType",
);

let testIdAttribute = "data-testid";
const selectors = withUnsupported(
  {
    setTestIdAttribute(name) {
      if (name == null || String(name) === "") {
        throwUnsupported("Selectors.setTestIdAttribute.empty");
      }
      testIdAttribute = String(name);
    },
  },
  "Selectors",
);
const errors = withUnsupported({ TimeoutError }, "errors");

export { chromium, firefox, webkit, selectors, errors, TimeoutError };
export const request = withUnsupported({}, "APIRequest");
export const devices = withUnsupported({}, "devices");
export default { chromium, firefox, webkit, request, selectors, devices, errors };
