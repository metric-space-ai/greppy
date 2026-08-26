const ops = Deno.core.ops;

function engineCall(method, params) {
  return ops.op_engine_call(method, params ?? {});
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
  }

  async click(options) {
    if (options != null) {
      return unsupported("Locator.click.options")();
    }
    if (this._selector && this._selector.type === "css" && this._selector.value) {
      this._page._lastFileSelector = this._selector.value;
    }
    await engineCall("locator.click", {
      page: this._page._id,
      selector: this._selector,
    });
    await this._page._flushPopups();
  }

  async tap(options) {
    if (options != null) {
      return unsupported("Locator.tap.options")();
    }
    await engineCall("locator.tap", {
      page: this._page._id,
      selector: this._selector,
    });
    await this._page._flushPopups();
  }

  async fill(value, options) {
    if (options != null) {
      return unsupported("Locator.fill.options")();
    }
    await engineCall("locator.fill", {
      page: this._page._id,
      selector: this._selector,
      value: String(value),
    });
  }

  async hover() {
    await engineCall("locator.hover", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async innerText() {
    const result = await engineCall("locator.innerText", {
      page: this._page._id,
      selector: this._selector,
    });
    return result.text;
  }

  async textContent() {
    return this.innerText();
  }

  async count() {
    const result = await engineCall("locator.count", {
      page: this._page._id,
      selector: this._selector,
    });
    return result.count;
  }

  async isVisible() {
    const result = await engineCall("locator.isVisible", {
      page: this._page._id,
      selector: this._selector,
    });
    return !!result.visible;
  }

  async waitFor() {
    await engineCall("locator.waitFor", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async check() {
    await engineCall("locator.check", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async uncheck() {
    await engineCall("locator.uncheck", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async selectOption(value) {
    await engineCall("locator.selectOption", {
      page: this._page._id,
      selector: this._selector,
      value: Array.isArray(value) ? value[0] : value,
    });
  }

  async inputValue() {
    const result = await engineCall("locator.inputValue", {
      page: this._page._id,
      selector: this._selector,
    });
    return result.value;
  }

  async getAttribute(name) {
    const result = await engineCall("locator.getAttribute", {
      page: this._page._id,
      selector: this._selector,
      name: String(name),
    });
    return result.value;
  }

  async isChecked() {
    const result = await engineCall("locator.isChecked", {
      page: this._page._id,
      selector: this._selector,
    });
    return !!result.checked;
  }

  async isEnabled() {
    const result = await engineCall("locator.isEnabled", {
      page: this._page._id,
      selector: this._selector,
    });
    return !!result.enabled;
  }

  async isDisabled() {
    const result = await engineCall("locator.isDisabled", {
      page: this._page._id,
      selector: this._selector,
    });
    return !!result.disabled;
  }

  async isHidden() {
    const result = await engineCall("locator.isHidden", {
      page: this._page._id,
      selector: this._selector,
    });
    return !!result.hidden;
  }

  async innerHTML() {
    const result = await engineCall("locator.innerHTML", {
      page: this._page._id,
      selector: this._selector,
    });
    return result.html;
  }

  async focus() {
    await engineCall("locator.focus", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async blur() {
    await engineCall("locator.blur", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async boundingBox() {
    return engineCall("locator.boundingBox", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async screenshot() {
    const result = await engineCall("locator.screenshot", {
      page: this._page._id,
      selector: this._selector,
    });
    return decodeBase64(result.png_base64 || "").buffer;
  }

  async allTextContents() {
    const result = await engineCall("locator.allTextContents", {
      page: this._page._id,
      selector: this._selector,
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
      page: this._page._id,
      selector: this._selector,
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
      page: this._page._id,
      selector: this._selector,
      source,
    });
    return result.value;
  }

  async dblclick(options) {
    if (options != null) {
      return unsupported("Locator.dblclick.options")();
    }
    await engineCall("locator.dblclick", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async dispatchEvent(type) {
    await engineCall("locator.dispatchEvent", {
      page: this._page._id,
      selector: this._selector,
      event: String(type),
    });
  }

  async clear() {
    await this.fill("");
  }

  async isEditable() {
    const result = await engineCall("locator.isEditable", {
      page: this._page._id,
      selector: this._selector,
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
    return new Locator(this._page, {
      type: "role",
      role,
      name: options.name ?? null,
      scope: this._selector,
    });
  }

  getByText(text) {
    return new Locator(this._page, { type: "text", value: String(text), scope: this._selector });
  }

  getByLabel(name) {
    return new Locator(this._page, { type: "label", name, scope: this._selector });
  }

  getByPlaceholder(name) {
    return new Locator(this._page, { type: "placeholder", name: String(name), scope: this._selector });
  }

  getByAltText(name) {
    return new Locator(this._page, { type: "alt", name: String(name), scope: this._selector });
  }

  getByTitle(name) {
    return new Locator(this._page, { type: "title", name: String(name), scope: this._selector });
  }

  getByTestId(name) {
    return new Locator(this._page, {
      type: "testid",
      name: String(name),
      attr: "data-testid",
      scope: this._selector,
    });
  }

  filter(options = {}) {
    return new Locator(this._page, {
      type: "filter",
      hasText: options.hasText != null ? String(options.hasText) : null,
      scope: this._selector,
    });
  }

  async scrollIntoViewIfNeeded() {
    await engineCall("locator.scrollIntoViewIfNeeded", {
      page: this._page._id,
      selector: this._selector,
    });
  }

  async selectText() {
    await engineCall("locator.selectText", {
      page: this._page._id,
      selector: this._selector,
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
}

class FrameLocator {
  constructor(page, frameSelector) {
    this._page = page;
    this._frame = frameSelector;
  }

  locator(selector) {
    return new Locator(this._page, {
      type: "framecss",
      frame: this._frame,
      value: selector,
    });
  }

  getByText(text) {
    return new Locator(this._page, {
      type: "frametext",
      frame: this._frame,
      value: String(text),
    });
  }

  getByRole(role, options = {}) {
    return new Locator(this._page, {
      type: "framerole",
      frame: this._frame,
      role,
      name: options.name ?? null,
    });
  }
}

class Frame {
  constructor(page, info) {
    this._page = page;
    this._id = info.id;
    this._name = info.name || "";
    this._url = info.url || "";
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
      return unsupported("Frame.goto.child")();
    }
    return this._page.goto(url, options);
  }

  async setContent(html) {
    if (this._id !== "main") {
      return unsupported("Frame.setContent.child")();
    }
    return this._page.setContent(html);
  }

  async waitForSelector(selector) {
    await this.locator(selector).waitFor();
  }

  async waitForLoadState(state) {
    if (this._id !== "main") {
      return unsupported("Frame.waitForLoadState.child")();
    }
    return this._page.waitForLoadState(state);
  }

  async addScriptTag(options) {
    if (this._id !== "main") {
      return unsupported("Frame.addScriptTag.child")();
    }
    return this._page.addScriptTag(options);
  }

  async addStyleTag(options) {
    if (this._id !== "main") {
      return unsupported("Frame.addStyleTag.child")();
    }
    return this._page.addStyleTag(options);
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
    if (!this._isMain()) {
      throwUnsupported("Frame.waitForTimeout.child");
    }
    return this._page.waitForTimeout(ms);
  }

  waitForFunction(pageFunction, arg) {
    if (!this._isMain()) {
      throwUnsupported("Frame.waitForFunction.child");
    }
    return this._page.waitForFunction(pageFunction, arg);
  }

  waitForURL(pattern) {
    if (!this._isMain()) {
      throwUnsupported("Frame.waitForURL.child");
    }
    return this._page.waitForURL(pattern);
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
    return this._page.frames();
  }
}

class Page {
  constructor(id) {
    this._id = id;
    this._closed = false;
    this._timeout = 30_000;
    this._context = null;
    this._handlers = {};
    this._consoleSeen = 0;
    this._popupWaiters = [];
    this._pendingPopups = [];
    this._openerId = null;
    this._navWaiters = [];
    this._consoleWaiters = [];
    this._pendingConsole = [];
    this._mouseX = 0;
    this._mouseY = 0;
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
    return new Locator(this, {
      type: "role",
      role,
      name: options.name ?? null,
    });
  }

  getByLabel(name) {
    return new Locator(this, { type: "label", name });
  }

  getByText(text) {
    return new Locator(this, { type: "text", value: String(text) });
  }

  getByPlaceholder(name) {
    return new Locator(this, { type: "placeholder", name: String(name) });
  }

  getByAltText(name) {
    return new Locator(this, { type: "alt", name: String(name) });
  }

  getByTitle(name) {
    return new Locator(this, { type: "title", name: String(name) });
  }

  getByTestId(name) {
    return new Locator(this, { type: "testid", name: String(name), attr: "data-testid" });
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
    throw new Error("timeout: waitForFunction");
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
    throw new Error("timeout: waitForURL " + needle);
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
    throw new Error("timeout: waitForRequest " + needle);
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
    const result = await engineCall("page.goto", { page: this._id, url });
    await this._flushPopups();
    await this._flushNavigation();
    await this._dispatchNetwork();
    return result;
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
    return {
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
    };
  }

  _responseFromRecord(rec, request) {
    const headers = rec.headers || {};
    const status = Number(rec.status) || 200;
    const bytes = () => decodeBase64(rec.bodyBase64 || "");
    return {
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
    };
  }

  _downloadFromRecord(rec) {
    const page = this;
    return {
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
    };
  }

  async _dispatchNetwork() {
    const result = await engineCall("page.requests", { page: this._id });
    const responses = ((await engineCall("page.responses", { page: this._id })).responses || []);
    const requests = result.requests || [];
    for (const rec of requests) {
      const request = this._requestFromRecord(rec);
      this._emit("request", request);
      const hit = responses.find((row) => row.url === rec.url);
      if (hit) {
        this._emit("response", this._responseFromRecord(hit, request));
      }
    }
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

  async evaluate(pageFunction, arg) {
    const result = await engineCall("page.evaluate", {
      page: this._id,
      source: serializeEvaluate(pageFunction, arg),
    });
    await this._flushPopups();
    await this._dispatchConsole();
    return result.value;
  }

  async _dispatchConsole() {
    this._consoleWaiters = this._consoleWaiters || [];
    this._pendingConsole = this._pendingConsole || [];
    const result = await engineCall("page.consoleMessages", { page: this._id });
    const messages = result.messages || [];
    for (let i = this._consoleSeen; i < messages.length; i++) {
      const rec = messages[i];
      const payload = {
        type: () => rec.type || "log",
        text: () => rec.text || "",
        page: () => this,
      };
      const waiter = this._consoleWaiters.shift();
      if (waiter) {
        waiter(payload);
      } else {
        this._pendingConsole.push(payload);
      }
      this._emit("console", payload);
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
    if (options != null) {
      return unsupported("Page.screenshot.options")();
    }
    const result = await engineCall("page.screenshot", { page: this._id });
    const binary = result.png_base64 || "";
    return decodeBase64(binary).buffer;
  }

  async setContent(html) {
    await engineCall("page.setContent", { page: this._id, html: String(html) });
  }

  async reload() {
    await engineCall("page.reload", { page: this._id });
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
    return (result.frames || []).map((info) => new Frame(this, info));
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
  }

  async isClosed() {
    if (this._closed) return true;
    const result = await engineCall("page.isClosed", { page: this._id });
    return !!result.closed;
  }

  async waitForEvent(event) {
    if (event === "dialog") {
      const result = await engineCall("page.dialogs", { page: this._id, consume: true });
      const rec = (result.dialogs || [])[0];
      if (!rec) {
        return unsupported("Page.waitForEvent.dialog.empty")();
      }
      return new Dialog(this, rec);
    }
    if (event === "filechooser") {
      const result = await engineCall("page.fileChoosers", { page: this._id, consume: true });
      const rec = (result.choosers || [])[0];
      if (!rec) {
        return unsupported("Page.waitForEvent.filechooser.empty")();
      }
      return new FileChooser(this, rec);
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
      return this._downloadFromRecord(rec);
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
      event === "console"
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
    return (result.messages || []).map((rec) => ({
      type: () => rec.type || "log",
      text: () => rec.text || "",
      page: () => this,
    }));
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

  async dragAndDrop() {
    return unsupported("Page.dragAndDrop")();
  }

  async route(url, handler) {
    const route = {
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
    };
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
      await engineCall("page.keyboard.type", { page: this._id, text: String(text) });
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
    this.tracing = {
      start: async () => unsupported("BrowserContext.tracing.start")(),
      stop: async () => unsupported("BrowserContext.tracing.stop")(),
    };
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
    return page;
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
    if (storageState) {
      await context._restoreStorageState(storageState);
    }
    return context;
  }

  async newPage(options) {
    if (options != null) {
      return unsupported("Browser.newPage.options")();
    }
    const context = await this.newContext();
    return context.newPage();
  }

  async close() {
    await engineCall("browser.close", { browser: this._id });
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

export { chromium, firefox, webkit };
export const request = withUnsupported({}, "APIRequest");
export const selectors = withUnsupported({}, "Selectors");
export const devices = withUnsupported({}, "devices");
export const errors = withUnsupported({}, "errors");
export default { chromium, firefox, webkit, request, selectors, devices, errors };
