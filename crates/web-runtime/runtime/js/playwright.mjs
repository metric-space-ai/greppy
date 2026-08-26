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
    await engineCall("locator.click", {
      page: this._page._id,
      selector: this._selector,
    });
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

  async boundingBox() {
    return engineCall("locator.boundingBox", {
      page: this._page._id,
      selector: this._selector,
    });
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

  locator(selector) {
    if (this._id === "main" || Number.isNaN(Number(this._id))) {
      return this._page.locator(selector);
    }
    return this._page.locator("iframe:nth-of-type(" + (Number(this._id) + 1) + ")");
  }

  getByRole(role, options) {
    return this._id === "main"
      ? this._page.getByRole(role, options)
      : this.locator("body").first();
  }

  getByText(text) {
    return this._id === "main" ? this._page.getByText(text) : this.locator("body").first();
  }

  getByLabel(name) {
    return this._id === "main" ? this._page.getByLabel(name) : this.locator("body").first();
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
}

class Page {
  constructor(id) {
    this._id = id;
    this._closed = false;
    this._timeout = 30_000;
    this._context = null;
    this._handlers = {};
    this._consoleSeen = 0;
    this.mouse = {
      click: async (x, y) => {
        await engineCall("page.mouse.click", { page: this._id, x, y });
      },
      move: async (x, y) => {
        await engineCall("page.mouse.move", { page: this._id, x, y });
      },
      down: async () => {
        await engineCall("page.mouse.down", { page: this._id, x: 0, y: 0 });
      },
      up: async () => {
        await engineCall("page.mouse.up", { page: this._id, x: 0, y: 0 });
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

  tap(selector, options) {
    return this.locator(selector).click(options);
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
      const hit = (result.requests || []).find((rec) => String(rec.url).includes(needle));
      if (hit) {
        return this._requestFromRecord(hit);
      }
      ops.op_sleep_ms(20);
    }
    throw new Error("timeout: waitForRequest " + needle);
  }

  async waitForResponse(pattern) {
    const request = await this.waitForRequest(pattern);
    return {
      url: () => request.url(),
      status: () => 200,
      ok: () => true,
      request: () => request,
    };
  }

  async goto(url, options) {
    if (options != null) {
      return unsupported("Page.goto.options")();
    }
    const result = await engineCall("page.goto", { page: this._id, url });
    await this._dispatchNetwork();
    return result;
  }

  _requestFromRecord(rec) {
    const headerList = rec.headers || [];
    return {
      url: () => rec.url,
      method: () => rec.method || "GET",
      headers: () => {
        const out = {};
        headerList.forEach((h) => {
          out[String(h.name).toLowerCase()] = h.value;
        });
        return out;
      },
      resourceType: () => (rec.main_frame ? "document" : "other"),
      frame: () => this.mainFrame(),
    };
  }

  async _dispatchNetwork() {
    const result = await engineCall("page.requests", { page: this._id });
    const requests = result.requests || [];
    for (const rec of requests) {
      const request = this._requestFromRecord(rec);
      this._emit("request", request);
      this._emit("response", {
        url: () => rec.url,
        status: () => 200,
        ok: () => true,
        request: () => request,
        text: async () => "",
      });
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
    await this._dispatchConsole();
    return result.value;
  }

  async _dispatchConsole() {
    const result = await engineCall("page.consoleMessages", { page: this._id });
    const messages = result.messages || [];
    for (let i = this._consoleSeen; i < messages.length; i++) {
      const rec = messages[i];
      this._emit("console", {
        type: () => rec.type || "log",
        text: () => rec.text || "",
      });
    }
    this._consoleSeen = messages.length;
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
    return engineCall("page.goBack", { page: this._id });
  }

  async goForward() {
    return engineCall("page.goForward", { page: this._id });
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
      const result = await engineCall("page.popups", { page: this._id });
      const id = (result.pages || [])[0];
      if (!id) {
        return null;
      }
      return new Page(id);
    }
    if (event === "download") {
      const result = await engineCall("page.downloads", { page: this._id });
      return (result.downloads || [])[0] || null;
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

  async consoleMessages() {
    const result = await engineCall("page.consoleMessages", { page: this._id });
    return (result.messages || []).map((rec) => ({
      type: () => rec.type || "log",
      text: () => rec.text || "",
    }));
  }

  async emulateMedia() {
    return unsupported("Page.emulateMedia")();
  }

  async workers() {
    return [];
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
          body: options.body || "",
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
    down: async () => unsupported("Keyboard.down")(),
    up: async () => unsupported("Keyboard.up")(),
    insertText: async (text) => {
      await engineCall("page.keyboard.type", { page: this._id, text: String(text) });
    },
  };

  async addInitScript(script) {
    const source =
      typeof script === "function" ? "(" + script.toString() + ")()" : String(script);
    await engineCall("page.addInitScript", { page: this._id, source });
  }

  async setViewportSize(size) {
    await engineCall("page.setViewportSize", {
      page: this._id,
      width: size.width,
      height: size.height,
    });
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
    this.tracing = {
      start: async () => {
        this._tracing = true;
      },
      stop: async () => {
        return engineCall("page.tracing", { page: this._lastPage || "page-1" });
      },
    };
    return withUnsupported(this, "BrowserContext");
  }

  async newPage() {
    const result = await engineCall("context.newPage", { context: this._id });
    const page = new Page(result.page);
    page._context = this;
    this._lastPage = result.page;
    this._pages = this._pages || [];
    this._pages.push(page);
    return page;
  }

  async cookies() {
    if (!this._lastPage) return [];
    const result = await engineCall("page.cookies", { page: this._lastPage });
    const raw = result.cookie || "";
    if (!raw) return [];
    return raw.split(";").map((part) => {
      const [name, ...rest] = part.trim().split("=");
      return { name, value: rest.join("=") };
    });
  }

  async addCookies(cookies) {
    if (!this._lastPage) return;
    await engineCall("page.addCookies", { page: this._lastPage, cookies });
  }

  async clearCookies() {
    if (!this._lastPage) return;
    await engineCall("page.clearCookies", { page: this._lastPage });
  }

  async storageState() {
    return { cookies: await this.cookies(), origins: [] };
  }

  async close() {
    for (const page of this.pages()) {
      await page.close();
    }
    this._pages = [];
    await engineCall("context.close", { context: this._id });
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

  async route(url, handler) {
    const pages = this.pages();
    if (pages[0]) {
      return pages[0].route(url, handler);
    }
  }
}

class Browser {
  constructor(id) {
    this._id = id;
    return withUnsupported(this, "Browser");
  }

  async newContext(options) {
    if (options != null) {
      return unsupported("Browser.newContext.options")();
    }
    const result = await engineCall("browser.newContext", { browser: this._id });
    return new BrowserContext(result.context);
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
export const devices = {};
export const errors = withUnsupported({}, "errors");
export default { chromium, firefox, webkit, request, selectors, devices, errors };
