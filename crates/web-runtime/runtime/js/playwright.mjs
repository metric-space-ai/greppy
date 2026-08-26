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

  first() {
    return new Locator(this._page, { ...this._selector, nth: 0 });
  }

  nth(index) {
    return new Locator(this._page, { ...this._selector, nth: index });
  }

  async setInputFiles(files) {
    if (this._selector.type !== "css") {
      return unsupported("Locator.setInputFiles.nonCss")();
    }
    return this._page.setInputFiles(this._selector.value, files);
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
    return this._page.locator(`iframe:nth-of-type(${Number(this._id) + 1})`);
  }
}

class Page {
  constructor(id) {
    this._id = id;
    this._closed = false;
    return withUnsupported(this, "Page");
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

  locator(selector) {
    return new Locator(this, { type: "css", value: selector });
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

  async goto(url, options) {
    if (options != null) {
      return unsupported("Page.goto.options")();
    }
    const result = await engineCall("page.goto", { page: this._id, url });
    await this._dispatchNetwork();
    return result;
  }

  async _dispatchNetwork() {
    const result = await engineCall("page.requests", { page: this._id });
    const requests = result.requests || [];
    for (const rec of requests) {
      const request = {
        url: () => rec.url,
        method: () => "GET",
        resourceType: () => (rec.main_frame ? "document" : "other"),
        frame: () => this.mainFrame(),
      };
      if (this._requestHandler) this._requestHandler(request);
      if (this._responseHandler) {
        this._responseHandler({
          url: () => rec.url,
          status: () => 200,
          ok: () => true,
          request: () => request,
        });
      }
    }
  }

  async evaluate(pageFunction, arg) {
    return engineCall("page.evaluate", {
      page: this._id,
      source: serializeEvaluate(pageFunction, arg),
    }).then((result) => result.value);
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

  async waitForLoadState() {
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
      return this;
    }
    if (event === "request" || event === "response" || event === "download" || event === "popup") {
      this[`_${event}Handler`] = handler;
      return this;
    }
    return unsupported(`Page.on.${event}`)();
  }

  once(event, handler) {
    return this.on(event, handler);
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
  };
}

class BrowserContext {
  constructor(id) {
    this._id = id;
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
    this._lastPage = result.page;
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
    await engineCall("context.close", { context: this._id });
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
