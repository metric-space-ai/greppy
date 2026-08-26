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
      if (prop in obj) {
        return obj[prop];
      }
      return unsupported(`${prefix}.${String(prop)}`);
    },
  });
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

  first() {
    return new Locator(this._page, { ...this._selector, nth: 0 });
  }

  nth(index) {
    return new Locator(this._page, { ...this._selector, nth: index });
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

  async goto(url, options) {
    if (options != null) {
      return unsupported("Page.goto.options")();
    }
    return engineCall("page.goto", { page: this._id, url });
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
    const bytes = Uint8Array.from(atob(binary), (c) => c.charCodeAt(0));
    return bytes.buffer;
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

  async waitForEvent(event) {
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
    await engineCall("page.setInputFiles", { page: this._id, files: list.map(String) });
    await this.locator(selector).click();
  }

  on(event, handler) {
    if (event === "dialog") {
      const dialog = {
        type: () => "alert",
        message: () => "",
        defaultValue: () => "",
        accept: (prompt) =>
          engineCall("page.setDialogPolicy", {
            page: this._id,
            action: "accept",
            prompt: prompt ?? null,
          }),
        dismiss: () =>
          engineCall("page.setDialogPolicy", { page: this._id, action: "dismiss" }),
      };
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

  async storageState() {
    return { cookies: await this.cookies(), origins: [] };
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
