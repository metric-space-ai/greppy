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

  async innerText() {
    const result = await engineCall("locator.innerText", {
      page: this._page._id,
      selector: this._selector,
    });
    return result.text;
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
}

class BrowserContext {
  constructor(id) {
    this._id = id;
    return withUnsupported(this, "BrowserContext");
  }

  async newPage() {
    const result = await engineCall("context.newPage", { context: this._id });
    return new Page(result.page);
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
