import { chromium, selectors, errors, Debugger, Credentials, Logger, WebError } from "playwright";

async function expectUnsupported(label, fn) {
  let failed = false;
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      try {
        await result;
      } catch (error) {
        if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
        return;
      }
      throw new Error(label + " resolved instead of throwing");
    }
  } catch (error) {
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    failed = true;
  }
  if (!failed) throw new Error(label + " did not throw");
}

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.setContent("<!DOCTYPE html><html><body><button id='b'>Go</button></body></html>");
await page.evaluate(() => console.log("closed-log"));

await expectUnsupported("Selectors.register", () => selectors.register("foo"));
await expectUnsupported("Page.workers", () => page.workers());
await expectUnsupported("Screencast.start", () => page.screencast.start());
await expectUnsupported("Keyboard.delay", () => page.keyboard.delay());
await expectUnsupported("Keyboard.type delay", () => page.keyboard.type("x", { delay: 5 }));
await expectUnsupported("Keyboard.press delay", () => page.keyboard.press("a", { delay: 5 }));
await expectUnsupported("Mouse.button", () => page.mouse.button());
await expectUnsupported("Mouse.click button", () => page.mouse.click(1, 1, { button: "right" }));
await expectUnsupported("Coverage.reportAnonymousScripts", () => page.coverage.reportAnonymousScripts());
await expectUnsupported("WebStorage.clear", () => page.localStorage.clear());
await expectUnsupported("WebStorage.getItem", () => page.sessionStorage.getItem("x"));
await expectUnsupported("Page.prependListener websocket", () => page.prependListener("websocket", () => {}));
await expectUnsupported("Page.on websocket", () => page.on("websocket", () => {}));
await expectUnsupported("Page.on worker", () => page.on("worker", () => {}));
await expectUnsupported("Page.on crash", () => page.on("crash", () => {}));
await expectUnsupported("BrowserContext.on pageload", () => context.on("pageload", () => {}));
await expectUnsupported("BrowserContext.on pageclose", () => context.on("pageclose", () => {}));
await expectUnsupported("BrowserContext.on backgroundpage", () => context.on("backgroundpage", () => {}));
await expectUnsupported("BrowserContext.on serviceworker", () => context.on("serviceworker", () => {}));
await expectUnsupported("Browser.on context", () => browser.on("context", () => {}));
await expectUnsupported("Tracing.group", () => context.tracing.group("g"));
await expectUnsupported("Tracing.startChunk", () => context.tracing.startChunk());
await expectUnsupported("Tracing.groupEnd", () => context.tracing.groupEnd());
await expectUnsupported("BrowserContext.routeWebSocket", () => context.routeWebSocket("**"));
await expectUnsupported("Locator.elementHandles", () => page.locator("#b").elementHandles());
await expectUnsupported("Locator.normalize", () => page.locator("#b").normalize());
await page.route("https://example.invalid/fail-closed", async (route) => {
  await expectUnsupported("Route.fallback", () => route.fallback());
  await expectUnsupported("Route.fetch", () => route.fetch());
  await expectUnsupported("Route.request", () => route.request());
});
await expectUnsupported("chromium.connect", () => chromium.connect());
await expectUnsupported("chromium.launchPersistentContext", () => chromium.launchPersistentContext());
await expectUnsupported("chromium.launchServer", () => chromium.launchServer());
await expectUnsupported("chromium.executablePath", () => chromium.executablePath());
await expectUnsupported("Page.addLocatorHandler", () => page.addLocatorHandler());
await expectUnsupported("Page.removeLocatorHandler", () => page.removeLocatorHandler());
await expectUnsupported("Page.requestGC", () => page.requestGC());
await expectUnsupported("Page.video", () => page.video());
await expectUnsupported("Coverage.startJSCoverage", () => page.coverage.startJSCoverage());
await expectUnsupported("Coverage.startCSSCoverage", () => page.coverage.startCSSCoverage());
await expectUnsupported("Clock.install", () => context.clock.install());
await expectUnsupported("Clock.fastForward", () => context.clock.fastForward());
await expectUnsupported("Clock.time", () => context.clock.time());
await expectUnsupported("Browser.bind", () => browser.bind());
await expectUnsupported("Browser.unbind", () => browser.unbind());
await expectUnsupported("APIRequestContext.get context", () => context.request.get("https://example.com"));
await expectUnsupported("APIRequestContext.get page", () => page.request.get("https://example.com"));
await expectUnsupported("BrowserContext.setHTTPCredentials", () => context.setHTTPCredentials({ username: "a", password: "b" }));
await expectUnsupported("BrowserContext.backgroundPages", () => context.backgroundPages());
await expectUnsupported("BrowserContext.serviceWorkers", () => context.serviceWorkers());
await expectUnsupported("Locator.elementHandle", () => page.locator("#b").elementHandle());
await expectUnsupported("Locator.evaluateHandle", () => page.locator("#b").evaluateHandle(() => 1));
await expectUnsupported("Frame.evaluateHandle", () => page.mainFrame().evaluateHandle(() => 1));
await expectUnsupported("Frame.frameElement", () => page.mainFrame().frameElement());
await expectUnsupported("Frame.dragAndDrop", () => page.mainFrame().dragAndDrop("#b", "#b"));
await expectUnsupported("Frame.setInputFiles", () => page.mainFrame().setInputFiles("#b", "x"));
await expectUnsupported("Locator.drop", () => page.locator("#b").drop(page.locator("#b")));
await expectUnsupported("Locator.hideHighlight", () => page.locator("#b").hideHighlight());
await expectUnsupported("BrowserContext.routeFromHAR", () => context.routeFromHAR("x.har"));
await expectUnsupported("BrowserContext.waitForEvent", () => context.waitForEvent("page"));
const messages = await page.consoleMessages();
if (!messages.length) throw new Error("expected console message");
await expectUnsupported("ConsoleMessage.args", () => messages[0].args());
await expectUnsupported("ConsoleMessage.location", () => messages[0].location());
await expectUnsupported("ConsoleMessage.timestamp", () => messages[0].timestamp());
await expectUnsupported("ConsoleMessage.worker", () => messages[0].worker());
await expectUnsupported("Debugger.requestPause", () => Debugger.requestPause());
await expectUnsupported("Credentials.create", () => Credentials.create());
await expectUnsupported("Logger.log", () => Logger.log("x"));
await expectUnsupported("WebError.error", () => WebError.error());
await expectUnsupported("BrowserContext.credentials.create", () => context.credentials.create());
await expectUnsupported("BrowserContext.debugger.requestPause", () => context.debugger.requestPause());
await expectUnsupported("Browser.logger.log", () => browser.logger.log("x"));
await expectUnsupported("errors.WebError", () => errors.WebError());
await browser.close();
