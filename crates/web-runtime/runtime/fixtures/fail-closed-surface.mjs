import { chromium } from "playwright";

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
await browser.close();
