import { chromium } from "playwright";

async function expectDisposed(label, fn) {
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      await result;
    }
    throw new Error(label + " did not throw");
  } catch (error) {
    const message = String(error && error.message ? error.message : error);
    if ((error && error.code === "object_disposed") || message.includes("object_disposed")) {
      return;
    }
    throw new Error(label + " threw " + message);
  }
}

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><button id='b'>Go</button></body></html>");
await page.close();
await expectDisposed("Page.evaluate after close", () => page.evaluate(() => 1));
await expectDisposed("Locator.click after close", () => page.locator("#b").click());
if (!(await page.isClosed())) {
  throw new Error("Page.isClosed should be true after close");
}

const livePage = await browser.newPage();
await livePage.setContent("<!DOCTYPE html><html><body><p>g</p></body></html>");
const pageGen = livePage._generation;
livePage._generation = pageGen + 9;
await expectDisposed("Page.evaluate stale generation", () => livePage.evaluate(() => 1));
livePage._generation = pageGen;
if ((await livePage.evaluate(() => 3)) !== 3) {
  throw new Error("restored page generation must work");
}
await livePage.close();

const contextA = await browser.newContext();
const pageA = await contextA.newPage();
await pageA.setContent("<!DOCTYPE html><html><body><p id='a'>A</p></body></html>");
const contextB = await browser.newContext();
const pageB = await contextB.newPage();
await pageB.setContent("<!DOCTYPE html><html><body><p id='b'>B</p></body></html>");
if ((await pageB.evaluate(() => document.getElementById("b").textContent)) !== "B") {
  throw new Error("sibling context page should be live before close");
}

await contextA.close();
await expectDisposed("Page.evaluate after owning context.close", () => pageA.evaluate(() => 1));
await expectDisposed("Context.newPage after close", () => contextA.newPage());
if ((await pageB.evaluate(() => document.getElementById("b").textContent)) !== "B") {
  throw new Error("sibling context page was disposed with another context");
}
const contextGen = contextB._generation;
contextB._generation = contextGen + 9;
await expectDisposed("Context.newPage stale generation", () => contextB.newPage());
contextB._generation = contextGen;
const browserGen = browser._generation;
browser._generation = browserGen + 9;
await expectDisposed("Browser.newContext stale generation", () => browser.newContext());
browser._generation = browserGen;
const pageB2 = await contextB.newPage();
if ((await pageB2.evaluate(() => 7)) !== 7) {
  throw new Error("live context must still create pages");
}

const page2 = await browser.newPage();
await page2.setContent("<!DOCTYPE html><html><body><p>ok</p></body></html>");
await browser.close();
await expectDisposed("Page.evaluate after browser.close", () => page2.evaluate(() => 1));
await expectDisposed("Browser.newPage after close", () => browser.newPage());
await expectDisposed("Browser.newContext after close", () => browser.newContext());
