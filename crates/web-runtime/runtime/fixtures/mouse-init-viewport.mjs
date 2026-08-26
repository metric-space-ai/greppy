import { chromium } from "playwright";

const browser = await chromium.launch();
if (!String(browser.version()).includes("Servo")) {
  throw new Error("version " + browser.version());
}
const context = await browser.newContext();
context.setDefaultTimeout(5_000);
const page = await context.newPage();
if (context.pages().length !== 1) throw new Error("pages");
if (page.context() !== context) throw new Error("context");
page.setDefaultTimeout(5_000);
const vp = await page.viewportSize();
if (vp.width !== 800 || vp.height !== 600) throw new Error("viewport " + JSON.stringify(vp));
await page.addInitScript(() => {
  window.__init = 1;
});
await page.setContent("<!DOCTYPE html><html><body><li>a</li><li>b</li><button id='m'>M</button></body></html>");
const init = await page.evaluate(() => window.__init);
if (init !== 1) throw new Error("init script " + JSON.stringify(init));
const last = await page.locator("li").last().innerText();
if (last.trim() !== "b") throw new Error("last " + last);
const all = await page.locator("li").all();
if (all.length !== 2) throw new Error("all " + all.length);
await page.mouse.move(10, 10);
await page.mouse.click(12, 12);
const main = page.mainFrame();
if ((await main.locator("li").count()) !== 2) throw new Error("main frame count");
if ((await main.getByText("a").count()) < 1) throw new Error("main getByText");
await browser.close();
