import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><head><title>Oracle</title></head><body><p id='x'>ok</p></body></html>",
);
const title = await page.title();
const value = await page.evaluate(() => 1 + 1);
const text = (await page.locator("#x").innerText()).trim();
if (title !== "Oracle" || value !== 2 || text !== "ok") {
  throw new Error(JSON.stringify({ title, value, text }));
}
const html = await page.content();
if (!String(html).includes("ok") || !/oracle/i.test(String(html))) {
  throw new Error("content missing markers: " + String(html).slice(0, 200));
}
if ((await page.locator("#x").count()) !== 1) {
  throw new Error("count");
}
if ((await page.locator("#x").innerHTML()).trim() !== "ok") {
  throw new Error("innerHTML");
}
if ((await page.innerHTML("#x")).trim() !== "ok") {
  throw new Error("page.innerHTML");
}
await browser.close();
