import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const seen = [];
let onceCount = 0;
page.once("request", () => {
  onceCount += 1;
});
page.on("request", (request) => {
  seen.push(request.url());
});
await page.setContent(
  "<!DOCTYPE html><html><body><p class='x'>a</p><p class='x'>b</p></body></html>",
);
const texts = await page.locator("p.x").allTextContents();
if (texts.length !== 2 || texts[0] !== "a" || texts[1] !== "b") {
  throw new Error("allTextContents " + JSON.stringify(texts));
}
const box = await page.locator("p.x").nth(0).boundingBox();
if (!box || box.width <= 0 || box.height <= 0) {
  throw new Error("boundingBox " + JSON.stringify(box));
}
await page.goto(fixtureUrl);
if (seen.length < 1 || !String(seen[0]).includes("http")) {
  throw new Error("expected request event after goto, got " + JSON.stringify(seen));
}
await page.goto(fixtureUrl.replace(/\/?$/, "/two"));
if (onceCount !== 1) {
  throw new Error("Page.once should fire once, got " + onceCount);
}
await browser.close();