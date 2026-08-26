import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const seen = [];
const handler = (msg) => {
  seen.push(msg.text());
};
page.on("console", handler);
await page.evaluate(() => {
  console.log("hello-console");
  return 1;
});
const messages = await page.consoleMessages();
const texts = messages.map((msg) => msg.text());
if (!texts.some((text) => String(text).includes("hello-console")) && !seen.some((text) => String(text).includes("hello-console"))) {
  throw new Error("console log not captured: " + JSON.stringify({ texts, seen }));
}
page.off("console", handler);
page.removeAllListeners("console");
await page.clearConsoleMessages();
const after = await page.consoleMessages();
if (after.length !== 0) throw new Error("clearConsoleMessages left " + after.length);
await browser.close();
