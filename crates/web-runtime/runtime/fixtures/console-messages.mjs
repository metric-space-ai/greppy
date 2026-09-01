import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const seen = [];
const handler = (msg) => {
  seen.push(msg.text());
};
page.on("console", handler);
const ctxConsole = [];
page.context().on("console", (msg) => {
  ctxConsole.push(msg.text());
});
const pageErrorsSeen = [];
page.on("pageerror", (error) => {
  pageErrorsSeen.push(String(error && error.message));
});
const waited = page.waitForEvent("console");
await page.evaluate(() => {
  console.log("hello-console");
  console.error("boom-err");
  return 1;
});
const waitedMsg = await waited;
if (waitedMsg.text() !== "hello-console") {
  throw new Error("waitForEvent console " + waitedMsg.text());
}
if (typeof waitedMsg.page !== "function" || waitedMsg.page() !== page) {
  throw new Error("ConsoleMessage.page");
}
const errors = await page.pageErrors();
if (!errors.some((error) => String(error.message).includes("boom-err"))) {
  throw new Error("pageErrors missing boom-err: " + JSON.stringify(errors.map((e) => e.message)));
}
const waitedError = await page.waitForEvent("pageerror");
if (!String(waitedError.message).includes("boom-err")) {
  throw new Error("waitForEvent pageerror " + waitedError.message);
}
if (!pageErrorsSeen.some((message) => message.includes("boom-err"))) {
  throw new Error("page.on pageerror " + JSON.stringify(pageErrorsSeen));
}
await page.clearPageErrors();
const afterErrors = await page.pageErrors();
if (afterErrors.some((error) => String(error.message).includes("boom-err"))) {
  throw new Error("clearPageErrors left boom-err");
}
const messages = await page.consoleMessages();
const texts = messages.map((msg) => msg.text());
if (!texts.some((text) => String(text).includes("hello-console")) && !seen.some((text) => String(text).includes("hello-console"))) {
  throw new Error("console log not captured: " + JSON.stringify({ texts, seen }));
}
if (!ctxConsole.some((text) => String(text).includes("hello-console"))) {
  throw new Error("context console " + JSON.stringify(ctxConsole));
}
page.off("console", handler);
page.removeAllListeners("console");
await page.clearConsoleMessages();
const after = await page.consoleMessages();
if (after.length !== 0) throw new Error("clearConsoleMessages left " + after.length);
await browser.close();
