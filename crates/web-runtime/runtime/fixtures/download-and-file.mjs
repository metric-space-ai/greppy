import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await context.tracing.start();
await page.route("**/data.bin", (route) =>
  route.fulfill({
    body: "hello-bytes",
    contentType: "application/octet-stream",
  }),
);
await page.goto(fixtureUrl + "data.bin");
const trace = await context.tracing.stop();
if (!trace.downloads || trace.downloads.length < 1) {
  throw new Error("expected a download record, got " + JSON.stringify(trace));
}
await page.setContent("<!DOCTYPE html><html><body><input id='f' type='file'></body></html>");
await browser.close();
