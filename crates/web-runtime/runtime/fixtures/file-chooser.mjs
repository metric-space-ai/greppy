import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await context.tracing.start();
await page.setContent(
  "<!DOCTYPE html><html><body><input id='f' type='file'></body></html>",
);
await page.setInputFiles("#f", ["FILE_PATH"]);
const trace = await context.tracing.stop();
if (!trace.file_paths || trace.file_paths.length < 1) {
  throw new Error("expected stored file paths, got " + JSON.stringify(trace));
}
await browser.close();
