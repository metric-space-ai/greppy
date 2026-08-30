import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><input id='f' type='file'></body></html>",
);
await page.setInputFiles("#f", ["FILE_PATH"]);
const got = await page.evaluate(() => {
  const el = document.getElementById("f");
  return {
    count: el && el.files ? el.files.length : -1,
    name: el && el.files && el.files[0] ? el.files[0].name : "",
  };
});
if (got.count < 1 || got.name !== "sample.txt") {
  throw new Error("expected stored input files, got " + JSON.stringify(got));
}
await browser.close();
