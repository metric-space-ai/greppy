import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><input id='f' type='file'></body></html>");
await page.locator("#f").click();
const chooser = await page.waitForEvent("filechooser");
if (chooser.isMultiple()) throw new Error("expected single file chooser");
await chooser.setFiles("FILE_PATH");
const count = await page.evaluate(() => document.querySelector("#f").files.length);
if (count !== 1) {
  throw new Error("setFiles did not populate FileList, got " + count);
}
await browser.close();
