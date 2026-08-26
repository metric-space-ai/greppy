import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><head><title>T</title></head><body><p id='x'>ok</p></body></html>",
);
if ((await page.title()) !== "T") {
  throw new Error("title mismatch");
}
const html = await page.content();
if (!html.includes("<p id=\"x\">ok</p>") && !html.includes("<p id='x'>ok</p>")) {
  if (!html.includes("id=\"x\"") && !html.includes("id=x")) {
    throw new Error("content missing marker: " + html.slice(0, 200));
  }
}
await browser.close();
