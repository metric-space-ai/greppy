import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><p>shot</p></body></html>");
const buffer = await page.screenshot();
if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) {
  throw new Error("screenshot expected PNG bytes, got " + typeof buffer);
}
const bytes = new Uint8Array(buffer);
const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
if (!png) {
  throw new Error("screenshot was not a PNG");
}
await browser.close();
