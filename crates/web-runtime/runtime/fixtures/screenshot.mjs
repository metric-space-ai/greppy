import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<div id="red" style="width:80px;height:40px;background:#ff0000">red</div>
<div id="blue" style="width:80px;height:40px;background:#0000ff">blue</div>
</body></html>`);
const buffer = await page.screenshot();
if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) {
  throw new Error("screenshot expected PNG bytes, got " + typeof buffer);
}
const bytes = new Uint8Array(buffer);
const png = bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47;
if (!png) {
  throw new Error("screenshot was not a PNG");
}
function pngSize(buf) {
  const view = new DataView(buf);
  return { width: view.getUint32(16), height: view.getUint32(20) };
}
const full = pngSize(buffer);
const red = await page.locator("#red").screenshot();
const blue = await page.locator("#blue").screenshot();
if (!(red instanceof ArrayBuffer) || red.byteLength < 8) {
  throw new Error("locator.screenshot red");
}
const redBytes = new Uint8Array(red);
if (redBytes[0] !== 0x89 || redBytes[1] !== 0x50) throw new Error("red not png");
const redSize = pngSize(red);
const blueSize = pngSize(blue);
if (redSize.width >= full.width && redSize.height >= full.height) {
  throw new Error("locator clip should be smaller than page: " + JSON.stringify({ redSize, full }));
}
if (red.byteLength === blue.byteLength && new Uint8Array(red).every((b, i) => b === new Uint8Array(blue)[i])) {
  throw new Error("red and blue locator screenshots were identical");
}
await browser.close();
