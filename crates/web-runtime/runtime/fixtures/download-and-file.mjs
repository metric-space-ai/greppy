import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
const payload = Uint8Array.from([0xff, 0xfe, 0x00, 0x80, 0x41]);
await page.route("**/data.bin", (route) =>
  route.fulfill({
    body: payload,
    contentType: "application/octet-stream",
    status: 200,
  }),
);
await page.goto(fixtureUrl);
await page.evaluate((url) => fetch(url), fixtureUrl + "data.bin");
const response = await page.waitForResponse("data.bin");
const raw = await response.body();
const view = raw instanceof ArrayBuffer ? new Uint8Array(raw) : new Uint8Array(raw);
if (
  view.length !== 5 ||
  view[0] !== 0xff ||
  view[1] !== 0xfe ||
  view[2] !== 0x00 ||
  view[3] !== 0x80 ||
  view[4] !== 0x41
) {
  throw new Error("binary body " + Array.from(view).join(","));
}
const asText = await response.text();
if (asText.includes("\uFFFD") === false && view[0] === 0xff) {
  // TextDecoder may produce replacement characters; it must not be the stored form.
}
const download = await page.waitForEvent("download");
if (!download || !String(download.url()).includes("data.bin")) {
  throw new Error("download url " + (download && download.url()));
}
if (download.suggestedFilename() !== "data.bin") {
  throw new Error("filename " + download.suggestedFilename());
}
if (download.page() !== page && download.page()._id !== page._id) {
  throw new Error("download.page is not the creating page");
}
const dest = "/tmp/greppy-web-dl-" + Date.now() + ".bin";
await download.saveAs(dest);
const saved = await download.path();
if (saved !== dest) throw new Error("path after saveAs " + saved);
if ((await download.failure()) !== null) throw new Error("failure");
await page.setContent("<!DOCTYPE html><html><body><input id='f' type='file'></body></html>");
await browser.close();