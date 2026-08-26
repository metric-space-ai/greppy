import { chromium } from "playwright";

function toHex(bytes) {
  const view = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes);
  let out = "";
  for (let i = 0; i < view.length; i++) {
    out += (view[i] + 256).toString(16).slice(-2);
  }
  return out;
}

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
const payload = Uint8Array.from([0xff, 0xfe, 0x00, 0x80, 0x41]);
const expectedHex = "fffe008041";
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
if (toHex(raw) !== expectedHex) {
  throw new Error("Response.body hex " + toHex(raw) + " expected " + expectedHex);
}
const asText = await response.text();
const afterText = await response.body();
if (toHex(afterText) !== expectedHex) {
  throw new Error("Response.body changed after text(): " + toHex(afterText) + " text=" + JSON.stringify(asText));
}
const ctxDownloads = [];
page.context().on("download", (item) => {
  ctxDownloads.push(item);
});
const download = await page.waitForEvent("download");
if (!download || !String(download.url()).includes("data.bin")) {
  throw new Error("download url " + (download && download.url()));
}
if (!ctxDownloads.some((item) => String(item.url()).includes("data.bin"))) {
  throw new Error("context download missing");
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
let stream = false;
try {
  await download.createReadStream();
  stream = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (stream) throw new Error("Download.createReadStream must fail closed");
let deleted = false;
try {
  await download.delete();
  deleted = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (deleted) throw new Error("Download.delete must fail closed");
await page.setContent("<!DOCTYPE html><html><body><input id='f' type='file'></body></html>");
await browser.close();