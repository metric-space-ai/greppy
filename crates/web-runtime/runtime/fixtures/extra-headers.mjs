import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setExtraHTTPHeaders({ "x-greppy-test": "yes" });
const nav = page.waitForNavigation();
await page.goto(fixtureUrl);
await nav;
await page.mainFrame().goto(fixtureUrl);
const text = (await page.locator("body").innerText()).trim();
if (text !== "HEADER_OK") {
  throw new Error("expected HEADER_OK from extra headers, got " + JSON.stringify(text));
}
const url = await page.waitForURL("http");
if (!String(url).includes("http")) throw new Error("waitForURL " + url);
const request = await page.waitForRequest("http");
if (!String(request.url()).includes("http")) throw new Error("waitForRequest");
if (request.method() !== "GET") throw new Error("method " + request.method());
if (request.resourceType() !== "document") {
  throw new Error("resourceType " + request.resourceType());
}
const headers = request.headers();
const headerBlob = JSON.stringify(headers).toLowerCase();
if (!headerBlob.includes("x-greppy-test")) {
  throw new Error("Request.headers missing extra header: " + JSON.stringify(headers));
}
let paused = false;
try {
  await page.pause();
  paused = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (paused) throw new Error("pause must not succeed as a no-op");
await browser.close();
