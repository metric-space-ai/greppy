import { chromium } from "playwright";

async function expectUnsupported(label, fn) {
  let failed = false;
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      try {
        await result;
      } catch (error) {
        if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
        return;
      }
      throw new Error(label + " resolved instead of throwing");
    }
  } catch (error) {
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    failed = true;
  }
  if (!failed) throw new Error(label + " did not throw");
}

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setExtraHTTPHeaders({ "x-greppy-test": "yes" });
const finished = [];
const ctxRequests = [];
page.on("requestfinished", (req) => {
  finished.push(String(req.url()));
});
page.context().on("request", (req) => {
  ctxRequests.push(String(req.url()));
});
let loaded = 0;
let dcl = 0;
page.on("load", () => {
  loaded += 1;
});
page.on("domcontentloaded", () => {
  dcl += 1;
});
const loadWait = page.waitForEvent("load");
const nav = page.waitForNavigation();
await page.goto(fixtureUrl);
await nav;
await loadWait;
if (loaded < 1) throw new Error("Page load event " + loaded);
if (dcl < 1) throw new Error("Page domcontentloaded event " + dcl);
if (!finished.some((url) => String(url).includes("http"))) {
  throw new Error("requestfinished " + JSON.stringify(finished));
}
if (!ctxRequests.some((url) => String(url).includes("http"))) {
  throw new Error("context request " + JSON.stringify(ctxRequests));
}
const frameNav = page.mainFrame().waitForNavigation();
await page.mainFrame().goto(fixtureUrl);
await frameNav;
const text = (await page.locator("body").innerText()).trim();
if (text !== "HEADER_OK") {
  throw new Error("expected HEADER_OK from extra headers, got " + JSON.stringify(text));
}
const url = await page.waitForURL("http");
if (!String(url).includes("http")) throw new Error("waitForURL " + url);
const frameUrl = await page.mainFrame().waitForURL("http");
if (!String(frameUrl).includes("http")) throw new Error("frame waitForURL " + frameUrl);
const request = await page.waitForRequest("http");
if (!String(request.url()).includes("http")) throw new Error("waitForRequest");
const dumped = await page.requests();
if (!dumped.some((item) => String(item.url()).includes("http"))) {
  throw new Error("Page.requests missing navigation: " + JSON.stringify(dumped.map((item) => item.url())));
}
if (request.method() !== "GET") throw new Error("method " + request.method());
if (request.resourceType() !== "document") {
  throw new Error("resourceType " + request.resourceType());
}
const headers = request.headers();
const headerBlob = JSON.stringify(headers).toLowerCase();
if (!headerBlob.includes("x-greppy-test")) {
  throw new Error("Request.headers missing extra header: " + JSON.stringify(headers));
}
if (request.headerValue("x-greppy-test") !== "yes") {
  throw new Error("headerValue " + request.headerValue("x-greppy-test"));
}
const allHeaders = await request.allHeaders();
if (!JSON.stringify(allHeaders).toLowerCase().includes("x-greppy-test")) {
  throw new Error("allHeaders " + JSON.stringify(allHeaders));
}
const headerArray = request.headersArray();
if (!headerArray.some((h) => String(h.name).toLowerCase() === "x-greppy-test" && h.value === "yes")) {
  throw new Error("headersArray " + JSON.stringify(headerArray));
}
if (!request.isNavigationRequest()) throw new Error("isNavigationRequest");
if (request.failure() !== null) throw new Error("request.failure");
if (request.postData() !== null) throw new Error("GET postData");
if (request.postDataJSON() !== null) throw new Error("GET postDataJSON");
if (request.postDataBuffer() !== null) throw new Error("GET postDataBuffer");
await expectUnsupported("Request.existingResponse", () => request.existingResponse());
await expectUnsupported("Request.serviceWorker", () => request.serviceWorker());
const response = await request.response();
if (response) {
  await expectUnsupported("Response.finished", () => response.finished());
  await expectUnsupported("Response.frame", () => response.frame());
  await expectUnsupported("Response.fromServiceWorker", () => response.fromServiceWorker());
  await expectUnsupported("Response.httpVersion", () => response.httpVersion());
  await expectUnsupported("Response.securityDetails", () => response.securityDetails());
  await expectUnsupported("Response.serverAddr", () => response.serverAddr());
}
const ctx = await browser.newContext();
if (ctx.browser() !== browser && ctx.browser()._id !== browser._id) {
  throw new Error("context.browser");
}
ctx.setDefaultNavigationTimeout(5_000);
await ctx.route("**/ctx-route", (route) =>
  route.fulfill({
    body: "<!DOCTYPE html><html><body><p id=cr>ctx-ok</p></body></html>",
    contentType: "text/html",
  }),
);
const p2 = await ctx.newPage();
await p2.goto(fixtureUrl + "ctx-route");
if ((await p2.innerText("#cr")).trim() !== "ctx-ok") throw new Error("context.route");
await ctx.unrouteAll();
await p2.goto(fixtureUrl + "ctx-cleared");
let stillRouted = false;
try {
  stillRouted = (await p2.locator("#cr").innerText()).trim() === "ctx-ok";
} catch (error) {
  stillRouted = false;
}
if (stillRouted) throw new Error("context.unrouteAll");
await ctx.route("**/ctx-one", (route) =>
  route.fulfill({ body: "<p id=one>one</p>", contentType: "text/html" }),
);
await ctx.unroute("**/ctx-one");
await p2.goto(fixtureUrl + "ctx-one");
let one = false;
try {
  one = (await p2.locator("#one").innerText()).trim() === "one";
} catch (error) {
  one = false;
}
if (one) throw new Error("context.unroute");
if (ctx.isClosed()) throw new Error("context still open");
await ctx.close();
if (!ctx.isClosed()) throw new Error("context.isClosed");
const ctxHeaders = await browser.newContext();
await ctxHeaders.setExtraHTTPHeaders({ "x-greppy-test": "yes" });
const p3 = await ctxHeaders.newPage();
await p3.goto(fixtureUrl);
if ((await p3.locator("body").innerText()).trim() !== "HEADER_OK") {
  throw new Error("context extra headers");
}
await ctxHeaders.close();
const ctxLaunchHeaders = await browser.newContext({
  extraHTTPHeaders: { "x-greppy-test": "yes" },
});
const p4 = await ctxLaunchHeaders.newPage();
await p4.goto(fixtureUrl, { waitUntil: "load", timeout: 15_000 });
await p4.waitForLoadState("load", { timeout: 15_000 });
if ((await p4.locator("body").innerText()).trim() !== "HEADER_OK") {
  throw new Error("newContext extraHTTPHeaders");
}
await expectUnsupported("newContext viewport", () =>
  browser.newContext({ viewport: { width: 800, height: 600 } }),
);
await expectUnsupported("goto networkidle", () =>
  p4.goto(fixtureUrl, { waitUntil: "networkidle" }),
);
await expectUnsupported("goto referer", () =>
  p4.goto(fixtureUrl, { referer: "https://example.invalid/" }),
);
await ctxLaunchHeaders.close();
let paused = false;
try {
  await page.pause();
  paused = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (paused) throw new Error("pause must not succeed as a no-op");
await browser.close();
