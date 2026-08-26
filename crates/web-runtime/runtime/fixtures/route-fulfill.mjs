import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/intercepted", (route) =>
  route.fulfill({
    body: "<!DOCTYPE html><html><body><p id='x'>intercepted-ok</p></body></html>",
    contentType: "text/html",
    status: 200,
  }),
);
await page.goto(fixtureUrl + "intercepted");
const response = await page.waitForResponse("intercepted");
if (response.status() !== 200) throw new Error("status " + response.status());
if (!response.ok()) throw new Error("ok");
if (response.statusText() !== "OK") throw new Error("statusText " + response.statusText());
const body = await response.text();
if (!body.includes("intercepted-ok")) throw new Error("body " + body);
const raw = await response.body();
const rawView = raw instanceof ArrayBuffer ? new Uint8Array(raw) : new Uint8Array(raw);
const marker = "intercepted-ok";
let found = false;
for (let i = 0; i <= rawView.length - marker.length; i++) {
  let ok = true;
  for (let j = 0; j < marker.length; j++) {
    if (rawView[i + j] !== marker.charCodeAt(j)) {
      ok = false;
      break;
    }
  }
  if (ok) {
    found = true;
    break;
  }
}
if (!found) throw new Error("Response.body missing intercepted-ok bytes");
const headers = response.headers();
if (!JSON.stringify(headers).toLowerCase().includes("text/html")) {
  throw new Error("headers " + JSON.stringify(headers));
}
if (response.headerValue("content-type") !== "text/html") {
  throw new Error("headerValue " + response.headerValue("content-type"));
}
const associated = response.request();
if (!associated || !String(associated.url()).includes("intercepted")) {
  throw new Error("response.request");
}
const text = await page.locator("#x").innerText();
if (text.trim() !== "intercepted-ok") {
  throw new Error("expected intercepted-ok, got " + JSON.stringify(text));
}
const recorded = await page.requests();
if (!recorded.length) throw new Error("page.requests empty");
await page.unroute("**/intercepted");
await page.goto(fixtureUrl + "intercepted");
let still = false;
try {
  still = (await page.locator("#x").innerText()).trim() === "intercepted-ok";
} catch (error) {
  still = false;
}
if (still) throw new Error("unroute did not stop fulfill");
await page.route("**/all-clear", (route) =>
  route.fulfill({ body: "<p id='z'>routed</p>", contentType: "text/html" }),
);
await page.unrouteAll();
await page.goto(fixtureUrl + "all-clear");
let routed = false;
try {
  routed = (await page.locator("#z").innerText()).trim() === "routed";
} catch (error) {
  routed = false;
}
if (routed) throw new Error("unrouteAll did not clear routes");
if ((await page.opener()) !== null) throw new Error("opener");
await browser.close();
