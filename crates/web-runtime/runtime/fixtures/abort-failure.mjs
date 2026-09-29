import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/aborted", (route) => route.abort());
const failed = [];
page.on("requestfailed", (req) => {
  failed.push(String(req.url()));
});
const ctxFailed = [];
page.context().on("requestfailed", (req) => {
  ctxFailed.push(String(req.url()));
});
const waitFailed = page.waitForEvent("requestfailed");
let navigationFailed = false;
const navigationStarted = Date.now();
try {
  await page.goto(fixtureUrl + "aborted");
} catch (_error) {
  navigationFailed = true;
}
const navigationElapsed = Date.now() - navigationStarted;
if (navigationElapsed >= 10_000) {
  throw new Error("known aborted navigation took " + navigationElapsed + "ms");
}
const failedReq = await waitFailed;
if (!failed.some((url) => url.includes("aborted"))) {
  throw new Error("requestfailed " + JSON.stringify(failed));
}
if (!ctxFailed.some((url) => url.includes("aborted"))) {
  throw new Error("context requestfailed " + JSON.stringify(ctxFailed));
}
if (!String(failedReq.url()).includes("aborted")) {
  throw new Error("waitForEvent requestfailed " + failedReq.url());
}
const request = await page.waitForRequest("aborted");
const failure = request.failure();
if (!failure || !String(failure.errorText || "").includes("ERR_FAILED")) {
  throw new Error(
    "abort should set Request.failure, got " +
      JSON.stringify(failure) +
      " url=" +
      request.url() +
      " navFailed=" +
      navigationFailed
  );
}
if (!navigationFailed) {
  throw new Error("aborted main-frame goto must reject");
}
await browser.close();
