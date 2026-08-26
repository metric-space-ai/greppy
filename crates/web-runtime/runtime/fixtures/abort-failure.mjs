import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/aborted", (route) => route.abort());
let navigationFailed = false;
try {
  await page.goto(fixtureUrl + "aborted");
} catch (_error) {
  navigationFailed = true;
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
