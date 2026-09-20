import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.route("**/same", (route) =>
  route.fulfill({ body: "ok", contentType: "text/plain", status: 200 }),
);
await page.goto(fixtureUrl + "same");
await page.unroute("**/same");
await page.route("**/same", (route) =>
  route.fulfill({ body: "missing", contentType: "text/plain", status: 404 }),
);
await page.goto(fixtureUrl + "same");

await page.route("**/failed", (route) => route.abort());
try {
  await page.goto(fixtureUrl + "failed");
} catch (_error) {
  // A failed transport is still a network record. Keep the page alive so the
  // caller can inspect it after this script returns.
}
