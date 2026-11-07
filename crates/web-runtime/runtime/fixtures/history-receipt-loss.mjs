import { chromium } from "playwright";

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.goto(fixtureUrl);
  const first = await page.url();
  await page.goto(fixtureUrl.replace(/\/?$/, "/two"));
  let timedOut = false;
  try {
    await page.goBack({ timeout: 1000 });
  } catch (error) {
    if (!String(error).includes("history traversal timed out")) throw error;
    timedOut = true;
  }
  if (!timedOut) {
    throw new Error("history traversal completed without a matching native receipt");
  }
  // A refused traversal does not publish its successful result into Page._url.
  // Read the real document independently of that controller-side cache.
  const actual = await page.evaluate(() => location.href);
  if (actual !== first) {
    throw new Error(`receipt-loss fixture did not actually traverse to its target: expected ${first}, document ${actual}, cached ${page.url()}`);
  }
} finally {
  await browser.close();
}
