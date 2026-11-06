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
  if (await page.url() !== first) {
    throw new Error("receipt-loss fixture did not actually traverse to its target");
  }
} finally {
  await browser.close();
}
