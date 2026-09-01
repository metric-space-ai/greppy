import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
let denied = false;
try {
  await page.goto("http://169.254.169.254/latest/meta-data/");
} catch (error) {
  denied = String(error.message).includes("policy_denied");
}
if (!denied) throw new Error("page.goto metadata must fail with policy_denied");
await browser.close();