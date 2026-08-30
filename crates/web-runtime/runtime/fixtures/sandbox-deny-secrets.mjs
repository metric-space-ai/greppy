import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
let leaked = false;
try {
  await page.goto("file:///etc/passwd");
  const body = await page.content();
  if (String(body).includes("root:")) {
    leaked = true;
  }
} catch (_error) {
  // OS sandbox or engine policy must deny host secret files.
}
if (leaked) {
  throw new Error("worker sandbox allowed file:///etc/passwd");
}
await browser.close();
