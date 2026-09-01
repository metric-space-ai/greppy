import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto(fixtureUrl);
const first = await page.url();
const secondUrl = fixtureUrl.replace(/\/?$/, "/two");
await page.goto(secondUrl);
const second = await page.url();
if (!String(second).includes("/two")) {
  throw new Error("second navigation url " + second);
}
if (second === first) {
  throw new Error("second navigation did not change url");
}
await page.goBack({ waitUntil: "load", timeout: 15_000 });
const back = await page.url();
if (back !== first) {
  throw new Error("goBack expected " + first + " got " + back);
}
await page.goForward({ timeout: 15_000 });
const forward = await page.url();
if (!String(forward).includes("/two")) {
  throw new Error("goForward expected /two got " + forward);
}
await browser.close();