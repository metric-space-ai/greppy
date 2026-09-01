import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const value = await page.evaluate((n) => n + 1, 41);
if (value !== 42) {
  throw new Error("evaluate arg expected 42, got " + JSON.stringify(value));
}

const undef = await page.evaluate(() => undefined);
if (undef !== undefined) {
  throw new Error("evaluate undefined expected undefined, got " + JSON.stringify(undef));
}

const nan = await page.evaluate(() => NaN);
if (!Number.isNaN(nan)) {
  throw new Error("evaluate NaN expected NaN, got " + JSON.stringify(nan));
}

const inf = await page.evaluate(() => Infinity);
if (inf !== Infinity) {
  throw new Error("evaluate Infinity expected Infinity, got " + String(inf));
}

const ninf = await page.evaluate(() => -Infinity);
if (ninf !== -Infinity) {
  throw new Error("evaluate -Infinity expected -Infinity, got " + String(ninf));
}

const negZero = await page.evaluate(() => -0);
if (!Object.is(negZero, -0)) {
  throw new Error("evaluate -0 expected -0, got " + String(negZero));
}

const fromUndef = await page.evaluate((v) => v === undefined, undefined);
if (fromUndef !== true) {
  throw new Error("evaluate undefined arg expected true");
}

const iso = "2020-01-02T00:00:00.000Z";
const fromDate = await page.evaluate(
  (d) => d instanceof Date && d.toISOString(),
  new Date(iso),
);
if (fromDate !== iso) {
  throw new Error("evaluate Date arg expected " + iso + ", got " + fromDate);
}

const fromNanArg = await page.evaluate((n) => Number.isNaN(n), NaN);
if (fromNanArg !== true) {
  throw new Error("evaluate NaN arg expected true");
}

await browser.close();
