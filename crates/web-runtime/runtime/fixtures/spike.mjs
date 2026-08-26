import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.goto(fixtureUrl);
await page.getByRole("button", { name: "Load" }).click();
await page.getByLabel("Query").fill("greppy");
const result = await page.locator("main").innerText();
const value = await page.evaluate(() => document.title);
await browser.close();
