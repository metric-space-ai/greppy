import { chromium } from "playwright";
const browser = await chromium.launch();
await browser.close();
