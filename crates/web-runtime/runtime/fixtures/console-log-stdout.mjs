import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage();
const out = { title: "Lokal", marker: "frame-channel-probe" };
console.log(JSON.stringify(out));
await browser.close();
