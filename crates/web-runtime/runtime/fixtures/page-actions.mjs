import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  "<!DOCTYPE html><html><body><button id='b'>Go</button><input id='q'><input type='checkbox' id='c'><select id='s'><option value='a'>A</option><option value='b'>B</option></select><p id='t'>Hello</p></body></html>",
);
await page.click("#b");
await page.fill("#q", "ok");
await page.check("#c");
await page.selectOption("#s", "b");
if ((await page.inputValue("#q")) !== "ok") throw new Error("fill");
if (!(await page.isChecked("#c"))) throw new Error("check");
if ((await page.innerText("#t")) !== "Hello") throw new Error("innerText");
if (!(await page.isVisible("#t"))) throw new Error("visible");
await page.hover("#b");
await page.uncheck("#c");
if (await page.isChecked("#c")) throw new Error("uncheck");
await page.type("#q", "!");
if ((await page.inputValue("#q")) !== "ok!") throw new Error("type");
await browser.close();
