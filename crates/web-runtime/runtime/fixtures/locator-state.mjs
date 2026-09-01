import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body>
  <input id="q" value="hello" data-k="v">
  <input type="checkbox" id="c" checked>
  <button id="go" disabled>Go</button>
  <p id="h" hidden>hidden</p>
  <div id="d"><span>inner</span></div>
  </body></html>`,
);
if ((await page.locator("#q").inputValue()) !== "hello") {
  throw new Error("inputValue");
}
if ((await page.locator("#q").getAttribute("data-k")) !== "v") {
  throw new Error("getAttribute");
}
if (!(await page.locator("#c").isChecked())) {
  throw new Error("isChecked");
}
if (!(await page.locator("#q").isEnabled()) || !(await page.locator("#go").isDisabled())) {
  throw new Error("enabled/disabled");
}
if (!(await page.locator("#h").isHidden())) {
  throw new Error("isHidden");
}
const html = await page.locator("#d").innerHTML();
if (!String(html).includes("inner")) {
  throw new Error("innerHTML " + html);
}
await page.locator("#q").focus();
if (await page.isClosed()) {
  throw new Error("page should be open");
}
await page.close();
if (!(await page.isClosed())) {
  throw new Error("page should be closed");
}
await browser.close();
