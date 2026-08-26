import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<input id="q" placeholder="search">
<img alt="logo" title="brand" data-testid="hero" width="10" height="10">
<button id="b">Go</button>
<p id="t">idle</p>
</body></html>`);
if ((await page.getByPlaceholder("search").count()) !== 1) throw new Error("placeholder");
if ((await page.getByAltText("logo").count()) !== 1) throw new Error("alt");
if ((await page.getByTitle("brand").count()) !== 1) throw new Error("title");
if ((await page.getByTestId("hero").count()) !== 1) throw new Error("testid");
if (!(await page.locator("#q").isEditable())) throw new Error("editable");
await page.fill("#q", "x");
await page.locator("#q").clear();
if ((await page.inputValue("#q")) !== "") throw new Error("clear");
await page.locator("#b").evaluate((el) => {
  el.addEventListener("dblclick", () => {
    document.getElementById("t").textContent = "dbl";
  });
});
await page.dblclick("#b");
if ((await page.innerText("#t")).trim() !== "dbl") throw new Error("dblclick");
await page.dispatchEvent("#t", "click");
await page.addStyleTag({ content: "#t { font-weight: 700; }" });
await page.addScriptTag({ content: "window.__greppy_tag = 1;" });
const tags = await page.evaluate(() => ({
  styles: document.querySelectorAll("style").length,
  scripts: document.querySelectorAll("script").length,
}));
if (tags.styles < 1 || tags.scripts < 1) throw new Error("tags " + JSON.stringify(tags));
await page.evaluate(() => {
  window.__greppy = 1;
});
const injected = await page.waitForFunction(() => window.__greppy === 1);
if (!injected) throw new Error("waitForFunction");
await browser.close();
