import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<label>Name<input id="n"></label>
<input id="q" placeholder="search">
<img alt="logo" title="brand" data-testid="hero" width="10" height="10">
<button id="b">Go</button>
<input type="checkbox" id="c">
<p id="t">idle</p>
</body></html>`);
if ((await page.getByPlaceholder("search").count()) !== 1) throw new Error("placeholder");
if ((await page.getByAltText("logo").count()) !== 1) throw new Error("alt");
if ((await page.getByTitle("brand").count()) !== 1) throw new Error("title");
if ((await page.getByTestId("hero").count()) !== 1) throw new Error("testid");
if ((await page.getByLabel("Name").count()) !== 1) throw new Error("label");
if ((await page.locator("body").getByAltText("logo").count()) !== 1) throw new Error("locator alt");
if ((await page.locator("body").getByTitle("brand").count()) !== 1) throw new Error("locator title");
if ((await page.locator("body").getByTestId("hero").count()) !== 1) throw new Error("locator testid");
if ((await page.locator("body").getByLabel("Name").count()) !== 1) throw new Error("locator label");
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
await page.mainFrame().addStyleTag({ content: "#t { font-weight: 700; }" });
await page.mainFrame().addScriptTag({ content: "window.__greppy_tag = 1;" });
await page.mainFrame().waitForLoadState();
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
if (!(await page.mainFrame().waitForFunction(() => window.__greppy === 1))) {
  throw new Error("frame waitForFunction");
}
const main = page.mainFrame();
await main.fill("#q", "frame-fill");
if ((await main.inputValue("#q")) !== "frame-fill") throw new Error("frame fill");
if ((await main.locator("#q").count()) !== 1) throw new Error("frame locator");
await main.click("#b");
await main.check("#c");
if (!(await page.isChecked("#c"))) throw new Error("frame check");
if (main.page() !== page && main.page()._id !== page._id) throw new Error("frame.page");
if (page.locator("#q").page() !== page && page.locator("#q").page()._id !== page._id) {
  throw new Error("locator.page");
}
if (main.isDetached()) throw new Error("main frame detached");
if ((await main.getByPlaceholder("search").count()) !== 1) throw new Error("frame placeholder");
if ((await main.getByLabel("Name").count()) !== 1) throw new Error("frame label");
if ((await main.getByAltText("logo").count()) !== 1) throw new Error("frame alt");
if ((await main.getByTitle("brand").count()) !== 1) throw new Error("frame title attr");
if ((await main.getByTestId("hero").count()) !== 1) throw new Error("frame testid");
if ((await main.innerHTML("#t")).trim() !== "dbl") throw new Error("frame innerHTML");
if ((await main.innerText("#t")).trim() !== "dbl") throw new Error("frame innerText");
if ((await main.textContent("#t")).trim() !== "dbl") throw new Error("frame textContent");
await main.uncheck("#c");
await page.locator("#c").setChecked(true);
if (!(await main.isChecked("#c"))) throw new Error("setChecked");
await main.type("#q", "!");
await page.locator("#q").pressSequentially("?");
if (!(await main.inputValue("#q")).includes("!")) throw new Error("frame type");
if ((await main.getAttribute("#t", "id")) !== "t") throw new Error("frame getAttribute");
if (!(await main.isEnabled("#q"))) throw new Error("frame isEnabled");
if (await main.isHidden("#t")) throw new Error("frame isHidden");
await page.locator("#q").focus();
await page.locator("#q").blur();
await main.focus("#q");
await main.dispatchEvent("#t", "click");
await main.setContent("<!DOCTYPE html><html><body><p id=fsc>frame-set</p></body></html>");
if ((await page.innerText("#fsc")).trim() !== "frame-set") throw new Error("frame setContent");
const frameHtml = await main.content();
if (!String(frameHtml).includes("frame-set")) throw new Error("frame content");
await browser.close();
