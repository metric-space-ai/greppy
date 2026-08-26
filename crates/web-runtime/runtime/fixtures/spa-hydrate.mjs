import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<div id="app">boot</div>
<script>
setTimeout(function() {
  document.getElementById("app").textContent = "hydrated";
}, 20);
</script>
</body></html>`);
await page.waitForFunction(() => document.getElementById("app").textContent === "hydrated");
const text = (await page.locator("#app").innerText()).trim();
if (text !== "hydrated") throw new Error("spa hydrate " + text);
await browser.close();