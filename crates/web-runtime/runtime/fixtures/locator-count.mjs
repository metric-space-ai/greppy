import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<ul class="todo-list">
<li>a</li>
<li>b</li>
<li style="display:none">c</li>
</ul>
</body></html>`);
const count = await page.locator("li").count();
if (count !== 3) {
  throw new Error("expected 3 list items, got " + count);
}
const visible = await page.locator(".todo-list li:visible").count();
if (visible !== 2) {
  throw new Error("expected 2 visible list items, got " + visible);
}
const hidden = await page.locator(".todo-list li:hidden").count();
if (hidden !== 1) {
  throw new Error("expected 1 hidden list item, got " + hidden);
}
if (!(await page.locator("li").nth(0).isVisible())) {
  throw new Error("first list item should be visible");
}
await browser.close();
