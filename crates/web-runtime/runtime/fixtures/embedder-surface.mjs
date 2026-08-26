import { chromium } from "playwright";

const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
page.on("dialog", (dialog) => dialog.accept());
await page.setContent(`<!DOCTYPE html><html><body>
<iframe name="child" srcdoc="<p id='in'>frame-ok</p>"></iframe>
<p>ok</p>
</body></html>`);
const frames = await page.frames();
if (frames.length < 1) {
  throw new Error("expected iframe, got " + frames.length);
}
const child = await page.frame({ name: "child" });
if (!child) {
  throw new Error("missing named frame");
}
await context.addCookies([{ name: "k", value: "v" }]);
await page.route("**/virtual-fulfill", (route) =>
  route.fulfill({ body: "<p>virtual</p>", contentType: "text/html" }),
);
await page.mainFrame().evaluate(() => document.title);
await browser.close();
