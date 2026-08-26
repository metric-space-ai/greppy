import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body><iframe name="child" srcdoc="<p id='in'>frame-ok</p><button>Go</button>"></iframe></body></html>`,
);
const text = (await page.frameLocator("iframe").locator("#in").innerText()).trim();
if (text !== "frame-ok") {
  throw new Error("frameLocator expected frame-ok, got " + JSON.stringify(text));
}
if ((await page.frameLocator("iframe").getByText("frame-ok").count()) < 1) {
  throw new Error("frame getByText");
}
if ((await page.frameLocator("iframe").getByRole("button").count()) < 1) {
  throw new Error("frame getByRole");
}
await browser.close();
