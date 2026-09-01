import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body><iframe name="child" srcdoc="<p id='in'>frame-ok</p><button>Go</button><label>Name<input id='n'></label><input placeholder='hold'><img alt='pic'><span title='tip'>x</span><div data-testid='tid'>t</div>"></iframe></body></html>`,
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
const frames = await page.frames();
if (frames.length < 2) throw new Error("expected child frame, got " + frames.length);
const child = frames[1];
if ((await child.getByLabel("Name").count()) < 1) throw new Error("child Frame.getByLabel");
if ((await child.getByPlaceholder("hold").count()) < 1) throw new Error("child Frame.getByPlaceholder");
if ((await child.getByAltText("pic").count()) < 1) throw new Error("child Frame.getByAltText");
if ((await child.getByTitle("tip").count()) < 1) throw new Error("child Frame.getByTitle");
if ((await child.getByTestId("tid").count()) < 1) throw new Error("child Frame.getByTestId");
await browser.close();
