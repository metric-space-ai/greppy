import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(
  `<!DOCTYPE html><html><body><iframe name="child" srcdoc="<p id='in'>frame-ok</p><button>Go</button><label>Name<input placeholder='search'></label>"></iframe></body></html>`,
);
const frames = await page.frames();
if (frames.length < 2) {
  throw new Error("expected main+iframe, got " + frames.length);
}
if (frames[0].parentFrame() !== null) {
  throw new Error("frames()[0] should be the main frame");
}
const child = await page.frame({ name: "child" });
if (!child) {
  throw new Error("missing named frame");
}
const inFrame = (await child.locator("#in").innerText()).trim();
if (inFrame !== "frame-ok") {
  throw new Error("child locator expected frame-ok, got " + JSON.stringify(inFrame));
}
if ((await child.getByText("frame-ok").count()) < 1) {
  throw new Error("child getByText");
}
if ((await child.getByRole("button").count()) < 1) {
  throw new Error("child getByRole");
}
await child.waitForSelector("#in");
const main = page.mainFrame();
if (main.isDetached()) throw new Error("main isDetached");
if (main.parentFrame() !== null) throw new Error("main parentFrame");
if (child.parentFrame() !== main && child.parentFrame()._id !== "main") {
  throw new Error("child parentFrame");
}
const kids = await main.childFrames();
if (kids.some((frame) => frame._isMain())) {
  throw new Error("childFrames included main");
}
if (!kids.some((frame) => frame.name() === "child")) {
  throw new Error("main childFrames missing child");
}
let nested = false;
try {
  await child.childFrames();
  nested = true;
} catch (error) {
  if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
}
if (nested) throw new Error("nested childFrames must fail closed");
const described = page.locator("iframe").describe("child-iframe");
if (!String(described.toString()).includes("child-iframe")) {
  throw new Error("locator.describe/toString " + described.toString());
}
if (described.description() !== "child-iframe") {
  throw new Error("locator.description " + described.description());
}
const viaContent = await page.locator("iframe").contentFrame().locator("#in").innerText();
if (viaContent.trim() !== "frame-ok") throw new Error("contentFrame " + viaContent);
const viaNested = await page.locator("body").frameLocator("iframe").locator("#in").innerText();
if (viaNested.trim() !== "frame-ok") throw new Error("locator.frameLocator " + viaNested);
if ((await page.frameLocator("iframe").getByLabel("Name").count()) !== 1) {
  throw new Error("FrameLocator.getByLabel");
}
if ((await page.frameLocator("iframe").getByPlaceholder("search").count()) !== 1) {
  throw new Error("FrameLocator.getByPlaceholder");
}
const mainNested = await main.frameLocator("iframe").locator("#in").innerText();
if (mainNested.trim() !== "frame-ok") throw new Error("main frameLocator " + mainNested);

await child.setContent(
  "<!DOCTYPE html><html><body><p id='in'>rewritten</p></body></html>",
);
if ((await child.locator("#in").innerText()).trim() !== "rewritten") {
  throw new Error("child setContent " + (await child.locator("#in").innerText()));
}
await child.addStyleTag({ content: "p { font-weight: 700; }" });
await child.addScriptTag({ content: "window.__child_tag = 7;" });
if ((await child.evaluate(() => window.__child_tag)) !== 7) {
  throw new Error("child addScriptTag");
}
await child.waitForLoadState();
await child.waitForTimeout(10);
if (!(await child.waitForFunction(() => document.getElementById("in")))) {
  throw new Error("child waitForFunction");
}
await child.goto("about:blank");
const blank = await child.waitForURL("about:blank");
if (!String(blank).includes("about:blank")) {
  throw new Error("child goto/waitForURL " + blank);
}

function expectUnsupported(label, fn) {
  let failed = false;
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      return result.then(
        () => {
          throw new Error(label + " resolved instead of throwing");
        },
        (error) => {
          if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
        },
      );
    }
  } catch (error) {
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    failed = true;
  }
  if (!failed) throw new Error(label + " did not throw");
}

expectUnsupported("child isDetached", () => child.isDetached());
expectUnsupported("child getByPlaceholder", () => child.getByPlaceholder("search"));
expectUnsupported("child getByAltText", () => child.getByAltText("logo"));
expectUnsupported("child getByTitle", () => child.getByTitle("brand"));
expectUnsupported("child getByTestId", () => child.getByTestId("hero"));
expectUnsupported("child getByLabel", () => child.getByLabel("Name"));
expectUnsupported("child frameLocator", () => child.frameLocator("iframe"));
await browser.close();
