import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";

const pw = process.env.PLAYWRIGHT_PACKAGE;
if (!pw) {
  throw new Error("PLAYWRIGHT_PACKAGE is required");
}
const mod = await import(pathToFileURL(`${pw}/index.mjs`).href);
const { chromium } = mod;

const executablePath =
  process.env.GREPPY_ORACLE_CHROMIUM ||
  `${process.env.HOME}/Library/Caches/ms-playwright/chromium-1234/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing`;

const out = process.argv[2] || "oracle-reference.json";
const browser = await chromium.launch({ executablePath, headless: true });

const setPage = await browser.newPage();
await setPage.setContent(
  "<!DOCTYPE html><html><head><title>Oracle</title></head><body><p id='x'>ok</p></body></html>",
);
const title = await setPage.title();
const value = await setPage.evaluate(() => 1 + 1);
const text = (await setPage.locator("#x").innerText()).trim();
const content = await setPage.content();
const count = await setPage.locator("#x").count();
const innerHTML = (await setPage.locator("#x").innerHTML()).trim();
const pageInnerHTML = (await setPage.innerHTML("#x")).trim();
const textContent = (await setPage.locator("#x").textContent()).trim();
const pageTextContent = (await setPage.textContent("#x")).trim();
const visible = await setPage.locator("#x").isVisible();
const pageVisible = await setPage.isVisible("#x");
const attr = await setPage.locator("#x").getAttribute("id");
await setPage.close();

const dialogPage = await browser.newPage();
let dialogMessage = null;
let dialogType = null;
dialogPage.on("dialog", async (dialog) => {
  dialogType = dialog.type();
  dialogMessage = dialog.message();
  await dialog.accept();
});
const dialogValue = await dialogPage.evaluate(() => {
  alert("native-hi");
  return 42;
});
await dialogPage.close();

const fillPage = await browser.newPage();
await fillPage.setContent(
  "<!DOCTYPE html><html><body><input id='q'><p id='out'></p></body></html>",
);
await fillPage.locator("#q").fill("ok");
const filled = await fillPage.evaluate(() => document.querySelector("#q").value);
await fillPage.close();

const consolePage = await browser.newPage();
let consoleText = null;
let consoleType = null;
consolePage.on("console", (msg) => {
  if (consoleText == null) {
    consoleType = msg.type();
    consoleText = msg.text();
  }
});
await consolePage.evaluate(() => {
  console.log("hello-console");
});
await consolePage.close();

await browser.close();
const receipt = {
  engine: "playwright@1.62.1+chromium-1234",
  browserVersion: "151.0.7922.34",
  title,
  value,
  text,
  cases: {
    setContent: { title, value, text },
    content: {
      includesOk: String(content).includes("ok"),
      includesOracle: /oracle/i.test(String(content)),
      count,
      innerHTML,
      pageInnerHTML,
      textContent,
      pageTextContent,
      visible,
      pageVisible,
      attr,
    },
    dialog: { value: dialogValue, type: dialogType, message: dialogMessage },
    fill: { value: filled },
    console: { type: consoleType, text: consoleText },
  },
};
writeFileSync(out, JSON.stringify(receipt, null, 2) + "\n");
console.log(JSON.stringify(receipt));
