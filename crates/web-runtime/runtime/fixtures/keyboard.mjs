import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent("<!DOCTYPE html><html><body><input id='q'></body></html>");
await page.locator("#q").click();
await page.keyboard.type("hi");
const value = await page.evaluate(() => document.querySelector("#q").value);
if (value !== "hi") {
  throw new Error("keyboard.type expected hi, got " + JSON.stringify(value));
}
await page.evaluate(() => {
  window.__input = [];
  window.__keysDuringInsert = [];
  const q = document.getElementById("q");
  q.addEventListener("input", (event) => window.__input.push(event.inputType || "input"));
  q.addEventListener("keydown", (event) => window.__keysDuringInsert.push(event.key));
});
await page.keyboard.insertText("!");
const inserted = await page.evaluate(() => document.querySelector("#q").value);
if (inserted !== "hi!") {
  throw new Error("insertText expected hi!, got " + JSON.stringify(inserted));
}
const insertEvents = await page.evaluate(() => ({ input: window.__input, keys: window.__keysDuringInsert }));
if (!insertEvents.input.length) {
  throw new Error("insertText must fire input: " + JSON.stringify(insertEvents));
}
if (insertEvents.keys.length) {
  throw new Error("insertText must not fire keydown: " + JSON.stringify(insertEvents));
}
await page.evaluate(() => {
  window.__keys = [];
  document.getElementById("q").addEventListener("keydown", (event) => {
    window.__keys.push(event.key);
  });
});
await page.keyboard.press("Enter");
const keys = await page.evaluate(() => window.__keys);
if (!Array.isArray(keys) || !keys.includes("Enter")) {
  throw new Error("keyboard.press Enter, got " + JSON.stringify(keys));
}
await page.evaluate(() => {
  window.__down = [];
  window.__up = [];
  const q = document.getElementById("q");
  q.addEventListener("keydown", (event) => window.__down.push(event.key));
  q.addEventListener("keyup", (event) => window.__up.push(event.key));
});
await page.keyboard.down("Shift");
const afterDown = await page.evaluate(() => ({ down: window.__down, up: window.__up }));
if (!afterDown.down.includes("Shift") || afterDown.up.includes("Shift")) {
  throw new Error("keyboard.down should be keydown-only: " + JSON.stringify(afterDown));
}
await page.keyboard.up("Shift");
const afterUp = await page.evaluate(() => ({ down: window.__down, up: window.__up }));
if (!afterUp.up.includes("Shift")) {
  throw new Error("keyboard.up should dispatch keyup: " + JSON.stringify(afterUp));
}
await page.close();
await browser.close();