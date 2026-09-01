import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
const ctxDialogs = [];
const liveDialogs = [];
page.context().on("dialog", (dialog) => {
  ctxDialogs.push(dialog.message());
});
page.on("dialog", (dialog) => {
  if (dialog.message()) liveDialogs.push(dialog.message());
  return dialog.accept();
});

const alertValue = await page.evaluate(() => {
  alert("native-hi");
  return 42;
});
if (alertValue !== 42) {
  throw new Error("expected 42 after native alert, got " + JSON.stringify(alertValue));
}
const alertDialog = await page.waitForEvent("dialog");
if (alertDialog.type() !== "alert" || alertDialog.message() !== "native-hi") {
  throw new Error(
    "alert dialog " +
      JSON.stringify({ type: alertDialog.type(), message: alertDialog.message() }),
  );
}
if (alertDialog.page() !== page && alertDialog.page()._id !== page._id) {
  throw new Error("dialog.page");
}
if (!ctxDialogs.includes("native-hi")) {
  throw new Error("context dialog " + JSON.stringify(ctxDialogs));
}
if (!liveDialogs.includes("native-hi")) {
  throw new Error("page.on live dialog " + JSON.stringify(liveDialogs));
}

const confirmed = await page.evaluate(() => confirm("native-confirm"));
if (confirmed !== true) {
  throw new Error("confirm accept expected true, got " + JSON.stringify(confirmed));
}
const confirmDialog = await page.waitForEvent("dialog");
if (confirmDialog.type() !== "confirm" || confirmDialog.message() !== "native-confirm") {
  throw new Error(
    "confirm dialog " +
      JSON.stringify({ type: confirmDialog.type(), message: confirmDialog.message() }),
  );
}

page.on("dialog", (dialog) => dialog.accept("typed-value"));
const prompted = await page.evaluate(() => prompt("native-prompt", "default"));
if (prompted !== "typed-value") {
  throw new Error("prompt expected typed-value, got " + JSON.stringify(prompted));
}
const promptDialog = await page.waitForEvent("dialog");
if (promptDialog.type() !== "prompt" || promptDialog.message() !== "native-prompt") {
  throw new Error(
    "prompt dialog " +
      JSON.stringify({ type: promptDialog.type(), message: promptDialog.message() }),
  );
}

page.on("dialog", (dialog) => dialog.dismiss());
const cancelled = await page.evaluate(() => confirm("native-cancel"));
if (cancelled !== false) {
  throw new Error("confirm dismiss expected false, got " + JSON.stringify(cancelled));
}

await browser.close();
