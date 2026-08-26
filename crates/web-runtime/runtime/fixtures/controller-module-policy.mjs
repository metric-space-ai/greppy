import { chromium } from "playwright";

let denied = false;
try {
  await import("node:fs");
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) throw new Error("node:fs import must be denied");

denied = false;
try {
  await import("file:///etc/passwd");
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) throw new Error("file URL import must be denied");

const browser = await chromium.launch();
await browser.close();
