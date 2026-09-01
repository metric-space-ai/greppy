let denied = false;
try {
  await import("file:///etc/passwd", { with: { type: "json" } });
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) {
  throw new Error("absolute json file import must be denied");
}

import { chromium } from "playwright";
const browser = await chromium.launch();
await browser.close();
