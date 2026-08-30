let denied = false;
try {
  await import("../Cargo.toml");
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) {
  throw new Error("parent-directory import must be denied");
}

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
if (!denied) {
  throw new Error("absolute file import must be denied");
}

import { chromium } from "playwright";
const browser = await chromium.launch();
await browser.close();
