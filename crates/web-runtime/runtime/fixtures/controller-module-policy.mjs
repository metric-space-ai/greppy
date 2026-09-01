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

denied = false;
try {
  await import("node:child_process");
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) throw new Error("node:child_process import must be denied");

if (typeof process === "undefined" || !process.env) {
  throw new Error("process.env allow-list must exist");
}
if (process.env.NODE_ENV !== "production") {
  throw new Error("process.env.NODE_ENV must be production, got " + process.env.NODE_ENV);
}
if (process.env.PATH || process.env.HOME || process.env.GREPPY_RUN_ID) {
  throw new Error("process.env leaked parent environment");
}

const browser = await chromium.launch();
await browser.close();
