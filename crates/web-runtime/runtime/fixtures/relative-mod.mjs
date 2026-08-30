import { chromium } from "playwright";
import { marker } from "./relative-mod-helper.mjs";

if (marker() !== "relative-ok") {
  throw new Error("relative module marker " + marker());
}

const browser = await chromium.launch();
await browser.close();
