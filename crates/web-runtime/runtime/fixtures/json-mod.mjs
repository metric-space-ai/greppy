import { chromium } from "playwright";
import data from "./json-mod-data.json" with { type: "json" };

if (data.marker !== "json-ok") {
  throw new Error("json module marker " + data.marker);
}

const browser = await chromium.launch();
await browser.close();
