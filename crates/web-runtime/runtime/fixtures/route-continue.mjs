import { chromium } from "playwright";

const httpsUrl = globalThis.httpsUrl || "";
if (!httpsUrl) {
  throw new Error("httpsUrl is required for Route.continue HTTPS");
}

async function expectUnsupported(label, fn) {
  let failed = false;
  try {
    const result = fn();
    if (result && typeof result.then === "function") {
      try {
        await result;
      } catch (error) {
        if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
        return;
      }
      throw new Error(label + " resolved instead of throwing");
    }
  } catch (error) {
    if (!String(error.message).includes("unsupported_playwright_operation")) throw error;
    failed = true;
  }
  if (!failed) throw new Error(label + " did not throw");
}

const browser = await chromium.launch();
const page = await browser.newPage();
const continueWithTestHeader = (route) =>
  route.continue({ headers: { "x-greppy-test": "yes" } });
await page.route("**/continued", continueWithTestHeader);
await page.route("**/sub.js", continueWithTestHeader);
await page.goto(fixtureUrl + "continued", { timeout: 15_000 });
if ((await page.locator("body").innerText()).trim() !== "HEADER_OK") {
  throw new Error("route.continue extra headers, got " + JSON.stringify(await page.locator("body").innerText()));
}
await page.waitForFunction(() => window.__greppySub === "ok", null, { timeout: 15_000 });
await expectUnsupported("Route.continue url", () =>
  page.route("**/nope-url", (route) => route.continue({ url: "https://example.invalid/" })),
);
await expectUnsupported("Route.continue method", () =>
  page.route("**/nope-method", (route) => route.continue({ method: "POST" })),
);
await expectUnsupported("Route.continue postData", () =>
  page.route("**/nope-body", (route) => route.continue({ postData: "x" })),
);
await page.goto(httpsUrl + "continued", { timeout: 15_000 });
if ((await page.locator("body").innerText()).trim().indexOf("HEADER_OK") < 0) {
  throw new Error("https route.continue extra headers");
}
await page.waitForFunction(() => window.__greppySub === "ok", null, { timeout: 15_000 });
await browser.close();
