import { chromium } from "playwright";
const sample = () => Deno.core.ops.op_engine_call("runtime.resourceUsage", {});
const baseline = await sample();
const browser = await chromium.launch();
const context = await browser.newContext();
const first = await context.newPage();
await first.setContent("<title>survivor</title><p>independent live page</p>");

const before = await sample();
for (let i = 0; i < 40; i++) {
  const page = await context.newPage();
  await page.setContent("<p>iteration " + i + "</p>");
  const locator = page.locator("p");
  await page.close();
  if (context.pages().length !== 1) throw new Error("closed Page retained by live context");
  try {
    await locator.textContent();
    throw new Error("closed locator accepted");
  } catch (error) {
    if (error.code !== "object_disposed") throw error;
  }
}
if (await first.title() !== "survivor") throw new Error("live page changed");
const after = await sample();
for (const key of ["pages", "contexts", "browsers"]) {
  if (after[key] !== before[key]) throw new Error("native retained historical " + key);
}
await context.close();
const closed = await sample();
if (closed.pages !== baseline.pages || closed.contexts !== baseline.contexts) throw new Error("native context cleanup incomplete");
await browser.close();
const final = await sample();
for (const key of ["pages", "contexts", "browsers"]) {
  if (final[key] !== baseline[key]) throw new Error("native browser cleanup incomplete: " + key);
}
console.log("RESOURCE_LIFECYCLE_OK");
