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
for (const key of ["pages", "contexts", "browsers", "delegates"]) {
  if (after[key] !== before[key]) throw new Error("native retained historical " + key);
}
// Close an opener before adopting its pending popup: the queue cannot own
// the opener, and the actual child delegate must be released with it.
for (let i = 0; i < 3; i++) {
  const opener = await context.newPage();
  await Deno.core.ops.op_engine_call("page.evaluate", {
    page: opener._id, source: "window.open('about:blank'); true",
  });
  await opener.close();
}
const pendingClosed = await sample();
if (pendingClosed.delegates !== before.delegates) throw new Error("pending popup retained native delegates");
await context.close();
const closed = await sample();
if (closed.pages !== baseline.pages || closed.contexts !== baseline.contexts) throw new Error("native context cleanup incomplete");
await browser.close();
const final = await sample();
for (const key of ["pages", "contexts", "browsers", "delegates"]) {
  if (final[key] !== baseline[key]) throw new Error("native browser cleanup incomplete: " + key);
}
console.log("RESOURCE_LIFECYCLE_OK");
