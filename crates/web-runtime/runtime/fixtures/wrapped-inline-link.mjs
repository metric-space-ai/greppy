import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await browser.newPage();
page.setDefaultTimeout(3_000);

// Inline links can span separate line boxes. Their bounding-box centre
// may hit the parent in the gap between lines, while every line is clickable.
await page.setContent(`<!DOCTYPE html><html><body style="margin:0">
<div style="width:300px;font:20px/48px sans-serif">
<a id="wrapped" href="#wrapped-done">Echo Dot (3rd Gen)<br>Smart Speaker with Alexa</a>
</div></body></html>`);
const wrappedGeometry = await page.evaluate(() => {
  const link = document.getElementById("wrapped");
  window.__wrappedClicks = 0;
  link.addEventListener("click", () => { window.__wrappedClicks += 1; });
  const bounds = link.getBoundingClientRect();
  const centre = document.elementFromPoint(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  return {
    centreHits: centre === link || link.contains(centre),
    lineHits: Array.from(link.getClientRects()).some(rect => {
      const top = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2);
      return top === link || link.contains(top);
    }),
  };
});
if (wrappedGeometry.centreHits || !wrappedGeometry.lineHits) {
  throw new Error("wrapped link fixture must expose the line-box gap: " + JSON.stringify(wrappedGeometry));
}
await page.locator("#wrapped").click();
if ((await page.evaluate(() => window.__wrappedClicks)) !== 1) {
  throw new Error("wrapped link must receive exactly one click");
}

await browser.close();
