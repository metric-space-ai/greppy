import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();
await page.setContent(`<!DOCTYPE html><html><body>
<input id="f" type="file">
<script>
window.changed = 0;
window.inputed = 0;
document.getElementById("f").addEventListener("change", function() { window.changed += 1; });
document.getElementById("f").addEventListener("input", function() { window.inputed += 1; });
</script>
</body></html>`);
await page.setInputFiles("#f", ["FILE_PATH"]);
const result = await page.locator("#f").setInputFiles(["FILE_PATH"]);
const got = await page.evaluate(() => {
  const el = document.getElementById("f");
  return {
    count: el && el.files ? el.files.length : -1,
    name: el && el.files && el.files[0] ? el.files[0].name : "",
    modified: el && el.files && el.files[0] ? el.files[0].lastModified : null,
    changed: window.changed,
    inputed: window.inputed,
  };
});
if (got.count !== 1 || got.name !== "sample.txt" || got.modified !== 1600000000000 || got.changed !== 2 || got.inputed !== 2) {
  throw new Error(
    "DOM FileList/change not populated (Servo blocker): " +
      JSON.stringify({ result, got }),
  );
}
await browser.close();
