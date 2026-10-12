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
// DataTransfer must retain the File's timestamp, including times before 1970.
// This isolates metadata loss in the DOM from native path staging.
const transferTimes = await page.evaluate(() => [1600000000000, -1000].map(modified => {
  const file = new File(["contents"], "metadata.txt", {lastModified: modified});
  const transfer = new DataTransfer();
  transfer.items.add(file);
  return [transfer.items[0].getAsFile().lastModified, transfer.files[0].lastModified];
}));
if (transferTimes.length !== 2 || transferTimes[0][0] !== 1600000000000 ||
    transferTimes[0][1] !== 1600000000000 || transferTimes[1][0] !== -1000 ||
    transferTimes[1][1] !== -1000) {
  throw new Error("DataTransfer lost File.lastModified: " + JSON.stringify(transferTimes));
}
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
