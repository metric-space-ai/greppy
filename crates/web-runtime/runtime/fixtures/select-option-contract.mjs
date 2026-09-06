// Native browser regression: an unknown value must fail before changing the
// selected option or dispatching input/change. No label-to-value guessing.
import { chromium } from "playwright";

const browser = await chromium.launch();
try {
  const page = await browser.newPage();
  await page.setContent(`<!doctype html><html><body>
    <select id="order" aria-label="Order">
      <option value="descending" selected>High to low</option>
      <option value="ascending">Low to high</option>
      <option value="">No sorting</option>
    </select>
  </body></html>`);
  await page.evaluate(() => {
    window.selectionEvents = [];
    const select = document.getElementById("order");
    for (const name of ["input", "change"]) {
      select.addEventListener(name, () => window.selectionEvents.push(name));
    }
  });
  const snapshot = () => page.evaluate(() => {
    const select = document.getElementById("order");
    return {
      value: select.value,
      selected: Array.from(select.selectedOptions).map(option => option.value),
      events: window.selectionEvents.slice(),
    };
  });
  const before = await snapshot();
  // Engine transport does not promise object-key order. Compare all requested
  // fields explicitly, retaining the order of selections and event delivery.
  const stateKey = state => JSON.stringify([state.value, state.selected, state.events]);
  if (stateKey(before) !== stateKey({
    value: "descending", selected: ["descending"], events: [],
  })) {
    throw new Error(`invalid selection fixture precondition: ${JSON.stringify(before)}`);
  }
  for (const unknown of ["low", "Low to high"]) {
    let failure;
    try {
      await page.locator("#order").selectOption(unknown);
    } catch (error) {
      failure = String(error.message || error);
    }
    const after = await snapshot();
    if (!failure) {
      throw new Error(`unknown option returned success: ${JSON.stringify({unknown, before, after})}`);
    }
    if (stateKey(after) !== stateKey(before)) {
      throw new Error(`unknown option mutated selection/events: ${JSON.stringify({unknown, before, after})}`);
    }
    for (const detail of ["OPTION_NOT_FOUND", "ascending", "Low to high"]) {
      if (!failure.includes(detail)) {
        throw new Error(`missing actionable option detail ${detail}: ${failure}`);
      }
    }
  }

  await page.locator("#order").selectOption("ascending");
  const selected = await snapshot();
  if (stateKey(selected) !== stateKey({
    value: "ascending", selected: ["ascending"], events: ["input", "change"],
  })) {
    throw new Error(`valid selection did not update application state: ${JSON.stringify(selected)}`);
  }
  await page.locator("#order").selectOption("ascending");
  if (stateKey(await snapshot()) !== stateKey(selected)) {
    throw new Error("same-value no-op dispatched duplicate selection events");
  }
  await page.locator("#order").selectOption("");
  const empty = await snapshot();
  if (stateKey(empty) !== stateKey({
    value: "", selected: [""], events: ["input", "change", "input", "change"],
  })) {
    throw new Error(`existing empty-value option was confused with a missing value: ${JSON.stringify(empty)}`);
  }
} finally {
  await browser.close();
}
