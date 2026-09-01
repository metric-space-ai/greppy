import { chromium } from "playwright";

const browser = await chromium.launch();
const page = await browser.newPage();

function assertEqual(label, got, expected) {
  const g = JSON.stringify(got);
  const e = JSON.stringify(expected);
  if (g !== e) {
    throw new Error(label + " expected " + e + ", got " + g);
  }
}

function assertFields(label, actual, expected) {
  if (!actual) {
    throw new Error(label + " missing event, expected " + JSON.stringify(expected));
  }
  for (const key of Object.keys(expected)) {
    if (actual[key] !== expected[key]) {
      throw new Error(
        label +
          "." +
          key +
          " expected " +
          JSON.stringify(expected[key]) +
          ", got " +
          JSON.stringify(actual[key]) +
          " in " +
          JSON.stringify(actual),
      );
    }
  }
}

async function openField(tag) {
  const html =
    tag === "textarea"
      ? "<!DOCTYPE html><html><body><textarea id='q'></textarea></body></html>"
      : "<!DOCTYPE html><html><body><input id='q'></body></html>";
  await page.setContent(html);
}

async function isolate(value, setup) {
  await page.evaluate(
    ({ value, setup }) => {
      window.__greppyKb = {
        mods: { Shift: false, Control: false, Alt: false, Meta: false },
        pressed: Object.create(null),
      };
      const q = document.getElementById("q");
      q.value = value;
      q.focus();
      try {
        const n = String(value).length;
        q.setSelectionRange(n, n);
      } catch (_err) {}
      window.__kblog = [];
      if (setup === "keydownPrevent") {
        q.addEventListener("keydown", (event) => event.preventDefault());
      }
      if (setup === "beforeinputPrevent") {
        q.addEventListener("beforeinput", (event) => event.preventDefault());
      }
      ["keydown", "keypress", "beforeinput", "textInput", "input", "keyup"].forEach((name) => {
        q.addEventListener(
          name,
          (event) => {
            window.__kblog.push({
              type: event.type,
              key: event.key,
              code: event.code,
              location: event.location,
              shiftKey: event.shiftKey,
              cancelable: event.cancelable,
              inputType: event.inputType,
              data: event.data,
              value: q.value,
            });
          },
          true,
        );
      });
    },
    { value, setup: setup || "" },
  );
}

async function kblog() {
  return page.evaluate(() => ({
    value: document.getElementById("q").value,
    events: window.__kblog.slice(),
    types: window.__kblog.map((event) => event.type),
  }));
}

await openField("input");
await isolate("");
await page.keyboard.type("hi");
const typed = await kblog();
if (typed.value !== "hi") {
  throw new Error("keyboard.type expected hi, got " + JSON.stringify(typed));
}
const typedKeys = typed.events.filter((event) => event.type === "keydown").map((event) => event.key);
if (!typedKeys.includes("h") || !typedKeys.includes("i")) {
  throw new Error("keyboard.type must fire keydown per character: " + JSON.stringify(typed));
}
if (!typed.types.includes("input")) {
  throw new Error("keyboard.type must fire input: " + JSON.stringify(typed));
}

await openField("input");
await isolate("hi");
await page.keyboard.insertText("!");
const inserted = await kblog();
if (inserted.value !== "hi!") {
  throw new Error("insertText expected hi!, got " + JSON.stringify(inserted));
}
if (!inserted.types.includes("input")) {
  throw new Error("insertText must fire input: " + JSON.stringify(inserted));
}
if (inserted.types.includes("keydown")) {
  throw new Error("insertText must not fire keydown: " + JSON.stringify(inserted));
}

await openField("input");
await isolate("");
await page.keyboard.down("Shift");
const afterDown = await kblog();
if (!afterDown.types.includes("keydown") || afterDown.types.includes("keyup")) {
  throw new Error("keyboard.down should be keydown-only: " + JSON.stringify(afterDown));
}
await page.keyboard.up("Shift");
const afterUp = await kblog();
if (!afterUp.types.includes("keyup")) {
  throw new Error("keyboard.up should dispatch keyup: " + JSON.stringify(afterUp));
}

await openField("input");
await isolate("");
await page.keyboard.press("Enter");
const enterInput = await kblog();
const enterKeys = enterInput.events.filter((event) => event.type === "keydown").map((event) => event.key);
if (!enterKeys.includes("Enter")) {
  throw new Error("keyboard.press Enter, got " + JSON.stringify(enterInput));
}

await openField("input");
await isolate("");
await page.keyboard.type("x");
const typedX = await kblog();
assertEqual("type x value", typedX.value, "x");
assertEqual("type x types", typedX.types, [
  "keydown",
  "keypress",
  "beforeinput",
  "textInput",
  "input",
  "keyup",
]);
assertFields("type x keydown", typedX.events[0], {
  type: "keydown",
  key: "x",
  code: "KeyX",
  cancelable: true,
});
assertFields("type x keypress", typedX.events[1], { type: "keypress", cancelable: true });
assertFields("type x beforeinput", typedX.events[2], {
  type: "beforeinput",
  cancelable: true,
  inputType: "insertText",
  data: "x",
});
assertFields("type x textInput", typedX.events[3], {
  type: "textInput",
  cancelable: true,
  data: "x",
});
assertFields("type x input", typedX.events[4], {
  type: "input",
  cancelable: false,
  inputType: "insertText",
  data: "x",
  value: "x",
});
assertFields("type x keyup", typedX.events[5], { type: "keyup", cancelable: true });

await openField("input");
await isolate("", "keydownPrevent");
await page.keyboard.type("z");
const kdPrevent = await kblog();
assertEqual("type keydown.preventDefault value", kdPrevent.value, "");
assertEqual("type keydown.preventDefault types", kdPrevent.types, ["keydown", "keyup"]);

await openField("input");
await isolate("", "beforeinputPrevent");
await page.keyboard.type("z");
const biPrevent = await kblog();
assertEqual("type beforeinput.preventDefault value", biPrevent.value, "");
assertEqual("type beforeinput.preventDefault types", biPrevent.types, [
  "keydown",
  "keypress",
  "beforeinput",
  "keyup",
]);

await openField("input");
await isolate("hi");
await page.keyboard.insertText("!");
const insertOracle = await kblog();
assertEqual("insertText value", insertOracle.value, "hi!");
assertEqual("insertText types", insertOracle.types, ["beforeinput", "textInput", "input"]);
assertFields("insertText beforeinput", insertOracle.events[0], {
  type: "beforeinput",
  cancelable: true,
  inputType: "insertText",
  data: "!",
});
assertFields("insertText textInput", insertOracle.events[1], {
  type: "textInput",
  cancelable: true,
  data: "!",
});
assertFields("insertText input", insertOracle.events[2], {
  type: "input",
  cancelable: false,
  inputType: "insertText",
  data: "!",
  value: "hi!",
});

await openField("input");
await isolate("hi", "keydownPrevent");
await page.keyboard.insertText("!");
const insertIgnoreKeydown = await kblog();
assertEqual("insertText ignores keydown.preventDefault value", insertIgnoreKeydown.value, "hi!");
assertEqual("insertText ignores keydown.preventDefault types", insertIgnoreKeydown.types, [
  "beforeinput",
  "textInput",
  "input",
]);
assertFields("insertText+keydownPrevent beforeinput", insertIgnoreKeydown.events[0], {
  type: "beforeinput",
  cancelable: true,
  inputType: "insertText",
  data: "!",
});
assertFields("insertText+keydownPrevent textInput", insertIgnoreKeydown.events[1], {
  type: "textInput",
  cancelable: true,
  data: "!",
});
assertFields("insertText+keydownPrevent input", insertIgnoreKeydown.events[2], {
  type: "input",
  cancelable: false,
  inputType: "insertText",
  data: "!",
  value: "hi!",
});

await openField("input");
await isolate("hi", "beforeinputPrevent");
await page.keyboard.insertText("!");
const insertBiPrevent = await kblog();
assertEqual("insertText beforeinput.preventDefault value", insertBiPrevent.value, "hi");
assertEqual("insertText beforeinput.preventDefault types", insertBiPrevent.types, ["beforeinput"]);

await openField("textarea");
await isolate("");
await page.keyboard.press("Enter");
const enterArea = await kblog();
assertEqual("press Enter textarea value", enterArea.value, "\n");
assertEqual("press Enter textarea types", enterArea.types, [
  "keydown",
  "keypress",
  "beforeinput",
  "textInput",
  "input",
  "keyup",
]);
assertFields("press Enter beforeinput", enterArea.events[2], {
  type: "beforeinput",
  cancelable: true,
  inputType: "insertLineBreak",
});
assertFields("press Enter textInput", enterArea.events[3], {
  type: "textInput",
  cancelable: true,
  data: "\n",
});
assertFields("press Enter input", enterArea.events[4], {
  type: "input",
  cancelable: false,
  inputType: "insertLineBreak",
  value: "\n",
});

await openField("input");
await isolate("");
await page.keyboard.press("Escape");
const esc = await kblog();
assertEqual("press Escape types", esc.types, ["keydown", "keyup"]);
assertFields("press Escape keydown", esc.events[0], {
  type: "keydown",
  key: "Escape",
  code: "Escape",
  cancelable: true,
});
assertFields("press Escape keyup", esc.events[1], {
  type: "keyup",
  key: "Escape",
  cancelable: true,
});

await openField("input");
await isolate("");
await page.keyboard.press("Shift");
const shift = await kblog();
assertEqual("press Shift types", shift.types, ["keydown", "keyup"]);
assertFields("press Shift keydown", shift.events[0], {
  type: "keydown",
  key: "Shift",
  code: "ShiftLeft",
  location: 1,
  shiftKey: true,
  cancelable: true,
});
assertFields("press Shift keyup", shift.events[1], {
  type: "keyup",
  key: "Shift",
  shiftKey: false,
  cancelable: true,
});

await openField("input");
await isolate("");
await page.keyboard.press("Shift+KeyA");
const shifted = await kblog();
assertEqual("press Shift+KeyA value", shifted.value, "A");
assertEqual("press Shift+KeyA types", shifted.types, [
  "keydown",
  "keydown",
  "keypress",
  "beforeinput",
  "textInput",
  "input",
  "keyup",
  "keyup",
]);
assertFields("Shift+KeyA Shift down", shifted.events[0], {
  type: "keydown",
  key: "Shift",
  code: "ShiftLeft",
  location: 1,
  shiftKey: true,
});
assertFields("Shift+KeyA A down", shifted.events[1], {
  type: "keydown",
  key: "A",
  code: "KeyA",
  shiftKey: true,
  cancelable: true,
});
assertFields("Shift+KeyA keypress", shifted.events[2], { type: "keypress", key: "A", shiftKey: true });
assertFields("Shift+KeyA beforeinput", shifted.events[3], {
  type: "beforeinput",
  cancelable: true,
  inputType: "insertText",
  data: "A",
});
assertFields("Shift+KeyA textInput", shifted.events[4], {
  type: "textInput",
  cancelable: true,
  data: "A",
});
assertFields("Shift+KeyA input", shifted.events[5], {
  type: "input",
  cancelable: false,
  inputType: "insertText",
  data: "A",
  value: "A",
});
assertFields("Shift+KeyA A up", shifted.events[6], { type: "keyup", key: "A", shiftKey: true });
assertFields("Shift+KeyA Shift up", shifted.events[7], {
  type: "keyup",
  key: "Shift",
  shiftKey: false,
});

await openField("input");
await isolate("");
await page.keyboard.type("A");
const typeA = await kblog();
assertEqual("type A value", typeA.value, "A");
assertFields("type A keydown", typeA.events[0], {
  type: "keydown",
  key: "A",
  code: "KeyA",
  shiftKey: false,
  cancelable: true,
});

await openField("input");
await isolate("ab");
await page.keyboard.press("Backspace");
const backspaced = await kblog();
assertEqual("press Backspace value", backspaced.value, "a");

let unknown = "";
try {
  await page.keyboard.press("NotARealKey");
} catch (error) {
  unknown = String(error && error.message ? error.message : error);
}
if (!unknown.includes("Unknown key")) {
  throw new Error("unknown key must throw: " + unknown);
}

await page.setContent(`<!DOCTYPE html><html><body>
<form id="f" action="/ziel.html"><input id="q" name="q"></form>
</body></html>`);
await page.evaluate(() => {
  document.getElementById("f").addEventListener("submit", (event) => {
    event.preventDefault();
    window.__greppySubmitted = true;
  });
});
await page.locator("#q").focus();
await page.keyboard.press("Enter");
const submitted = await page.evaluate(() => !!window.__greppySubmitted);
if (!submitted) {
  throw new Error("press Enter must implicitly submit the nearest form");
}

await page.close();
await browser.close();
