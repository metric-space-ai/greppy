const { chromium } = require("playwright");

let denied = false;
try {
  require("fs");
} catch (error) {
  const message = String(error && error.message);
  if (message.includes("denied") || message.includes("controller module policy")) {
    denied = true;
  } else {
    throw error;
  }
}
if (!denied) {
  throw new Error("CJS require(fs) must be denied");
}

module.exports = (async () => {
  const browser = await chromium.launch();
  await browser.close();
})();
