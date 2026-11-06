import { pathToFileURL } from 'node:url';
import { readFileSync } from 'node:fs';
import { runEfficiency } from '../runtime/fixtures/efficiency-workload.mjs';

if (!process.env.PLAYWRIGHT_PACKAGE) throw new Error('PLAYWRIGHT_PACKAGE must name an existing installation; no download is performed');
const root = process.env.PLAYWRIGHT_PACKAGE;
const { chromium } = await import(pathToFileURL(`${root}/index.mjs`).href);
const version = JSON.parse(readFileSync(`${root}/package.json`, 'utf8')).version;
process.on('SIGTERM', async () => {
  try { await globalThis.efficiencyBrowser?.close(); } finally { process.exit(124); }
});
await runEfficiency({ launch: async (options) => {
  const browser = await chromium.launch(options);
  console.log(JSON.stringify({ engine_metadata: {
    playwright_version: version,
    chromium_version: browser.version(),
    chromium_executable: chromium.executablePath(),
  } }));
  return browser;
} }, process.argv[2]);
