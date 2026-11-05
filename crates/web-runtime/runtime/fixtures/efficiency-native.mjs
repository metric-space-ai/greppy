import { chromium } from 'playwright';
import { runEfficiency } from './efficiency-workload.mjs';

// The native runtime has a fixed headless execution mode and rejects launch
// options. Accept only the common workload's exact intent, never other options.
export function nativeHeadlessBrowserType(browserType) {
  return { launch: async (options) => {
    if (options === null || typeof options !== 'object'
        || Object.getPrototypeOf(options) !== Object.prototype
        || Reflect.ownKeys(options).length !== 1 || options.headless !== true) {
      throw new Error('native efficiency launch requires exactly { headless: true }; other options are unsupported');
    }
    return browserType.launch();
  } };
}

await runEfficiency(nativeHeadlessBrowserType(chromium), fixtureUrl);
