import { chromium } from 'playwright';
import { runEfficiency } from './efficiency-workload.mjs';
await runEfficiency(chromium, fixtureUrl);
