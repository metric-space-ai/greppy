// Identical workload for the native runtime and the installed Playwright reference.
// This measures browser/runtime work; it does not measure an LLM agent.
export async function runEfficiency(chromium, fixtureUrl) {
  const browser = await chromium.launch({ headless: true });
  globalThis.efficiencyBrowser = browser;
  const rounds = [];
  let failure = null;
  try {
    const firstContext = await browser.newContext();
    const firstPage = await firstContext.newPage();
    await firstPage.goto(fixtureUrl);
    const config = JSON.parse(await firstPage.locator('#workload-config').textContent());
    const workers = [{ context: firstContext, page: firstPage }];
    for (let i = 1; i < config.concurrency; i++) {
      const context = await browser.newContext();
      workers.push({ context, page: await context.newPage() });
    }
    const expected = Array.from({ length: config.nodes }, (_, i) => `payload-${i}`)
      .concat(`end:${config.nodes}`).join(' ');
    const postReceipt = async (path, receipt) => {
      const status = await workers[0].page.evaluate(async ({ url, receipt }) => {
        const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(receipt) });
        return response.status;
      }, { url: fixtureUrl + path, receipt });
      if (status !== 200) throw new Error('fixture server rejected ' + path);
    };
    for (let round = 0; round < config.rounds; round++) {
      const roundStart = Date.now();
      const tasks = (await Promise.all(workers.map(async ({ page }, worker) => {
        const results = [];
        for (let task = 0; task < config.tasks; task++) {
          const start = Date.now();
          const token = `${config.nonce}:r${round}:w${worker}:t${task}:é&+`;
          let error = null;
          try {
            await page.goto(fixtureUrl);
            await page.evaluate((id) => {
              const existing = localStorage.getItem('efficiency-worker');
              if (existing !== null && existing !== id) throw new Error('context isolation violated');
              localStorage.setItem('efficiency-worker', id);
            }, String(worker));
            await page.locator('#article').click();
            const text = (await page.locator('main').innerText()).replace(/\s+/g, ' ').trim();
            if (text !== expected) throw new Error('document read did not match the complete expected text');
            const stored = await page.evaluate(() => localStorage.getItem('efficiency-worker'));
            if (stored !== String(worker)) throw new Error('context isolation/persistence violated');
            await page.locator('#query').fill(token);
            await page.locator('#submit').click();
            await page.waitForFunction((value) => {
              const receipt = document.querySelector('#receipt');
              return receipt !== null && receipt.textContent === value;
            }, token);
            if (await page.locator('#receipt').textContent() !== token) throw new Error('POST receipt mismatch');
          } catch (e) {
            error = String(e).slice(0, 500);
          }
          results.push({ round, worker, task, duration_ms: Date.now() - start, ok: error === null, error });
        }
        return results;
      }))).flat();
      const duration = Date.now() - roundStart;
      const succeeded = tasks.filter((task) => task.ok).length;
      rounds.push({ round, duration_ms: duration, succeeded, attempted: tasks.length,
        successful_tasks_per_second: duration > 0 ? succeeded * 1000 / duration : null, tasks });
      await postReceipt('round', rounds[round]);
    }
    const receipt = { schema: 'greppy.web-efficiency.workload.v1', rounds };
    await postReceipt('result', receipt);
    if (rounds.some((round) => round.succeeded !== round.attempted)) failure = new Error('one or more complete tasks failed; no retry');
  } finally {
    await browser.close();
    globalThis.efficiencyBrowser = null;
  }
  if (failure) throw failure;
  return rounds;
}
