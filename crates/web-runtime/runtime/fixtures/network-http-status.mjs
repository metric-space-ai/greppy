for (const origin of [fixtureUrl, httpsFixtureUrl]) {
  await page.goto(origin + "ok");
  await page.goto(origin + "missing");
  await page.goto(origin + "repeat");
  await page.goto(origin + "repeat");
  await page.goto(origin + "jump");
  await page.goto(origin + "chunked");
  await page.evaluate(async (url) => {
    const response = await fetch(url);
    await response.arrayBuffer();
  }, origin + "empty");
}
try {
  await page.goto(failedFixtureUrl);
} catch (_error) {
  // The completion callback must preserve a transport failure without inventing
  // a response status.
}
// Keep the active page alive for the caller's subsequent web.network read.
