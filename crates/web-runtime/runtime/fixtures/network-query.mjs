const sameRequests = [];
const sameResponseStatuses = [];
const failedUrls = [];
const finishedUrls = [];
page.on("request", (request) => {
  if (String(request.url()).endsWith("/same")) sameRequests.push(request);
});
page.on("response", (response) => {
  if (String(response.url()).endsWith("/same")) {
    sameResponseStatuses.push(response.status());
  }
});
page.on("requestfailed", (request) => failedUrls.push(String(request.url())));
page.on("requestfinished", (request) => finishedUrls.push(String(request.url())));

await page.route("**/same", (route) =>
  route.fulfill({ body: "ok", contentType: "text/plain", status: 200 }),
);
await page.goto(fixtureUrl + "same");
await page.unroute("**/same");
await page.route("**/same", (route) =>
  route.fulfill({ body: "missing", contentType: "text/plain", status: 404 }),
);
await page.goto(fixtureUrl + "same");

if (sameResponseStatuses.join(",") !== "200,404") {
  throw new Error("same URL response statuses " + JSON.stringify(sameResponseStatuses));
}
const linkedStatuses = await Promise.all(
  sameRequests.map(async (request) => {
    const response = await request.response();
    return response && response.status();
  }),
);
if (linkedStatuses.join(",") !== "200,404") {
  throw new Error("same URL linked response statuses " + JSON.stringify(linkedStatuses));
}

await page.route("**/failed", (route) => route.abort());
try {
  await page.goto(fixtureUrl + "failed");
} catch (_error) {
  // A failed transport is still a network record. Keep the page alive so the
  // caller can inspect it after this script returns.
}
if (!failedUrls.some((url) => url.endsWith("/failed"))) {
  throw new Error("aborted request did not emit requestfailed");
}
if (finishedUrls.some((url) => url.endsWith("/failed"))) {
  throw new Error("aborted request emitted requestfinished");
}
