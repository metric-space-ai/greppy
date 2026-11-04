// Only used when an HTTP navigation has neither a recorded response nor a
// typed transport failure. Any element beyond the implied document shell
// already proves content (including a body-less frameset). Stop at its first
// match; do not enumerate every element or read/serialize the whole page.
(function () {
  if (document.querySelector('*:not(html):not(head):not(body)')) return false;
  const text = (document.body && document.body.innerText) || '';
  if (text.trim()) return false;
  return document.documentElement.outerHTML.length < 500;
})()
