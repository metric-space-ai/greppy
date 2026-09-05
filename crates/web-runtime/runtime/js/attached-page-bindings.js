// A native descriptor is validated completely before any facade is constructed.
// The caller must obtain daemon-authorized metadata; this function never discovers or
// creates a browser, context, page, navigation, or fallback on its own.
function greppyAttachedPageBindings(descriptor, expectedPage, constructors) {
  function refuse(reason) {
    const error = new Error('invalid_attached_page: ' + reason);
    error.code = 'invalid_attached_page';
    throw error;
  }
  function identity(value) {
    return value && typeof value.id === 'string' && value.id.length > 0 &&
      Number.isSafeInteger(value.generation) && value.generation > 0;
  }
  if (typeof expectedPage !== 'string' || !expectedPage) refuse('an explicit page is required');
  if (!descriptor || descriptor.schema !== 'greppy.web.attached-page.v1') refuse('unsupported descriptor schema');
  if (descriptor.selected_page !== expectedPage) refuse('selected page does not match the request');
  if (!identity(descriptor.browser) || !Array.isArray(descriptor.contexts) || !descriptor.contexts.length) {
    refuse('browser or contexts are missing');
  }
  const ids = new Set([descriptor.browser.id]);
  let selectedCount = 0;
  for (const context of descriptor.contexts) {
    if (!identity(context) || ids.has(context.id) || !Array.isArray(context.pages)) refuse('invalid context identity');
    ids.add(context.id);
    for (const page of context.pages) {
      if (!identity(page) || ids.has(page.id) || typeof page.url !== 'string') refuse('invalid page identity or URL');
      ids.add(page.id);
      if (page.id === expectedPage) selectedCount++;
    }
  }
  if (selectedCount !== 1) refuse('the requested page must occur exactly once');
  for (const name of ['Browser', 'BrowserContext', 'Page']) {
    if (!constructors || typeof constructors[name] !== 'function') refuse('facade constructors are unavailable');
  }
  const browser = new constructors.Browser(descriptor.browser.id, descriptor.browser.generation);
  let selected;
  for (const item of descriptor.contexts) {
    const context = new constructors.BrowserContext(item.id, item.generation);
    context._browser = browser;
    browser._contexts.push(context);
    for (const node of item.pages) {
      const page = new constructors.Page(node.id, node.generation);
      page._context = context;
      page._url = node.url;
      context._pages.push(page);
      context._lastPage = node.id;
      if (node.id === expectedPage) selected = page;
    }
  }
  selected._context._lastPage = selected._id;
  return { browser, context: selected._context, page: selected, borrowed: true };
}
