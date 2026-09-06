(function(condition, boundNodes, validateOnly) {
  function norm(value) { return String(value == null ? '' : value).replace(/\s+/g, ' ').trim(); }
  function fail(message) { throw new Error('WORKFLOW_CONDITION_SYNTAX: ' + message); }
  function regexp(value) {
    var parsed = /^\/([\s\S]*)\/([imsu]*)$/.exec(value);
    if (!parsed) fail('expected /pattern/flags');
    return new RegExp(parsed[1], parsed[2]);
  }
  function parse(query) {
    query = query.trim();
    if (!query) fail('empty query');
    if (/^@[0-9]+$/.test(query)) return {kind:'ref', value:query};
    var match = /^([a-z]+)(=|~)([\s\S]*)$/.exec(query);
    var kinds = ['css', 'xpath', 'text', 'role', 'id', 'tag'];
    if (match && match[2] === '~' && kinds.indexOf(match[1]) < 0) match = null;
    var parsed = match ? {kind:match[1], op:match[2], value:match[3]} : {kind:'css', op:'=', value:query};
    if (kinds.indexOf(parsed.kind) < 0) fail('unknown query kind ' + parsed.kind);
    if (parsed.op === '~' && parsed.kind !== 'text') fail('regex query needs text');
    if (parsed.op === '=') {
      try { var decoded = JSON.parse(parsed.value); if (typeof decoded === 'string') parsed.value = decoded; } catch (_) {}
    }
    if (!parsed.value.trim()) fail('empty query value');
    return parsed;
  }
  function validateQuery(parsed) {
    if (parsed.kind === 'css') {
      // Let the engine parse against an empty detached fragment. Do not traverse
      // or match the live page, and do not execute a caller-supplied predicate.
      document.createDocumentFragment().querySelectorAll(parsed.value);
    } else if (parsed.kind === 'xpath') {
      document.createExpression(parsed.value, null);
    } else if (parsed.kind === 'text' && parsed.op === '~') {
      regexp(parsed.value);
    }
  }
  function nodesFor(parsed) {
    var all;
    switch (parsed.kind) {
      case 'ref':
        if (!Array.isArray(boundNodes)) throw new Error('REF_CONDITION_UNSUPPORTED');
        return boundNodes;
      case 'css': return Array.from(document.querySelectorAll(parsed.value));
      case 'xpath':
        var iterator = document.evaluate(parsed.value, document, null, 5, null), found = [], node;
        while ((node = iterator.iterateNext())) found.push(node);
        return found;
      case 'id': var node = document.getElementById(parsed.value); return node ? [node] : [];
      case 'tag': return Array.from(document.getElementsByTagName(parsed.value));
      case 'text':
        all = Array.from(document.querySelectorAll('*'));
        if (parsed.op === '~') { var re = regexp(parsed.value); return all.filter(function(node) { return re.test(norm(node.textContent)); }); }
        return all.filter(function(node) { return norm(node.textContent) === norm(parsed.value); });
      case 'role':
        return Array.from(document.querySelectorAll('*')).filter(function(node) {
          var role = node.getAttribute('role');
          if (role) return role === parsed.value;
          var tag = node.tagName.toLowerCase(), type = node.type || '';
          switch (parsed.value) {
            case 'button': return tag === 'button' || (tag === 'input' && /^(button|submit|reset)$/.test(type));
            case 'link': return tag === 'a' && node.hasAttribute('href');
            case 'textbox': return tag === 'textarea' || (tag === 'input' && !/^(button|submit|reset|checkbox|radio|file|number)$/.test(type));
            case 'checkbox': return tag === 'input' && type === 'checkbox';
            case 'spinbutton': return tag === 'input' && type === 'number';
            case 'combobox': return tag === 'select';
            case 'dialog': return tag === 'dialog';
            case 'heading': return /^h[1-6]$/.test(tag);
            default: return false;
          }
        });
    }
    fail('unhandled query kind');
  }
  function matcher(pattern) {
    var trimmed = pattern.trim();
    return trimmed[0] === '~' ? regexp(trimmed.slice(1).trim()) : null;
  }
  // Keep syntax context on the exception, without guessing a different query
  // or serializing action values. Execution-time condition behavior is unchanged.
  function checked(field, syntax, operation) {
    try { return operation(); } catch (error) {
      if (!validateOnly) throw error;
      var diagnostic = new Error(String(error));
      diagnostic.workflowField = field;
      diagnostic.workflowSyntax = syntax;
      throw diagnostic;
    }
  }
  var query = condition.query == null ? null : checked('expectation.query', 'query', function() {
    return parse(condition.query);
  });
  if (validateOnly && query) checked('expectation.query', query.kind === 'text' && query.op === '~' ? 'text-regex' : query.kind, function() {
    validateQuery(query);
  });
  var urlRegex = condition.url == null ? null : checked('expectation.url', 'pattern', function() {
    return matcher(condition.url);
  });
  var titleRegex = condition.title == null ? null : checked('expectation.title', 'pattern', function() {
    return matcher(condition.title);
  });
  if (validateOnly) return true;
  var held = true;
  if (query) held = nodesFor(query).length > 0;
  if (condition.url != null) held = held && (urlRegex ? urlRegex.test(String(location.href)) : String(location.href) === condition.url.trim());
  if (condition.title != null) held = held && (titleRegex ? titleRegex.test(String(document.title)) : String(document.title) === condition.title.trim());
  return condition.absent ? !held : held;
})
