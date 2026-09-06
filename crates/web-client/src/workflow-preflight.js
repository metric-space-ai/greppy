(function(inputs, validateCondition) {
  for (var index = 0; index < inputs.length; index++) {
    var field = 'action.selector', syntax = null;
    try {
      var selector = inputs[index].selector;
      if (selector && selector.type === 'css') {
        syntax = 'css';
        document.createDocumentFragment().querySelectorAll(selector.value);
      } else if (selector && selector.type === 'xpath') {
        syntax = 'xpath';
        document.createExpression(selector.value, null);
      }
      if (inputs[index].condition) {
        field = 'expectation';
        syntax = null;
        validateCondition(inputs[index].condition, null, true);
      }
    } catch (error) {
      return {valid:false, step:index + 1,
        field:field === 'expectation' ? error && error.workflowField : field,
        syntax:field === 'expectation' ? error && error.workflowSyntax : syntax,
        error:String(error)};
    }
  }
  return {valid:true};
})
