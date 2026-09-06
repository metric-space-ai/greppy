(function(inputs, validateCondition) {
  for (var index = 0; index < inputs.length; index++) {
    try {
      var selector = inputs[index].selector;
      if (selector && selector.type === 'css') {
        document.createDocumentFragment().querySelectorAll(selector.value);
      } else if (selector && selector.type === 'xpath') {
        document.createExpression(selector.value, null);
      }
      if (inputs[index].condition) validateCondition(inputs[index].condition, null, true);
    } catch (error) {
      return {valid:false, step:index + 1, error:String(error)};
    }
  }
  return {valid:true};
})
