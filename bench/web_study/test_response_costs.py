from response_costs import costs


def metadata(reasoning=3):
    usage = {'input_tokens': 100, 'cached_input_tokens': 80, 'output_tokens': 10}
    if reasoning is not None:
        usage['reasoning_output_tokens'] = reasoning
    return {'token_usage_records': [{'response_id': 'one', 'usage': usage}]}


def test_uses_reported_component_counts_without_reading_text():
    result, limits = costs(metadata())
    assert not limits
    assert result['model_responses'] == 1
    assert result['output_tokens'] == 10
    assert result['reasoning_output_tokens'] == 3
    assert result['non_reasoning_output_tokens'] == 7


def test_missing_component_is_unknown_not_zero():
    result, limits = costs(metadata(None))
    assert result['output_tokens'] == 10
    assert result['reasoning_output_tokens'] is None
    assert result['non_reasoning_output_tokens'] is None
    assert limits


def test_conflicting_or_impossible_components_are_not_subtracted():
    for source in (metadata(11), {**metadata(), 'token_usage_conflicts': [{'response_id': 'one'}]}):
        result, limits = costs(source)
        assert result['reasoning_output_tokens'] is None
        assert result['non_reasoning_output_tokens'] is None
        assert limits
