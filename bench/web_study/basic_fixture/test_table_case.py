import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import server


@pytest.fixture
def inventory(tmp_path):
    server.set_dir(tmp_path)
    state = server.new_state('abcdef012345', 'table-test-1', 'table')
    server.write_state(state)
    return state


def target(state):
    return min((i for i in state['facts']['inventory'] if i['region'] == 'EU' and i['available'] >= 3),
               key=lambda i: i['unit_cents'])


def action(state, name, value, origin=None):
    payload = {'value': value}
    if origin is not None: payload['origin'] = origin
    server.mutate(state, name, payload)


def filters(state, skip=None):
    for name, value in [('filter_region', 'EU'), ('filter_capacity', True), ('sort_price', 'ascending')]:
        if name != skip: action(state, name, value)


def reserve(state, item=None, quantity=3):
    action(state, 'reserve_item', {'id': (item or target(state))['id'], 'quantity': quantity}, 'reservation-dialog')


def test_seed_reproducibility_varies_target_not_just_initial_row_order():
    names = set()
    for seed in range(12):
        a = server.new_state('abcdef012345', str(seed), 'table')
        b = server.new_state('123456abcdef', str(seed), 'table')
        assert a['facts'] == b['facts']
        assert len({i['unit_cents'] for i in a['facts']['inventory']}) == 6
        names.add(target(a)['id'])
    assert len(names) > 1


def test_complete_goal_persists_and_reserved_row_leaves_filter(inventory):
    filters(inventory)
    reserve(inventory)
    stored = server.load(inventory['run_id'])
    assert server.verify(stored)['ok']
    assert stored['revision'] == 4
    assert stored['values']['remaining'][target(stored)['id']] < 3
    assert len(stored['values']['reservations']) == 1


@pytest.mark.parametrize('missing', ['filter_region', 'filter_capacity', 'sort_price'])
def test_reservation_alone_does_not_prove_filter_sort_workflow(inventory, missing):
    filters(inventory, skip=missing)
    reserve(inventory)
    assert not server.verify(server.load(inventory['run_id']))['ok']


def test_valid_reservation_of_wrong_item_is_not_task_success(inventory):
    filters(inventory)
    wrong = next(i for i in inventory['facts']['inventory'] if i['region'] == 'US')
    reserve(inventory, wrong)
    assert not server.verify(inventory)['checks']['one_correct_reservation']


def test_duplicate_reservations_are_retained_and_not_success(inventory):
    filters(inventory)
    reserve(inventory, quantity=1)
    reserve(inventory, quantity=1)
    assert len(inventory['values']['reservations']) == 2
    assert not server.verify(inventory)['ok']


@pytest.mark.parametrize('quantity', [0, -1, True, 1.5, 999])
def test_invalid_reservation_does_not_erase_prior_work(inventory, quantity):
    filters(inventory)
    before = copy.deepcopy(inventory)
    with pytest.raises(ValueError): reserve(inventory, quantity=quantity)
    assert inventory == before
    assert server.load(inventory['run_id']) == before


def test_dialog_origin_required(inventory):
    before = copy.deepcopy(inventory)
    with pytest.raises(ValueError, match='dialog'):
        action(inventory, 'reserve_item', {'id': target(inventory)['id'], 'quantity': 3}, 'outside')
    assert inventory == before
