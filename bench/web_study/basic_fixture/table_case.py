"""Deterministic inventory task; the oracle derives the goal from original facts."""
import random

ACTIONS = {'filter_region', 'filter_capacity', 'sort_price', 'reserve_item'}


def new_state(run_id, seed):
    rng = random.Random(str(seed))
    prices = [900, 1200, 1500, 1800, 2100, 2500]
    rng.shuffle(prices)
    inventory = [{'id': name.lower(), 'name': name, 'region': region,
                  'available': stock, 'unit_cents': price}
                 for (name, region, stock), price in zip([
                     ('Atlas', 'EU', 0), ('Beacon', 'US', 8), ('Cedar', 'EU', 3),
                     ('Delta', 'EU', 2), ('Ember', 'EU', 4), ('Flint', 'EU', 3)], prices)]
    rng.shuffle(inventory)
    return {'run_id': run_id, 'seed': seed, 'case': 'table', 'revision': 0,
            'facts': {'inventory': inventory}, 'events': [],
            'values': {'region': 'all', 'capacity_only': False, 'price_order': 'none',
                       'remaining': {i['id']: i['available'] for i in inventory}, 'reservations': []}}


def apply(s, action, payload):
    v = s['values']
    value = payload.get('value')
    if action == 'filter_region':
        if value not in ('all', 'EU', 'US', 'APAC'): raise ValueError('unknown region')
        v['region'] = value
    elif action == 'filter_capacity':
        if not isinstance(value, bool): raise ValueError('capacity filter must be boolean')
        v['capacity_only'] = value
    elif action == 'sort_price':
        if value not in ('none', 'ascending', 'descending'): raise ValueError('unknown price order')
        v['price_order'] = value
    elif action == 'reserve_item':
        if not isinstance(value, dict): raise ValueError('reservation must be an object')
        item = next((i for i in s['facts']['inventory'] if i['id'] == value.get('id')), None)
        quantity = value.get('quantity')
        if item is None: raise ValueError('unknown item')
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError('quantity must be a positive integer')
        if payload.get('origin') != 'reservation-dialog': raise ValueError('confirm in reservation dialog')
        if quantity > v['remaining'][item['id']]: raise ValueError('not enough remaining stock')
        reservation = {'id': item['id'], 'name': item['name'], 'quantity': quantity,
                       'total_cents': quantity * item['unit_cents']}
        v['remaining'][item['id']] -= quantity
        v['reservations'].append(reservation)
    else:
        raise ValueError('unknown inventory action')


def checks(s):
    v = s['values']
    eligible = [i for i in s['facts']['inventory'] if i['region'] == 'EU' and i['available'] >= 3]
    target = min(eligible, key=lambda i: i['unit_cents'])
    expected = {'id': target['id'], 'name': target['name'], 'quantity': 3,
                'total_cents': target['unit_cents'] * 3}
    return {'region_filter': v['region'] == 'EU', 'capacity_filter': v['capacity_only'] is True,
            'ascending_price': v['price_order'] == 'ascending',
            'one_correct_reservation': v['reservations'] == [expected],
            'stock_effect': all(v['remaining'][i['id']] == i['available'] - (3 if i['id'] == target['id'] else 0)
                                for i in s['facts']['inventory'])}
