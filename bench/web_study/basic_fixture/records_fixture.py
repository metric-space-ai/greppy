"""Synthetic delivery task with independent file-backed outcome verification."""
from __future__ import annotations
import argparse
import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import random
import re
import threading
import time

TASK = ('Choose the cheapest visible and available EU delivery that arrives within 3 days. '
        'Replace the delivery note with Leave with reception, accept the delivery conditions, '
        'save and confirm the booking exactly once. Reload and verify that it persisted. '
        'Preserve the existing booking.')


def initial(trial, seed):
    rng = random.Random(str(seed))
    offers = []
    for i in range(18):
        offers.append({'id': f'offer-{i}', 'name': f'Carrier {i+1:02d}',
            'price': i + 4, 'days': 1 + rng.randrange(5),
            'region': ['EU','US','EU'][i % 3], 'available': i % 7 != 0, 'visible': True})
    # Guaranteed distinct, legitimate alternatives; original facts drive oracle.
    offers[2].update(region='EU', days=2, available=True)
    offers.extend([
        {'id':'hidden','name':'Carrier 03','price':1,'days':1,'region':'EU','available':True,'visible':False},
        {'id':'disabled','name':'Carrier 03','price':2,'days':1,'region':'EU','available':False,'visible':True},
    ])
    rng.shuffle(offers)
    return {'trial': trial, 'seed': seed, 'offers': offers, 'booking': None,
            'existing': {'id':'KEEP-782','note':'Existing booking must remain'}, 'events': [], 'reloads_after_save': 0}


def oracle(state):
    eligible = [o for o in state['offers'] if o['visible'] and o['available'] and o['region']=='EU' and o['days']<=3]
    expected = min(eligible, key=lambda x: x['price'])
    booking = state['booking'] or {}
    return {'correct_offer': booking.get('offer') == expected['id'],
            'note': booking.get('note') == 'Leave with reception',
            'conditions': booking.get('accepted') is True,
            'one_save': len(state['events']) == 1,
            'reloaded_after_save': state['reloads_after_save'] >= 1,
            'existing_preserved': state['existing'] == {'id':'KEEP-782','note':'Existing booking must remain'}}


def page(state):
    buttons = []
    for offer in state['offers']:
        label = f"{offer['name']} — {offer['region']} — EUR {offer['price']} — {offer['days']} days"
        buttons.append(f'<button type="button" class="offer" id="{offer["id"]}" '
            f'data-price="{offer["price"]}" data-days="{offer["days"]}" data-region="{offer["region"]}" '
            f'{"hidden" if not offer["visible"] else ""} {"disabled" if not offer["available"] else ""} '
            f'onclick="choose(this)">{html.escape(label)}</button>')
    booking = state['booking']
    saved = 'No booking yet.' if not booking else f'Saved booking: {booking["offer"]}; note: {booking["note"]}; conditions accepted: {booking["accepted"]}'
    body = '''<!doctype html><html><head><title>Delivery booking</title><style>
body{font:16px system-ui;margin:24px;max-width:1000px}.offer{display:block;margin:6px;padding:8px}.offer[hidden]{display:none}label{display:block;margin:12px}dialog{padding:24px}
</style></head><body><h1>Delivery booking</h1><p>Choose a delivery option, add your note, then save and confirm.</p>
<p id="existing">Existing booking KEEP-782: Existing booking must remain</p><section id="offers"><h2>Delivery offers</h2>''' + ''.join(buttons) + '''</section>
<p id="selected" role="status">No option selected</p><label>Delivery note <input id="note" value="Draft note"></label>
<label><input type="checkbox" id="accepted">Accept delivery conditions</label>
<button id="save" onclick="beginSave()">Save booking</button><p id="saved" role="status">''' + html.escape(saved) + '''</p>
<dialog id="confirmation"><h2>Confirm booking</h2><p id="confirmation-text"></p><button id="confirm" onclick="commit()">Confirm booking</button><button onclick="document.querySelector('#confirmation').close()">Cancel</button></dialog>
<script>
let selected=null;
function choose(button){selected=button.id;document.querySelector('#selected').textContent='Selected: '+button.textContent;}
function beginSave(){if(!selected||!document.querySelector('#accepted').checked){document.querySelector('#saved').textContent='Select an option and accept the conditions.';return;}
document.querySelector('#confirmation-text').textContent=document.querySelector('#selected').textContent+'; note: '+document.querySelector('#note').value;document.querySelector('#confirmation').showModal();}
async function commit(){const button=document.querySelector('#confirm');button.disabled=true;
try{const response=await fetch(location.pathname+'/commit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({offer:selected,note:document.querySelector('#note').value,accepted:document.querySelector('#accepted').checked})});
const result=await response.json();if(!response.ok)throw new Error(result.error);document.querySelector('#saved').textContent='Saved booking: '+result.booking.offer+'; note: '+result.booking.note+'; conditions accepted: '+result.booking.accepted;document.querySelector('#confirmation').close();}
catch(e){document.querySelector('#saved').textContent='Save failed: '+e.message;button.disabled=false;}}
</script></body></html>'''
    return body.encode()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass
    def reply(self, code, content, kind='application/json'):
        self.send_response(code); self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(content))); self.end_headers(); self.wfile.write(content)
    def trial_path(self, commit=False):
        pattern = r'/trial/([a-z0-9-]+)' + ('/commit' if commit else '')
        match = re.fullmatch(pattern, self.path)
        return self.server.state_dir / (match[1]+'.json') if match else None
    def do_GET(self):
        path = self.trial_path()
        if path is None or not path.exists():
            self.reply(404, b'{"error":"unknown trial"}'); return
        with self.server.state_lock:
            state = json.loads(path.read_text())
            if state['booking']:
                state['reloads_after_save'] += 1
                path.write_text(json.dumps(state))
            body = page(state)
        self.reply(200, body, 'text/html; charset=utf-8')
    def do_POST(self):
        path = self.trial_path(commit=True)
        if path is None or not path.exists():
            self.reply(404, b'{"error":"unknown trial"}'); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size < 4096: raise ValueError('invalid request size')
            value = json.loads(self.rfile.read(size))
            with self.server.state_lock:
                state = json.loads(path.read_text())
                offer = next((o for o in state['offers'] if o['id']==value.get('offer')), None)
                if not offer or not offer['visible'] or not offer['available']: raise ValueError('unavailable offer')
                if value.get('accepted') is not True: raise ValueError('conditions missing')
                if not isinstance(value.get('note'), str) or len(value['note']) > 300: raise ValueError('invalid note')
                if state['booking'] is not None: raise ValueError('booking already saved')
                state['booking'] = {'offer': offer['id'], 'note': value['note'], 'accepted': True}
                state['events'].append({'kind':'confirm_save','time':time.time(), **state['booking']})
                path.write_text(json.dumps(state))
            self.reply(200, json.dumps({'booking':state['booking']}).encode())
        except (ValueError, TypeError) as error:
            self.reply(400, json.dumps({'error':str(error)}).encode())

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state-dir', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    a = p.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.state_dir = a.state_dir
    server.state_lock = threading.Lock()
    a.receipt.write_text(json.dumps({'port':server.server_port,'pid':__import__('os').getpid()}))
    print(f'FIXTURE http://127.0.0.1:{server.server_port}', flush=True)
    try: server.serve_forever()
    finally: server.server_close()
