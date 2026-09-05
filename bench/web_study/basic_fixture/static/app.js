const runId = new URLSearchParams(location.search).get('run_id');
const task = document.getElementById('task');
let state;

async function send(action, value, origin) {
  const response = await fetch('/api/action', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({run_id: runId, action, payload: {value, origin}})
  });
  if (!response.ok) { document.getElementById('events').textContent = await response.text(); return; }
  await refresh();
}

function button(label, action, inputId) {
  const b = document.createElement('button'); b.textContent = label;
  b.onclick = () => send(action, document.getElementById(inputId).value);
  return b;
}

function renderText() {
  task.innerHTML = '<h2>Task</h2><p>Replace the existing note with <strong>Ready for review</strong>, then save it.</p>';
  const input = document.createElement('input'); input.id = 'note'; input.value = state.values.note;
  task.append(input, button('Save note', 'set_note', 'note'));
}

function renderCheckbox() {
  task.innerHTML = '<h2>Task</h2><p>Enable the checkbox and change the quantity to <strong>3</strong>.</p>';
  const label = document.createElement('label'); label.textContent = ' Include extra copies';
  const check = document.createElement('input'); check.type = 'checkbox'; check.id = 'enabled'; check.checked = state.values.enabled;
  check.onchange = () => send('set_enabled', check.checked); label.prepend(check);
  const quantity = document.createElement('input'); quantity.id = 'quantity'; quantity.type = 'number'; quantity.min = '1'; quantity.value = state.values.quantity; quantity.disabled = !state.values.enabled;
  quantity.onchange = () => send('set_quantity', Number(quantity.value));
  task.append(label, document.createTextNode(' Quantity: '), quantity);
}

function renderAddress() {
  task.innerHTML = '<h2>Task</h2><p>Choose Germany and Berlin, enter postcode <strong>10115</strong>, and wait for visible validation.</p>';
  const country = document.createElement('select'); country.id = 'country';
  country.innerHTML = '<option value="">Choose country</option>' + Object.keys(state.facts.countries).map(c => `<option>${c}</option>`).join(''); country.value = state.values.country;
  const city = document.createElement('select'); city.id = 'city';
  const cities = state.facts.countries[state.values.country] || [];
  city.innerHTML = '<option value="">Choose city</option>' + cities.map(c => `<option>${c}</option>`).join(''); city.value = state.values.city;
  const postcode = document.createElement('input'); postcode.id = 'postcode'; postcode.value = state.values.postcode;
  country.onchange = () => send('set_country', country.value);
  city.onchange = () => send('set_city', city.value);
  postcode.onchange = () => send('set_postcode', postcode.value);
  const validation = document.createElement('span'); validation.id = 'validation'; validation.textContent = state.values.postcode_valid ? ' Valid postcode' : ' Not validated';
  task.append(country, city, postcode, validation);
}

function renderDialog() {
  task.innerHTML = '<h2>Task</h2><p>Open the <strong>Complete basic task</strong> dialog and click its Save button. The other Save button does not complete this task.</p>';
  const outside = document.createElement('button'); outside.textContent = 'Save'; outside.onclick = () => send('save', '', 'outside');
  const open = document.createElement('button'); open.textContent = 'Open Complete basic task dialog';
  const dialog = document.createElement('dialog'); dialog.innerHTML = '<h2>Complete basic task</h2><p>Review the task, then save here.</p>';
  const inside = document.createElement('button'); inside.textContent = 'Save'; inside.onclick = () => { send('save', '', 'task4-dialog'); dialog.close(); };
  const cancel = document.createElement('button'); cancel.textContent = 'Cancel'; cancel.onclick = () => dialog.close();
  dialog.append(inside, cancel); open.onclick = () => dialog.showModal(); task.append(outside, open, dialog);
}

async function refresh() {
  const response = await fetch('/api/state?run_id=' + encodeURIComponent(runId));
  if (!response.ok) { document.getElementById('events').textContent = await response.text(); return; }
  state = await response.json(); document.getElementById('run').textContent = `Run ${state.run_id} · ${state.case} · revision ${state.revision}`;
  ({text: renderText, checkbox: renderCheckbox, address: renderAddress, dialog: renderDialog}[state.case])();
  document.getElementById('events').textContent = `Revision events: ${state.events.length}`;
}
refresh();
