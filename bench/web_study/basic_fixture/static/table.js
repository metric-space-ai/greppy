function renderTable() {
  if (!document.getElementById('inventory-table')) {
    task.innerHTML = `<h2>Inventory reservation</h2>
      <p>Show <strong>EU</strong> items with <strong>at least 3 available</strong>, sort unit price <strong>low to high</strong>, then reserve <strong>3</strong> of the cheapest matching item. Confirm in its reservation dialog.</p>
      <div class="inventory-filters">
        <label>Region <select id="inventory-region"><option value="all">All regions</option><option>EU</option><option>US</option><option>APAC</option></select></label>
        <label><input id="inventory-capacity" type="checkbox"> At least 3 available</label>
        <label>Unit price order <select id="inventory-order"><option value="none">Unsorted</option><option value="ascending">Low to high</option><option value="descending">High to low</option></select></label>
      </div>
      <p id="inventory-count" role="status"></p>
      <table id="inventory-table"><caption>Available inventory</caption><thead><tr>
        <th scope="col">Item</th><th scope="col">Region</th><th scope="col">Available</th><th scope="col" id="price-heading" aria-sort="none">Unit price</th><th scope="col">Action</th>
      </tr></thead><tbody></tbody></table>
      <p id="inventory-empty" hidden>No matching items.</p>
      <h3>Reservations</h3><div id="reservation-status" role="status"></div>
      <dialog id="reservation-dialog" aria-labelledby="reservation-title">
        <form id="reservation-form"><h2 id="reservation-title">Reserve item</h2>
          <p id="reservation-price"></p>
          <label>Quantity <input id="reservation-quantity" type="number" required min="1" step="1" value="1"></label>
          <button type="submit">Confirm reservation</button><button id="reservation-cancel" type="button">Cancel</button>
        </form>
      </dialog>`;
    document.getElementById('inventory-region').onchange = event => send('filter_region', event.target.value);
    document.getElementById('inventory-capacity').onchange = event => send('filter_capacity', event.target.checked);
    document.getElementById('inventory-order').onchange = event => send('sort_price', event.target.value);
    document.getElementById('reservation-cancel').onclick = () => document.getElementById('reservation-dialog').close();
    document.getElementById('reservation-form').onsubmit = async event => {
      event.preventDefault();
      const dialog = document.getElementById('reservation-dialog');
      const count = state.values.reservations.length;
      await send('reserve_item', {id: dialog.dataset.itemId, quantity: Number(document.getElementById('reservation-quantity').value)}, 'reservation-dialog');
      if (state.values.reservations.length > count) dialog.close();
    };
  }
  const values = state.values;
  document.getElementById('inventory-region').value = values.region;
  document.getElementById('inventory-capacity').checked = values.capacity_only;
  document.getElementById('inventory-order').value = values.price_order;
  document.getElementById('price-heading').setAttribute('aria-sort', values.price_order);
  const rows = state.facts.inventory.filter(item =>
    (values.region === 'all' || item.region === values.region) &&
    (!values.capacity_only || values.remaining[item.id] >= 3));
  if (values.price_order !== 'none') rows.sort((a, b) => (a.unit_cents - b.unit_cents) * (values.price_order === 'ascending' ? 1 : -1));
  const body = document.querySelector('#inventory-table tbody');
  body.replaceChildren();
  for (const item of rows) {
    const row = document.createElement('tr');
    for (const value of [item.name, item.region, values.remaining[item.id], 'EUR ' + (item.unit_cents / 100).toFixed(2)]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    const cell = document.createElement('td');
    const reserve = document.createElement('button'); reserve.textContent = 'Reserve';
    reserve.setAttribute('aria-label', 'Reserve ' + item.name);
    reserve.disabled = values.remaining[item.id] === 0;
    reserve.onclick = () => {
      const dialog = document.getElementById('reservation-dialog');
      document.getElementById('reservation-title').textContent = 'Reserve ' + item.name;
      document.getElementById('reservation-price').textContent = 'Unit price: EUR ' + (item.unit_cents / 100).toFixed(2);
      const quantity = document.getElementById('reservation-quantity');
      quantity.value = '1'; quantity.max = String(state.values.remaining[item.id]);
      dialog.dataset.itemId = item.id;
      dialog.showModal();
    };
    cell.append(reserve); row.append(cell); body.append(row);
  }
  document.getElementById('inventory-count').textContent = rows.length + ' matching items';
  document.getElementById('inventory-empty').hidden = rows.length !== 0;
  const status = document.getElementById('reservation-status');
  status.replaceChildren();
  if (!values.reservations.length) status.textContent = 'No reservations yet.';
  for (const reservation of values.reservations) {
    const line = document.createElement('p');
    line.textContent = 'Reserved ' + reservation.quantity + ' × ' + reservation.name + '. Total: EUR ' + (reservation.total_cents / 100).toFixed(2) + '.';
    status.append(line);
  }
}
