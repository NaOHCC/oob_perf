const state = {
    summary: null,
    selectedOperator: null,
    operatorSort: "actual_ms",
    operatorDirection: "desc",
    callSort: "actual_ms",
    callDirection: "desc",
    status: "all",
    offset: 0,
    limit: 100,
};

const formatNumber = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

function escapeHtml(value) {
    return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function formatMs(value) {
    return value == null ? "-" : `${formatNumber.format(value)} ms`;
}

function formatRatio(value) {
    return value == null ? "-" : `${formatNumber.format(value)}x`;
}

function formatBytes(value) {
    if (value == null) return "-";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let number = value;
    let index = 0;
    while (number >= 1000 && index < units.length - 1) {
        number /= 1000;
        index += 1;
    }
    return `${formatNumber.format(number)} ${units[index]}`;
}

function formatFlops(value) {
    if (value == null) return "-";
    const units = ["F", "KF", "MF", "GF", "TF", "PF"];
    let number = value;
    let index = 0;
    while (number >= 1000 && index < units.length - 1) {
        number /= 1000;
        index += 1;
    }
    return `${formatNumber.format(number)} ${units[index]}`;
}

async function getJson(path) {
    const response = await fetch(path, { cache: "no-store" });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error ?? `Request failed: ${response.status}`);
    }
    return response.json();
}

function operatorValue(operator, field) {
    const value = operator[field];
    return value == null ? Number.NEGATIVE_INFINITY : value;
}

function renderSummary() {
    const summary = state.summary;
    document.querySelector("#workload-context").textContent = `${summary.workload_name} | ${summary.device} | ${summary.hardware.label} | actual source: ${summary.actual_source}`;
    document.querySelector("#t1").textContent = formatMs(summary.t1_ms);
    document.querySelector("#t2-wall").textContent = formatMs(summary.t2_wall_ms);
    document.querySelector("#t2-device").textContent = formatMs(summary.t2_device_ms);
    document.querySelector("#efficiency").textContent = formatRatio(summary.efficiency);
    const diagnostics = summary.diagnostics ?? [];
    const panel = document.querySelector("#diagnostics");
    panel.hidden = diagnostics.length === 0;
    document.querySelector("#diagnostic-list").innerHTML = diagnostics.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderOperators() {
    const search = document.querySelector("#operator-search").value.trim().toLowerCase();
    const operators = state.summary.operators
        .filter((operator) => operator.name.toLowerCase().includes(search))
        .sort((left, right) => {
            const difference = operatorValue(left, state.operatorSort) - operatorValue(right, state.operatorSort);
            return state.operatorDirection === "desc" ? -difference : difference;
        });
    document.querySelector("#operator-rows").innerHTML = operators.map((operator) => `
    <tr class="operator-row ${operator.name === state.selectedOperator ? "selected" : ""}" data-operator="${escapeHtml(operator.name)}">
      <td><button class="operator-name" type="button" data-operator="${escapeHtml(operator.name)}">${escapeHtml(operator.name)}</button></td>
      <td>${formatMs(operator.actual_ms)}</td>
      <td>${formatMs(operator.projected_ms)}</td>
      <td>${formatRatio(operator.efficiency)}</td>
    </tr>
  `).join("");
}

function selectedOperator() {
    return state.summary.operators.find((operator) => operator.name === state.selectedOperator);
}

function renderSelectedOperator() {
    const operator = selectedOperator();
    document.querySelector("#selected-operator").textContent = operator.name;
    document.querySelector("#operator-t1").textContent = formatMs(operator.projected_ms);
    document.querySelector("#operator-actual").textContent = formatMs(operator.actual_ms);
    document.querySelector("#operator-efficiency").textContent = formatRatio(operator.efficiency);
    document.querySelector("#operator-calls").textContent = `${operator.projected_calls}/${operator.actual_calls}`;
    const pairing = operator.pairing;
    document.querySelector("#coverage").innerHTML = `
    <span class="badge paired">paired ${pairing.sequence_paired}</span>
    <span class="badge actual">actual-only ${pairing.actual_only}</span>
    <span class="badge projected">projected-only ${pairing.projected_only}</span>
  `;
    document.querySelector("#pairing-note").textContent = "Calls pair only by canonical operator and occurrence order across independent metric and trace passes. Single-sided rows intentionally have no per-call R.";
}

function callOrder(call) {
    return call.actual_index ?? call.projected_index ?? "-";
}

function detailRow(call) {
    return `
    <tr class="call-detail" hidden>
      <td colspan="8">
        <dl class="detail-grid">
          <div><dt>Projected raw operator</dt><dd>${escapeHtml(call.projected_raw_name)}</dd></div>
          <div><dt>Actual raw operator</dt><dd>${escapeHtml(call.actual_raw_name)}</dd></div>
          <div><dt>External id</dt><dd>${escapeHtml(call.external_id)}</dd></div>
          <div><dt>Bound</dt><dd>${escapeHtml(call.bound)}</dd></div>
          <div><dt>Input dimensions</dt><dd>${escapeHtml(call.input_dims)}</dd></div>
          <div><dt>Input strides</dt><dd>${escapeHtml(call.input_strides)}</dd></div>
        </dl>
      </td>
    </tr>
  `;
}

function renderCalls(payload) {
    document.querySelector("#call-count").textContent = `${payload.offset + 1}-${Math.min(payload.offset + payload.calls.length, payload.total)} of ${payload.total}`;
    document.querySelector("#previous-page").disabled = payload.offset === 0;
    document.querySelector("#next-page").disabled = payload.offset + payload.calls.length >= payload.total;
    document.querySelector("#call-rows").innerHTML = payload.calls.map((call) => `
    <tr class="call-row" tabindex="0">
      <td>${escapeHtml(callOrder(call))}</td>
      <td class="state ${escapeHtml(call.match_status)}">${escapeHtml(call.match_status)}</td>
      <td>${formatMs(call.actual_ms)}</td>
      <td>${formatMs(call.projected_ms)}</td>
      <td>${formatRatio(call.efficiency)}</td>
      <td>${formatFlops(call.flops)}</td>
      <td>${formatBytes(call.memory_bytes)}</td>
      <td>${call.timestamp_us == null ? "-" : `${formatNumber.format(call.timestamp_us)} us`}</td>
    </tr>
    ${detailRow(call)}
  `).join("");
}

async function loadCalls() {
    const query = new URLSearchParams({
        operator: state.selectedOperator,
        status: state.status,
        sort: state.callSort,
        direction: state.callDirection,
        offset: String(state.offset),
        limit: String(state.limit),
    });
    const payload = await getJson(`/api/calls?${query}`);
    renderCalls(payload);
}

async function selectOperator(name) {
    state.selectedOperator = name;
    state.offset = 0;
    renderOperators();
    renderSelectedOperator();
    await loadCalls();
}

function toggleSort(target, kind) {
    const field = target.dataset[`${kind}Sort`];
    const sortKey = `${kind}Sort`;
    const directionKey = `${kind}Direction`;
    state[directionKey] = state[sortKey] === field && state[directionKey] === "desc" ? "asc" : "desc";
    state[sortKey] = field;
}

function bindEvents() {
    document.querySelector("#operator-search").addEventListener("input", renderOperators);
    document.querySelector("#operator-rows").addEventListener("click", (event) => {
        const button = event.target.closest("[data-operator]");
        if (button) selectOperator(button.dataset.operator).catch(showError);
    });
    document.querySelectorAll("[data-operator-sort]").forEach((button) => {
        button.addEventListener("click", () => {
            toggleSort(button, "operator");
            renderOperators();
        });
    });
    document.querySelectorAll("[data-call-sort]").forEach((button) => {
        button.addEventListener("click", () => {
            toggleSort(button, "call");
            state.offset = 0;
            loadCalls().catch(showError);
        });
    });
    document.querySelector("#status-filter").addEventListener("change", (event) => {
        state.status = event.target.value;
        state.offset = 0;
        loadCalls().catch(showError);
    });
    document.querySelector("#previous-page").addEventListener("click", () => {
        state.offset = Math.max(0, state.offset - state.limit);
        loadCalls().catch(showError);
    });
    document.querySelector("#next-page").addEventListener("click", () => {
        state.offset += state.limit;
        loadCalls().catch(showError);
    });
    document.querySelector("#call-rows").addEventListener("click", (event) => {
        const row = event.target.closest(".call-row");
        if (row) row.nextElementSibling.hidden = !row.nextElementSibling.hidden;
    });
    document.querySelector("#call-rows").addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
            const row = event.target.closest(".call-row");
            if (row) row.nextElementSibling.hidden = !row.nextElementSibling.hidden;
        }
    });
}

function showError(error) {
    document.querySelector("#selected-operator").textContent = `Viewer error: ${error.message}`;
}

async function initialize() {
    state.summary = await getJson("/api/summary");
    renderSummary();
    const initial = [...state.summary.operators].sort((left, right) => operatorValue(right, "actual_ms") - operatorValue(left, "actual_ms"))[0];
    if (!initial) throw new Error("Analysis contains no operators");
    bindEvents();
    await selectOperator(initial.name);
}

initialize().catch(showError);
