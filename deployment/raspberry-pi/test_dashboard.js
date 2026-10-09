"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {test} = require("node:test");

const valid = (changes = {}) => ({
  boot_id: "boot-a", sequence: 1, mode: "recorded_raw_replay", status: "ok",
  age_seconds: 1, prediction: {SBP_mmHg: 120, DBP_mmHg: 75}, inference_ms: 6,
  pipeline_ms: 10, waveforms: {ecg: Array(1250).fill(.5), ppg: Array(1250).fill(.5)},
  updated_at: "2026-10-09T00:00:00Z", message: "Raw replay", ...changes,
});

async function dashboard() {
  const elements = new Map();
  let now = 0;
  let payload = valid();
  let pending = null;
  let calls = 0;
  const intervals = [];
  const events = {};
  const context = vm.createContext({
    document: {getElementById(id) {
      if (!elements.has(id)) elements.set(id, {
        textContent: "", clientWidth: 300, clientHeight: 150,
        classList: {toggle(_class, enabled) {this.enabled = enabled;}},
        getContext: () => new Proxy({}, {get: () => () => {}, set: () => true}),
      });
      return elements.get(id);
    }},
    window: {devicePixelRatio: 1, addEventListener: (name, cb) => {events[name] = cb;}},
    performance: {now: () => now}, AbortController,
    setTimeout: () => 1, clearTimeout: () => {}, setInterval: cb => intervals.push(cb),
    fetch: async () => {
      calls++;
      if (pending) await pending;
      return {ok: true, json: async () => payload};
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, "web/app.js"), "utf8"), context);
  await new Promise(resolve => setImmediate(resolve));
  return {elements, context, intervals, events, calls: () => calls,
    setPayload: value => {payload = value;}, setTime: value => {now = value;},
    block: promise => {pending = promise;}};
}

test("raw recordings are labeled replay; invalid BP never gets a green state", async () => {
  const ui = await dashboard();
  assert.equal(ui.elements.get("source-badge").textContent, "RAW RECORDING / replay");
  assert.equal(ui.elements.get("sbp").textContent, "120.0");
  ui.setPayload(valid({sequence: 2, prediction: {SBP_mmHg: 70, DBP_mmHg: 90}}));
  await ui.context.refresh();
  assert.equal(ui.elements.get("sbp").textContent, "--");
  assert.equal(ui.elements.get("status-dot").classList.enabled, false);
});

test("service restart accepts a new sequence and redraws cleared readings", async () => {
  const ui = await dashboard();
  ui.setPayload(valid({sequence: 2, status: "poor_signal", prediction: null}));
  await ui.context.refresh();
  assert.equal(ui.elements.get("sbp").textContent, "--");
  ui.setPayload(valid({boot_id: "boot-b", sequence: 0}));
  await ui.context.refresh();
  assert.equal(ui.elements.get("sbp").textContent, "120.0");
  assert.equal(ui.elements.get("sequence").textContent, "0");
});

test("readings expire while a request is in flight", async () => {
  const ui = await dashboard();
  ui.setTime(12000);
  ui.intervals[0]();
  assert.equal(ui.elements.get("sbp").textContent, "--");
  assert.equal(ui.elements.get("status").textContent, "Input stale");
});

test("requests never overlap and resize does not request data", async () => {
  const ui = await dashboard();
  let unblock;
  ui.block(new Promise(resolve => {unblock = resolve;}));
  const first = ui.context.refresh();
  await ui.context.refresh();
  ui.events.resize();
  assert.equal(ui.calls(), 2); // initial fetch plus the single pending request
  unblock();
  await first;
});

test("network transit time cannot make a delayed reading appear fresh", async () => {
  const ui = await dashboard();
  let unblock;
  ui.setPayload(valid({sequence: 2, age_seconds: 11}));
  ui.block(new Promise(resolve => {unblock = resolve;}));
  const request = ui.context.refresh();
  ui.setTime(2000);
  unblock();
  await request;
  assert.equal(ui.elements.get("sbp").textContent, "--");
});
