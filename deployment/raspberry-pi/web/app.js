"use strict";
const el = (id) => document.getElementById(id);
const ids = ["sbp", "dbp", "segment", "source-name", "sequence", "updated", "latency", "hostname", "age", "pipeline"];
const sourceCopy = {
  recorded_pulsedb_replay: {
    badge: "SAMPLE RECORDING / PulseDB",
    description: "Previously recorded ECG and PPG. No device is connected; segments may come from different people.",
    sourceName: "PulseDB sample",
    active: "Sample data active",
  },
  device_live: {
    badge: "CONNECTED DEVICE / prototype",
    description: "ECG and PPG from a connected device. Device compatibility requires validation.",
    sourceName: "Connected device",
    active: "Device input active",
  },
  recorded_raw_replay: {
    badge: "RAW RECORDING / replay",
    description: "Previously recorded raw ECG and PPG, replayed at acquisition speed. No device is connected.",
    sourceName: "Raw recording",
    active: "Raw replay active",
  },
};
let lastSequence = null;
let currentMode = null;
let bootId = null;
let responseSequence = -1;
let refreshing = false;
let lastData = null;
let renderedAt = null;

function presentSource(mode) {
  const copy = sourceCopy[mode];
  if (!copy || currentMode === mode) return Boolean(copy);
  el("source-badge").textContent = copy.badge;
  el("source-description").textContent = copy.description;
  el("source-name").textContent = copy.sourceName;
  currentMode = mode;
  return true;
}

function sourceUnavailable() {
  currentMode = null;
  el("source-badge").textContent = "SOURCE UNAVAILABLE";
  el("source-description").textContent = "Waiting for source status from the Raspberry Pi.";
}

function clearReadings() {
  lastSequence = null;
  lastData = null;
  renderedAt = null;
  for (const id of ids) el(id).textContent = "--";
  drawWave("ecg-chart", null, "#85d6c8");
  drawWave("ppg-chart", null, "#edc07d");
}

function setStatus(message, ok) {
  el("status").textContent = message;
  el("status-dot").classList.toggle("ok", ok);
}

function drawWave(id, samples, color) {
  const canvas = el(id);
  const scale = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * scale);
  canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.clearRect(0, 0, width, height);
  const left = 39, right = width - 12, top = 12, bottom = height - 24;
  ctx.font = "11px Arial";
  ctx.strokeStyle = "#30434b";
  ctx.fillStyle = "#94a9ad";
  ctx.lineWidth = 1;
  for (const value of [0, 0.5, 1]) {
    const y = bottom - value * (bottom - top);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    ctx.fillText(String(value), 5, y + 4);
  }
  for (const second of [0, 5, 10]) {
    const x = left + second / 10 * (right - left);
    ctx.fillText(`${second}s`, x - 6, height - 6);
  }
  if (!Array.isArray(samples) || samples.length !== 1250) return;
  ctx.beginPath();
  for (let i = 0; i < samples.length; i++) {
    const x = left + i / (samples.length - 1) * (right - left);
    const value = Math.min(1, Math.max(0, samples[i]));
    const y = bottom - value * (bottom - top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.stroke();
}

async function refresh() {
  if (refreshing) return;
  refreshing = true;
  const controller = new AbortController();
  const requestedAt = performance.now();
  const timer = setTimeout(() => controller.abort(), 3500);
  try {
    const response = await fetch("/api/state", {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (Number.isFinite(data.age_seconds)) {
      // Conservatively include transit time; an old delayed response is not fresh.
      data.age_seconds += (performance.now() - requestedAt) / 1000;
    }
    if (typeof data.boot_id !== "string" || !Number.isInteger(data.sequence)) {
      throw new Error("Missing service identity");
    }
    if (bootId !== data.boot_id) {
      bootId = data.boot_id;
      responseSequence = -1;
      lastSequence = null;
    }
    if (data.sequence < responseSequence) return;
    responseSequence = data.sequence;
    if (!presentSource(data.mode)) {
      clearReadings();
      sourceUnavailable();
      setStatus("Unsupported source", false);
      el("message").textContent = "This source mode is not approved for display.";
      return;
    }
    const prediction = data.prediction;
    if (data.status !== "ok" ||
        !Number.isFinite(data.age_seconds) || data.age_seconds < 0 || data.age_seconds > 12 ||
        !prediction || !Number.isFinite(prediction.SBP_mmHg) || !Number.isFinite(prediction.DBP_mmHg) ||
        prediction.SBP_mmHg <= prediction.DBP_mmHg || prediction.DBP_mmHg <= 0 ||
        !Number.isFinite(data.inference_ms) || !data.waveforms) {
      clearReadings();
      const labels = {initializing: "Collecting a window", poor_signal: "Check contact / movement",
        discontinuity: "Collecting after a data gap", disconnected: "Sensor disconnected",
        ended: "Recording ended", invalid_prediction: "Estimate unavailable", stale: "Input stale"};
      setStatus(labels[data.status] || "Estimate unavailable", false);
      el("message").textContent = data.message || "Waiting for a fresh input window.";
      return;
    }
    setStatus(sourceCopy[data.mode].active, true);
    el("sbp").textContent = data.prediction.SBP_mmHg.toFixed(1);
    el("dbp").textContent = data.prediction.DBP_mmHg.toFixed(1);
    el("segment").textContent = data.segment_id || "--";
    el("source-name").textContent = sourceCopy[data.mode].sourceName;
    el("sequence").textContent = String(data.sequence);
    el("updated").textContent = new Date(data.updated_at).toLocaleString();
    el("latency").textContent = `${data.inference_ms.toFixed(2)} ms`;
    el("hostname").textContent = data.hostname;
    el("age").textContent = `${data.age_seconds.toFixed(1)} s`;
    el("pipeline").textContent = Number.isFinite(data.pipeline_ms) ? `${data.pipeline_ms.toFixed(1)} ms` : "--";
    el("message").textContent = data.message;
    if (lastSequence !== data.sequence) {
      drawWave("ecg-chart", data.waveforms.ecg, "#85d6c8");
      drawWave("ppg-chart", data.waveforms.ppg, "#edc07d");
      lastSequence = data.sequence;
    }
    lastData = data;
    renderedAt = performance.now();
  } catch (error) {
    clearReadings();
    lastSequence = null;
    sourceUnavailable();
    setStatus("Unavailable / disconnected", false);
    el("message").textContent = "Raspberry Pi service is unreachable. No current reading is shown.";
  } finally {
    clearTimeout(timer);
    refreshing = false;
    setTimeout(refresh, 1000);
  }
}

window.addEventListener("resize", () => {
  drawWave("ecg-chart", lastData?.waveforms.ecg, "#85d6c8");
  drawWave("ppg-chart", lastData?.waveforms.ppg, "#edc07d");
});
// Expire displayed data even while a slow request is in flight.
setInterval(() => {
  if (lastData && renderedAt !== null) {
    const age = lastData.age_seconds + (performance.now() - renderedAt) / 1000;
    if (age > 12) {
      clearReadings();
      setStatus("Input stale", false);
      el("message").textContent = "Waiting for a fresh input window.";
    }
  }
}, 250);
clearReadings();
refresh();
