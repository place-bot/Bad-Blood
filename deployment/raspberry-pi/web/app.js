"use strict";
const el = (id) => document.getElementById(id);
const ids = ["sbp", "dbp", "segment", "position", "cycle", "sequence", "updated", "latency", "hostname"];
let lastSequence = null;

function clearReadings() {
  for (const id of ids) el(id).textContent = "--";
  drawWave("ecg-chart", null, "#087f78");
  drawWave("ppg-chart", null, "#4268b4");
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
  ctx.strokeStyle = "#dce7e6";
  ctx.fillStyle = "#587078";
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
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 3500);
  try {
    const response = await fetch("/api/state", {cache: "no-store", signal: controller.signal});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!(["ok", "review_prediction"].includes(data.status)) ||
        data.age_seconds == null || data.age_seconds > 12 || !data.prediction) {
      clearReadings();
      setStatus(data.status === "initializing" ? "Starting" : "Unavailable / stale", false);
      el("message").textContent = data.message || "Waiting for a fresh recorded segment.";
      return;
    }
    setStatus(data.status === "ok" ? "Replay active" : "Review prediction", true);
    el("sbp").textContent = data.prediction.SBP_mmHg.toFixed(1);
    el("dbp").textContent = data.prediction.DBP_mmHg.toFixed(1);
    el("segment").textContent = data.segment_id || "--";
    el("position").textContent = `${data.position} / ${data.total}`;
    el("cycle").textContent = String(data.cycle) + (data.loop_replay ? " (looped)" : "");
    el("sequence").textContent = String(data.sequence);
    el("updated").textContent = new Date(data.updated_at).toLocaleString();
    el("latency").textContent = `${data.inference_ms.toFixed(2)} ms`;
    el("hostname").textContent = data.hostname;
    el("message").textContent = data.message;
    if (lastSequence !== data.sequence) {
      drawWave("ecg-chart", data.waveforms.ecg, "#087f78");
      drawWave("ppg-chart", data.waveforms.ppg, "#4268b4");
      lastSequence = data.sequence;
    }
  } catch (error) {
    clearReadings();
    lastSequence = null;
    setStatus("Unavailable / disconnected", false);
    el("message").textContent = "Raspberry Pi service is unreachable. No current reading is shown.";
  } finally {
    clearTimeout(timer);
  }
}

window.addEventListener("resize", () => {lastSequence = null; refresh();});
clearReadings();
refresh();
setInterval(refresh, 1000);
