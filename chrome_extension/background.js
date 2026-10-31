/**
 * Local MV3 service-worker relay for the disclosed processed-frame preview.
 *
 * Connects only to ws://127.0.0.1:8765 and relays video frames plus optional
 * timing telemetry to the localhost preview content script.
 *
 * Supports:
 *  1) 24-byte C-struct binary header (magic = 0x5546, "!HHIfffHH") + JPEG payload
 *  2) Raw JPEG ArrayBuffer (0xFF 0xD8)
 *  3) JSON text {frame_jpeg, rtt_ms, server_gpu_ms, frame_id, fps, roi_mode, occluded}
 */
"use strict";

const DEFAULT_WS_URL = "ws://127.0.0.1:8765";
const HEADER_MAGIC = 0x5546;
const HEADER_SIZE = 24;

const activePorts = new Set();
let ws = null;
let reconnectTimer = null;
const MIN_RECONNECT_DELAY = 50;
const MAX_RECONNECT_DELAY = 8000;
const BACKOFF_FACTOR = 1.5;
let reconnectDelay = MIN_RECONNECT_DELAY;

let latestRttMs = 80.0;
let latestServerGpuMs = 0.0;
let latestFps = 30.0;

function log(...args) {
  console.log("[ProcessedCameraPreview:SW]", ...args);
}

function resetReconnectBackoff() {
  reconnectDelay = MIN_RECONNECT_DELAY;
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  const delay = reconnectDelay;
  reconnectDelay = Math.min(MAX_RECONNECT_DELAY, Math.round(reconnectDelay * BACKOFF_FACTOR));
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connectWebSocket();
  }, delay);
}

function arrayBufferToBase64(buffer) {
  if (typeof Buffer !== "undefined") {
    return Buffer.from(buffer).toString("base64");
  }
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  const CHUNK_SIZE = 0x8000;
  for (let i = 0; i < len; i += CHUNK_SIZE) {
    const chunk = bytes.subarray(i, Math.min(i + CHUNK_SIZE, len));
    binary += String.fromCharCode.apply(null, chunk);
  }
  if (typeof btoa === "function") {
    return btoa(binary);
  }
  return binary;
}

/**
 * Parses the 24-byte C-struct binary telemetry header ("!HHIfffHH"):
 *   0..2:   uint16 magic (0x5546)
 *   2..4:   uint16 flags (bit 0 = roi_mode, bit 1 = occluded)
 *   4..8:   uint32 frame_seq_id
 *   8..12:  float32 rtt_ms
 *  12..16:  float32 server_gpu_ms
 *  16..20:  float32 fps
 *  20..22:  uint16 roi_w
 *  22..24:  uint16 roi_h
 *  24..end: JPEG bytes
 */
function parseBinaryHeaderFrame(buffer) {
  const ab =
    buffer instanceof ArrayBuffer
      ? buffer
      : ArrayBuffer.isView(buffer)
      ? buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
      : null;
  if (!ab || ab.byteLength <= HEADER_SIZE) return null;

  const view = new DataView(ab);
  const magic = view.getUint16(0, false);
  if (magic !== HEADER_MAGIC) return null;

  const flags = view.getUint16(2, false);
  const frameSeqId = view.getUint32(4, false);
  const rttMs = view.getFloat32(8, false);
  const serverGpuMs = view.getFloat32(12, false);
  const fpsVal = view.getFloat32(16, false);
  const roiW = view.getUint16(20, false);
  const roiH = view.getUint16(22, false);
  const jpegBuffer = ab.slice(HEADER_SIZE);

  if (Number.isFinite(rttMs) && rttMs >= 0) latestRttMs = rttMs;
  if (Number.isFinite(serverGpuMs) && serverGpuMs >= 0) latestServerGpuMs = serverGpuMs;
  if (Number.isFinite(fpsVal) && fpsVal >= 0) latestFps = fpsVal;

  return {
    isBinaryHeader: true,
    flags,
    roi_mode: Boolean(flags & 1),
    occluded: Boolean(flags & 2),
    frame_id: frameSeqId,
    rtt_ms: latestRttMs,
    server_gpu_ms: latestServerGpuMs,
    fps: latestFps,
    roi_w: roiW,
    roi_h: roiH,
    jpegBuffer,
    dataUrl: "data:image/jpeg;base64," + arrayBufferToBase64(jpegBuffer),
  };
}

function parseIncomingWsData(raw) {
  if (typeof raw === "string") {
    if (raw.startsWith("{")) {
      try {
        const parsed = JSON.parse(raw);
        const rawJpeg = parsed.frame_jpeg || parsed.data || "";
        const dataUrl = rawJpeg.startsWith("data:")
          ? rawJpeg
          : rawJpeg
          ? "data:image/jpeg;base64," + rawJpeg
          : null;
        if (typeof parsed.rtt_ms === "number" && !Number.isNaN(parsed.rtt_ms)) {
          latestRttMs = parsed.rtt_ms;
        }
        if (typeof parsed.server_gpu_ms === "number" && !Number.isNaN(parsed.server_gpu_ms)) {
          latestServerGpuMs = parsed.server_gpu_ms;
        }
        if (typeof parsed.fps === "number" && !Number.isNaN(parsed.fps)) {
          latestFps = parsed.fps;
        }
        return {
          dataUrl,
          rtt_ms: latestRttMs,
          server_gpu_ms: latestServerGpuMs,
          fps: latestFps,
          frame_id: parsed.frame_id || 0,
          roi_mode: Boolean(parsed.roi_mode),
          occluded: Boolean(parsed.occluded),
        };
      } catch (_) {}
    }
    const dataUrl = raw.startsWith("data:") ? raw : "data:image/jpeg;base64," + raw;
    return {
      dataUrl,
      rtt_ms: latestRttMs,
      server_gpu_ms: latestServerGpuMs,
      fps: latestFps,
      frame_id: 0,
      roi_mode: false,
      occluded: false,
    };
  }

  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) {
    const headerPacket = parseBinaryHeaderFrame(raw);
    if (headerPacket) return headerPacket;

    const buf =
      raw instanceof ArrayBuffer
        ? raw
        : raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    return {
      dataUrl: "data:image/jpeg;base64," + arrayBufferToBase64(buf),
      rtt_ms: latestRttMs,
      server_gpu_ms: latestServerGpuMs,
      fps: latestFps,
      frame_id: 0,
      roi_mode: false,
      occluded: false,
    };
  }
  return null;
}

function formatFrameDataUrl(data) {
  const parsed = parseIncomingWsData(data);
  return parsed ? parsed.dataUrl : null;
}

function broadcastFrame(framePacket) {
  if (!framePacket || activePorts.size === 0) return;
  const msg =
    typeof framePacket === "string"
      ? {
          type: "PROCESSED_FRAME",
          data: framePacket,
          rtt_ms: latestRttMs,
          server_gpu_ms: latestServerGpuMs,
          fps: latestFps,
        }
      : {
          type: "PROCESSED_FRAME",
          data: framePacket.dataUrl,
          rtt_ms: typeof framePacket.rtt_ms === "number" ? framePacket.rtt_ms : latestRttMs,
          server_gpu_ms:
            typeof framePacket.server_gpu_ms === "number"
              ? framePacket.server_gpu_ms
              : latestServerGpuMs,
          fps: typeof framePacket.fps === "number" ? framePacket.fps : latestFps,
          frame_id: framePacket.frame_id || 0,
          roi_mode: Boolean(framePacket.roi_mode),
          occluded: Boolean(framePacket.occluded),
        };

  if (!msg.data) return;

  for (const port of [...activePorts]) {
    try {
      port.postMessage(msg);
    } catch (_) {
      activePorts.delete(port);
    }
  }
}

let latestFrameSeq = 0;

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  try {
    ws = new WebSocket(DEFAULT_WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = function () {
      resetReconnectBackoff();
      for (const port of activePorts) {
        try {
          port.postMessage({ type: "STREAM_OPEN", url: DEFAULT_WS_URL });
        } catch (_) {}
      }
    };

    ws.onmessage = function (event) {
      if (activePorts.size === 0 || !event || !event.data) return;
      const packet = parseIncomingWsData(event.data);
      if (packet && packet.dataUrl) {
        packet.seq = ++latestFrameSeq;
        broadcastFrame(packet);
      }
    };

    ws.onerror = function () {};

    ws.onclose = function () {
      ws = null;
      if (activePorts.size > 0) {
        scheduleReconnect();
      }
    };
  } catch (_) {
    scheduleReconnect();
  }
}

if (typeof chrome !== "undefined" && chrome.runtime && chrome.runtime.onConnect) {
  chrome.runtime.onConnect.addListener((port) => {
    if (port.name === "processed-camera-preview") {
      activePorts.add(port);
      connectWebSocket();

      port.onDisconnect.addListener(() => {
        activePorts.delete(port);
      });
    }
  });
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    DEFAULT_WS_URL,
    HEADER_MAGIC,
    HEADER_SIZE,
    MIN_RECONNECT_DELAY,
    MAX_RECONNECT_DELAY,
    BACKOFF_FACTOR,
    connectWebSocket,
    parseBinaryHeaderFrame,
    parseIncomingWsData,
    formatFrameDataUrl,
    arrayBufferToBase64,
    broadcastFrame,
    scheduleReconnect,
    resetReconnectBackoff,
    getReconnectDelay: () => reconnectDelay,
    activePorts,
  };
}
