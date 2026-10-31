(function () {
  "use strict";

  if (window.top !== window || document.getElementById("processed-camera-research-preview")) {
    return;
  }

  const panel = document.createElement("aside");
  panel.id = "processed-camera-research-preview";
  panel.setAttribute("aria-label", "Processed camera research preview");
  panel.innerHTML = [
    '<header class="pcrp-header">',
    '<div><strong>Processed video preview</strong><span>Consent-based local research</span></div>',
    '<button type="button" class="pcrp-toggle" aria-expanded="true" aria-label="Collapse processed video preview">−</button>',
    "</header>",
    '<div class="pcrp-body">',
    '<div class="pcrp-frame">',
    '<img alt="Processed camera frame. This output is digitally modified." hidden />',
    '<div class="pcrp-empty">Waiting for the local processed-frame stream…</div>',
    '<div class="pcrp-disclosure">DIGITALLY PROCESSED OUTPUT</div>',
    "</div>",
    '<div class="pcrp-status" role="status"><span class="pcrp-dot"></span><span class="pcrp-state">Connecting to localhost…</span></div>',
    '<dl class="pcrp-telemetry"><div><dt>Round trip</dt><dd>—</dd></div><div><dt>GPU</dt><dd>—</dd></div><div><dt>Stream</dt><dd>—</dd></div></dl>',
    '<p class="pcrp-note">Preview only. This extension does not replace, rename or conceal a camera device.</p>',
    "</div>",
  ].join("");
  document.documentElement.appendChild(panel);

  const body = panel.querySelector(".pcrp-body");
  const toggle = panel.querySelector(".pcrp-toggle");
  const image = panel.querySelector("img");
  const empty = panel.querySelector(".pcrp-empty");
  const state = panel.querySelector(".pcrp-state");
  const dot = panel.querySelector(".pcrp-dot");
  const telemetry = panel.querySelectorAll(".pcrp-telemetry dd");

  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") === "true";
    toggle.setAttribute("aria-expanded", String(!expanded));
    toggle.setAttribute(
      "aria-label",
      expanded ? "Expand processed video preview" : "Collapse processed video preview"
    );
    toggle.textContent = expanded ? "+" : "−";
    body.hidden = expanded;
  });

  function formatMilliseconds(value) {
    return Number.isFinite(value) && value >= 0 ? value.toFixed(1) + " ms" : "—";
  }

  function showFrame(message) {
    if (!message.data || !message.data.startsWith("data:image/")) return;
    image.src = message.data;
    empty.hidden = true;
    image.hidden = false;
    state.textContent = "Receiving local processed frames";
    dot.classList.add("is-live");
    telemetry[0].textContent = formatMilliseconds(message.rtt_ms);
    telemetry[1].textContent = formatMilliseconds(message.server_gpu_ms);
    telemetry[2].textContent =
      Number.isFinite(message.fps) && message.fps > 0 ? message.fps.toFixed(1) + " fps" : "—";
  }

  function connect() {
    let port;
    try {
      port = chrome.runtime.connect({ name: "processed-camera-preview" });
    } catch (_) {
      state.textContent = "Extension relay unavailable";
      return;
    }

    port.onMessage.addListener((message) => {
      if (!message) return;
      if (message.type === "PROCESSED_FRAME") showFrame(message);
      if (message.type === "STREAM_OPEN") {
        state.textContent = "Connected; waiting for a processed frame";
        dot.classList.add("is-ready");
      }
    });

    port.onDisconnect.addListener(() => {
      state.textContent = "Local stream disconnected";
      dot.classList.remove("is-live", "is-ready");
    });
  }

  connect();
})();
