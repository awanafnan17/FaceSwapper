"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const relay = require("./background.js");

test("binary telemetry packets are parsed without changing the JPEG", () => {
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
  const packet = Buffer.alloc(relay.HEADER_SIZE + jpeg.length);
  packet.writeUInt16BE(relay.HEADER_MAGIC, 0);
  packet.writeUInt16BE(3, 2);
  packet.writeUInt32BE(42, 4);
  packet.writeFloatBE(18.5, 8);
  packet.writeFloatBE(7.25, 12);
  packet.writeFloatBE(29.97, 16);
  packet.writeUInt16BE(320, 20);
  packet.writeUInt16BE(180, 22);
  jpeg.copy(packet, relay.HEADER_SIZE);

  const parsed = relay.parseBinaryHeaderFrame(packet);
  assert.equal(parsed.frame_id, 42);
  assert.equal(parsed.roi_mode, true);
  assert.equal(parsed.occluded, true);
  assert.equal(parsed.roi_w, 320);
  assert.equal(parsed.roi_h, 180);
  assert.deepEqual(Buffer.from(parsed.jpegBuffer), jpeg);
});

test("manifest is limited to local web and websocket origins", () => {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(__dirname, "manifest.json"), "utf8")
  );
  const origins = [
    ...manifest.host_permissions,
    ...manifest.content_scripts.flatMap((script) => script.matches),
  ];
  assert(origins.length > 0);
  assert(origins.every((origin) => /^(?:http|ws):\/\/(?:localhost|127\.0\.0\.1)\//.test(origin)));
});

test("public preview contains no camera identity or API concealment hooks", () => {
  const sourceFiles = ["preview.js", "background.js"]
    .map((file) => fs.readFileSync(path.join(__dirname, file), "utf8"))
    .join("\n");
  const forbidden = [
    "enumerateDevices",
    "getUserMedia",
    "Function.prototype.toString",
    "MediaStreamTrack",
    "hardwareId",
    "vendorId",
    "productId",
    "[native code]",
  ];
  for (const token of forbidden) {
    assert.equal(sourceFiles.includes(token), false, "unexpected concealment token: " + token);
  }
  assert(sourceFiles.includes("DIGITALLY PROCESSED OUTPUT"));
  assert(sourceFiles.includes("does not replace, rename or conceal a camera device"));
});
