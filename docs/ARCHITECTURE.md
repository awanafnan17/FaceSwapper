# Architecture and design decisions

## Desktop transport

`ClientCore` keeps camera acquisition separate from network work. `LatestCaptureSlot` wraps a bounded queue; replacing older frames prevents a slow network/GPU from creating an ever-growing backlog. Despite historical “lock-free” language, Python's `queue` uses synchronization internally. JPEG encoding and decoding also copy data, so this is not a fully zero-copy pipeline.

The local client captures at a requested resolution/frame rate, encodes JPEG (default quality 88), sends it through a ZeroMQ DEALER connection and receives a processed JPEG. Requested camera settings are not guarantees that every device will honor them. The capture loop drops stale frames, trading temporal completeness for freshness.

The client supports a desktop preview and a local WebSocket broadcast at `ws://127.0.0.1:8765`. A generated-image loopback exercises the same transport without inference. The default capture path uses macOS AVFoundation when available; other platforms fall back to OpenCV's default backend.

## GPU worker

The server warms CUDA through PyTorch, loads the archived runtime modules, configures ONNX Runtime and prepares model sessions. It captures ZeroMQ messages using a ROUTER socket, retains the latest payload and decodes the JPEG. Public defaults bind only to loopback.

For each target frame:

1. Detect a face using the configured InsightFace pack; use FaceFusion's detector fallback when needed.
2. Select the target by spatial continuity rather than always choosing the leftmost face.
3. Track upper-face anchors with optical flow and smooth rigid pose with a causal One Euro filter.
4. Warp the target into aligned crop coordinates; run InSwapper using the cached source identity.
5. Transform the 128-pixel swap into the 256-pixel composition crop.
6. Combine box/occlusion/optional region masks and apply color transfer.
7. Optionally restore the aligned crop with GFPGAN.
8. Inverse-warp the crop and blend it back into the original frame; optionally preserve the target mouth.
9. Encode at JPEG quality 90 and return the frame.

## Tracking behavior and failure modes

The tracker is intended to reduce jumps during motion or partial occlusion. Its `never_drop` setting can retain estimated geometry; it is not an identity guarantee. The processing function returns the original frame when no usable target or swap crop is found. Do not rely on the pipeline to guarantee continuous concealment, identity consistency or error-free tracking.

Very dark scenes, extreme poses, small faces, motion blur, multiple people and foreground objects can degrade results. Restoration can alter identity detail. Preserving the original mouth trades visual consistency for clearer articulation. Evaluate on consented examples in the actual environment.

## Network boundary

Raw ZeroMQ TCP has no built-in application authentication or encryption in this implementation. Keep the server on loopback and use an authenticated SSH tunnel for a remote worker. The local WebSocket endpoint is also unauthenticated; do not forward it to the public internet. A public multi-user deployment would require authentication, origin checks, rate limits, payload limits, session isolation and a clear data-retention design.

## Browser preview

The optional Manifest V3 extension is deliberately a **preview, not a camera adapter**. Its service worker connects to the loopback WebSocket and passes decoded frames to a content script on localhost or 127.0.0.1. The content script renders a floating panel that permanently identifies the image as digitally processed and exposes connection/latency telemetry.

The public extension does not override `getUserMedia`, enumerate or rename devices, modify `MediaStreamTrack`, conceal JavaScript functions, synthesize a hardware identity or run on third-party sites. It cannot act as an operating-system virtual camera and is intentionally unsuitable for interview or identity-verification services.

## Public release changes

- Preserved current single-image server/client source and the archived compatible runtime source.
- Restricted the public model path to one explicit, consented `--source` image and loopback binding.
- Replaced machine-specific runtime roots and personal asset paths with configuration.
- Removed automatic searches for personal reference photos and the personalized-model branch.
- Replaced archived camera/device interception with a visibly labelled, localhost-only preview.
- Corrected misleading model and performance descriptions in the public documentation.
- Excluded personal images, video, personalized weights, model binaries, histories and credentials.
