# Validation and limits

The public snapshot was reviewed on 2026-10-31.

## Checks performed

- Python files parsed/compiled without importing the CUDA/model stack.
- JavaScript parsed with Node's syntax checker.
- Extension tests verified binary packet parsing, localhost-only manifest origins, disclosure text, and the absence of camera enumeration, media interception, hardware-identity or function-concealment hooks.
- Client synthetic loopback: 30 generated 320×240 frames sent and 30 responses received using Python 3.14.6.
- Gitleaks scanned the source with redacted output. The public FaceFusion model identifiers `corridor_key_*` are an annotated false positive, not a credential.
- Source inventory checked for private photos, video, model binaries, database dumps, environment files, keys and machine-specific personal paths.

The loopback timing measures a local generated-image echo path, not model inference or end-to-end video performance. It must not be advertised as GPU latency or a real-world frame-rate guarantee.

### Measured local echo result

Test machine: Apple M2 MacBook Air, 8 GB RAM, macOS 27.0.

| Metric | Result |
|---|---:|
| Frames sent / received | 30 / 30 |
| Average client-reported latency | 0.64 ms |
| P95 client-reported latency | 1.04 ms |
| Minimum / maximum | 0.31 / 1.06 ms |
| Effective loopback rate | 25.20 FPS |

The numbers exclude webcam capture, remote networking, CUDA, face detection, InSwapper, masks, restoration and screen presentation.

## Not verified in this release

- Fresh installation of the full GPU environment and every optional model.
- CUDA performance, VRAM needs, long-duration stability or visual-quality comparisons.
- A physical webcam on Windows/Linux or all macOS devices.
- End-to-end GPU latency or compatibility across CUDA/driver combinations.
- Browser use beyond the intentionally limited local preview.
- Security readiness for internet-exposed or multi-user operation.

Future work: reproducible GPU dependency lock, model provenance manifest, measured consented evaluation set, authenticated transport, browser integration tests, explicit degraded-output indicators and platform-specific setup verification.
