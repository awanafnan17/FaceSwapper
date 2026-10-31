# Third-party source and model notices

This repository includes integration code and an archived runtime snapshot. Attribution is retained so the portfolio distinguishes engineering integration from underlying model authorship.

| Component | Provenance | Included terms |
|---|---|---|
| Deep-Live-Cam runtime | `hacksider/Deep-Live-Cam`, metadata version 2.1.5 | `vendor/Deep-Live-Cam/LICENSE` (GNU GPLv3) |
| FaceFusion runtime | `facefusion/facefusion`, metadata version 3.9.1 | `vendor/official_facefusion/LICENSE.md` (OpenRAIL-AS notice) |
| InsightFace / InSwapper | Referenced by the runtime; model weights not included | Follow the code and each model publisher's separate terms |
| GFPGAN | Optional ONNX restoration interface; weights not included | Follow the original project and model terms |

The archived FaceFusion license file contains a short OpenRAIL-AS copyright notice rather than the complete use-condition text. Consult the upstream distribution for the complete applicable terms before redistributing its components or deploying them. The root GPLv3 file does not replace separate FaceFusion or model conditions.

The vendor snapshot was obtained from the owner's project backup; original Git commit IDs were not present. Versions come from included metadata. It may contain local modifications and should not be represented as an unmodified upstream release. Model binaries, training media and personalized weights are not redistributed.

Public snapshot modifications are documented in `docs/ARCHITECTURE.md`. A narrowly annotated secret-scanner false positive in the FaceFusion model-name type list refers to public `corridor_key_*` model identifiers, not credentials.
