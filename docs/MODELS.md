# Model audit: what actually runs

## 1. Recommended: single-image InSwapper

Launch the server with `--source /absolute/path/to/consented-source.jpg`. A detected source face is converted into the recognition embedding used by the swapper. This path is identity transfer with a pretrained model; it is not training a new model from one image.

The source of truth is `vendor/Deep-Live-Cam/modules/processors/frame/face_swapper.py:get_face_swapper`. It chooses `inswapper_128_fp16.onnx` when present and otherwise `inswapper_128.onnx`, then calls `insightface.model_zoo.get_model`. The model's face input is 128×128 and identity input is a 512-dimensional embedding. The server obtains `bgr_fake` and its affine matrix with `paste_back=False`, then performs its own compositing.

The single-image mode is preferred for this repository because it removes the need to manage a personal training dataset and personalized model. There is no controlled benchmark proving it best for every lighting condition, head pose or identity.

## 2. Detection, identity and landmarks

`get_face_analyser` constructs `insightface.app.FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition', 'landmark_2d_106'])`. The actual detector comes from that pack. A function named `_scrfd_cascade_detect` is not evidence that the `scrfd_2.5g` file is loaded. Historical server comments used that name; the reviewed runtime invokes `fa.det_model.detect` from the configured pack.

FaceFusion's `yolo_face` path is a fallback detector. Source face selection and live target tracking are separate operations. The source image must contain a usable face; multi-person reference images are ambiguous and should be cropped before use.

## 3. Masking and restoration

| Component | Role | Activation |
|---|---|---|
| XSeg `xseg_1` | Occlusion mask to preserve objects in front of the face | Normal pipeline; optional CUDA-specific file preferred when present |
| BiSeNet ResNet-34 | Semantic face-region mask | `--bisenet-region` |
| GFPGAN 1.4 ONNX / FP16 | Optional aligned face restoration | `--onnx-gfpgan` plus available weights |
| Feathered box mask | Smooth face boundary | Always computed |
| Masked Reinhard transfer | Match crop color statistics with luminance clamping | Applied before compositing |

The native swap crop is mapped into a 256×256 FFHQ composition space. Optional GFPGAN operates with a 512×512 alignment and blends its result at 0.25. Upsampling/restoration does not change InSwapper's native inference resolution.

## 4. Removed private and experimental branches

The source archive also contained personalized training material and a private DFM workflow. Those assets and the automatic multi-image reference search are not part of this public repository. The public server requires one explicit `--source` image, making the identity input visible and deliberate on every run.

## 5. Names that are not the selected model

- **HyperSwap 1a 256:** present in historical text and FaceFusion's generic model catalog, but the reviewed server's swapper loader uses Deep-Live-Cam's InSwapper implementation.
- **LivePortrait/GhostFace:** a separate local prototype contained classes with these names, implemented with procedural NumPy keypoints, pooled pixel features and image warps. Those experimental approximations are not included in this repository and must not be described as pretrained LivePortrait/GhostFace integration.
- **“Zero glitch,” “zero dropout,” and fixed latency:** names and development ambitions, not guarantees. Some failure branches return the input frame; tracking can also preserve stale geometry.

## Weights and provenance

Runtime weights are downloaded or supplied separately. The source contains download helpers and public model catalogs, but weights are not covered automatically by the repository's code license. Obtain each model from its recognized publisher and follow its terms. No private facial embeddings, training data or personalized weights are distributed.
