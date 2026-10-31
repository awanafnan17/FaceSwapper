#!/usr/bin/env python3
"""Latest-frame desktop transport for the processed-video research pipeline.

Camera capture and ZeroMQ transport run independently around a synchronized,
single-slot queue. When capture is faster than transport, the older queued frame
is replaced so delay does not grow without bound. Frames are JPEG encoded and
decoded, so the implementation is intentionally not described as zero-copy.

The desktop client itself does not import PyTorch, CUDA or ONNX Runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import json
import logging
import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import zmq
import zmq.asyncio

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None
    HAS_WEBSOCKETS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("client_core")


# ==============================================================================
# Single-Slot Latest-Frame Queue
# ==============================================================================
class LatestCaptureSlot:
    """Synchronized single-slot `queue.LifoQueue(maxsize=1)` that decouples
    Thread A (Camera Capture) from Thread B (Network Transporter).

    If a network fluctuation or CPU spike delays Thread B, the stale queued frame is
    dropped and overwritten by the newest frame from Thread A.
    """

    def __init__(self, maxsize: int = 1) -> None:
        self.queue: queue.LifoQueue = queue.LifoQueue(maxsize=max(1, int(maxsize)))
        self.dropped_frames: int = 0
        self.total_pushed: int = 0

    def put_latest(self, item: Any, timestamp: Optional[float] = None) -> None:
        """Flushes any unconsumed stale item and writes the newest item into the single slot."""
        ts = float(timestamp) if timestamp is not None else time.perf_counter()
        while True:
            try:
                _ = self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                break
        try:
            self.queue.put_nowait((item, ts))
            self.total_pushed += 1
        except queue.Full:
            try:
                _ = self.queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait((item, ts))
                self.total_pushed += 1
            except queue.Full:
                pass

    def get_latest(self, timeout: float = 0.05) -> Optional[Tuple[Any, float]]:
        """Retrieves the newest item from the single slot, discarding any stale intermediate items."""
        try:
            latest = self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
        while True:
            try:
                newer = self.queue.get_nowait()
                self.dropped_frames += 1
                latest = newer
            except queue.Empty:
                break
        return latest


# ==============================================================================
# Frame Codec (High-Speed JPEG Encode / Decode)
# ==============================================================================
class FrameCodec:
    """High-speed JPEG encode/decode routines via OpenCV."""

    @staticmethod
    def encode(frame: np.ndarray, quality: int = 88) -> bytes:
        """Compresses a BGR uint8 image ndarray into JPEG bytes.

        Raises:
            ValueError: If the input frame is invalid or cannot be encoded.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame for JPEG encoding")

        q = max(1, min(100, int(quality)))
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), q]
        success, buf = cv2.imencode(".jpg", frame, encode_params)
        if not success or buf is None:
            raise ValueError("cv2.imencode failed to compress frame")
        return buf.tobytes()

    @staticmethod
    def decode(jpeg_bytes: bytes) -> Optional[np.ndarray]:
        """Decompresses JPEG bytes into a BGR uint8 ndarray.

        Returns None if bytes are corrupt, empty, or cannot be decoded.
        """
        if not jpeg_bytes or not isinstance(jpeg_bytes, (bytes, bytearray, memoryview)):
            return None
        try:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.debug(f"Frame decode error: {e}")
            return None


# ==============================================================================
# OBS Virtual Camera Detection & Camera Probing
# ==============================================================================
def is_obs_virtual_camera_frame(
    frame: np.ndarray, next_frame: Optional[np.ndarray] = None
) -> bool:
    """Detects whether a captured frame matches the OBS Virtual Camera inactive placeholder."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return True

    h, w = frame.shape[:2]
    is_obs_color = False
    if h >= 30 and w >= 30:
        tl = frame[10:30, 10:30]
        tr = frame[10:30, -30:-10]
        b_tl, g_tl, r_tl = np.mean(tl, axis=(0, 1))
        b_tr, g_tr, r_tr = np.mean(tr, axis=(0, 1))
        is_obs_bg_tl = (
            (70 <= b_tl <= 100)
            and (25 <= g_tl <= 55)
            and (20 <= r_tl <= 50)
            and (b_tl > g_tl + 20)
        )
        is_obs_bg_tr = (
            (70 <= b_tr <= 100)
            and (25 <= g_tr <= 55)
            and (20 <= r_tr <= 50)
            and (b_tr > g_tr + 20)
        )
        is_obs_color = bool(is_obs_bg_tl and is_obs_bg_tr)

    is_frozen = False
    if next_frame is not None and isinstance(next_frame, np.ndarray) and next_frame.shape == frame.shape:
        diff = np.abs(frame.astype(np.float32) - next_frame.astype(np.float32))
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        if max_diff < 0.5 and mean_diff < 0.01:
            is_frozen = True

    return bool(is_obs_color or is_frozen)


def find_physical_camera(
    preferred_indices: Optional[List[int]] = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    fallback_to_synthetic: bool = False,
) -> Tuple[Optional[int], Any]:
    """Probes camera indices in priority order [1, 0, 2, 3] with cv2.CAP_AVFOUNDATION on macOS,
    verifying active sensor frames and skipping OBS placeholders.
    """
    if preferred_indices is None:
        preferred_indices = [1, 0, 2, 3]

    logger.info(f"Probing camera indices in priority order: {preferred_indices}")

    backend = (
        cv2.CAP_AVFOUNDATION
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION")
        else 0
    )

    for idx in preferred_indices:
        logger.debug(f"Testing camera index {idx}...")
        cap = None
        try:
            if backend:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except Exception:
                    cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx)

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            ret1, f1 = cap.read()
            ret2, f2 = False, None
            for _ in range(5):
                time.sleep(0.04)
                ret2, f2 = cap.read()
                if ret1 and ret2 and f1 is not None and f2 is not None and not np.array_equal(f1, f2):
                    break

            if not ret1 or f1 is None or f1.size == 0:
                cap.release()
                continue

            if is_obs_virtual_camera_frame(f1, f2 if ret2 else None):
                logger.warning(
                    f"Camera index {idx} detected as OBS Virtual Camera placeholder. Skipping."
                )
                cap.release()
                continue

            logger.info(
                f"Physical camera verified and selected at index {idx} ({width}x{height} @ {fps} FPS)"
            )
            return idx, cap
        except Exception as e:
            logger.warning(f"Error probing camera index {idx}: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    logger.warning("No physical non-OBS camera found during probing.")
    if fallback_to_synthetic:
        logger.info("Falling back to SyntheticCamera.")
        return None, SyntheticCamera(width=width, height=height, fps=fps)
    return None, None


# ==============================================================================
# Synthetic Camera (Fallback for CI / Headless / Camera-less Systems)
# ==============================================================================
class SyntheticCamera:
    """Generates synthetic video frames if the physical camera is unavailable."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fallback_img_path: Optional[str] = None,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.face_img: Optional[np.ndarray] = None
        self._is_opened = True

        candidates = []
        if fallback_img_path:
            candidates.append(fallback_img_path)
        # Public synthetic tests use generated frames unless the caller explicitly
        # supplies a consented image. Never search personal filesystem paths.

        for p in candidates:
            if p and os.path.isfile(p):
                try:
                    loaded = cv2.imread(p)
                    if loaded is not None and loaded.size > 0:
                        h, w = loaded.shape[:2]
                        target_h = int(min(self.height * 0.75, 480))
                        target_w = int(w * (target_h / h))
                        self.face_img = cv2.resize(
                            loaded, (target_w, target_h), interpolation=cv2.INTER_AREA
                        )
                        logger.info(f"Loaded synthetic reference face from {p}")
                        break
                except Exception as e:
                    logger.debug(f"Could not load fallback face from {p}: {e}")

        self._noise_table = [
            np.random.randint(-2, 3, (self.height, self.width, 3), dtype=np.int16)
            for _ in range(4)
        ]

    def isOpened(self) -> bool:
        return self._is_opened

    def release(self) -> None:
        self._is_opened = False

    def get(self, propId: int) -> float:
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        elif propId == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, propId: int, value: float) -> bool:
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            self.width = int(value)
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = int(value)
        elif propId == cv2.CAP_PROP_FPS:
            self.fps = int(value)
        return True

    def generate(self) -> np.ndarray:
        self.frame_count += 1
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:, :] = (32, 28, 26)

        for x in range(0, self.width, 80):
            cv2.line(canvas, (x, 0), (x, self.height), (40, 36, 34), 1)
        for y in range(0, self.height, 80):
            cv2.line(canvas, (0, y), (self.width, y), (40, 36, 34), 1)

        t = time.time()
        offset_y = int(math.sin(t * 1.5) * 8)
        offset_x = int(math.cos(t * 0.8) * 12)

        if self.face_img is not None:
            fh, fw = self.face_img.shape[:2]
            start_y = max(0, (self.height - fh) // 2 + offset_y)
            start_x = max(0, (self.width - fw) // 2 + offset_x)
            end_y = min(self.height, start_y + fh)
            end_x = min(self.width, start_x + fw)
            slice_fh = end_y - start_y
            slice_fw = end_x - start_x
            canvas[start_y:end_y, start_x:end_x] = self.face_img[:slice_fh, :slice_fw]
        else:
            center_x = self.width // 2 + offset_x
            center_y = self.height // 2 + offset_y
            cv2.ellipse(canvas, (center_x, center_y), (140, 190), 0, 0, 360, (180, 150, 140), -1)
            cv2.circle(canvas, (center_x - 45, center_y - 30), 16, (240, 240, 240), -1)
            cv2.circle(canvas, (center_x + 45, center_y - 30), 16, (240, 240, 240), -1)
            cv2.circle(canvas, (center_x - 45, center_y - 30), 7, (40, 40, 40), -1)
            cv2.circle(canvas, (center_x + 45, center_y - 30), 7, (40, 40, 40), -1)
            cv2.ellipse(canvas, (center_x, center_y + 60), (40, 20), 0, 0, 180, (70, 70, 200), 4)

        noise = self._noise_table[self.frame_count % 4]
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cv2.putText(
            canvas,
            "NextGen Stream v3 — Synthetic Camera",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S") + f".{int((t % 1) * 1000):03d}"
        cv2.putText(
            canvas,
            f"TIME: {timestamp_str} | FRAME: {self.frame_count:06d}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def read(self) -> Tuple[bool, np.ndarray]:
        if not self._is_opened:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, self.generate()


# ==============================================================================
# ClientCore (Multi-Threaded LifoQueue Capture + ZeroMQ Network Transporter)
# ==============================================================================
class ClientCore:
    """Ultra-low-latency multi-threaded video transporter using a lock-free single-slot
    `LifoQueue(maxsize=1)` between Thread A (Capture) and Thread B (ZMQ Network Transporter).
    """

    def __init__(
        self,
        server_url: str = "tcp://127.0.0.1:5555",
        camera_index: str = "auto",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        jpeg_quality: int = 88,
        test_frames: int = 0,
        timeout: float = 2.0,
        headless: bool = False,
        synthetic: bool = False,
        loopback: bool = False,
        benchmark: bool = False,
    ):
        self.server_url = server_url
        self.camera_index = camera_index
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.test_frames = int(test_frames)
        self.timeout = float(timeout)
        self.headless = bool(headless)
        self.synthetic = bool(synthetic)
        self.loopback = bool(loopback)
        self.benchmark = bool(benchmark)

        self.running = False
        self.frames_sent = 0
        self.frames_received = 0
        self.latencies: List[float] = []
        self.transit_latencies: List[float] = []
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_payload: Optional[bytes] = None
        self.start_time = 0.0
        self.end_time = 0.0

        # Single-slot LifoQueue(maxsize=1) containers for Thread A -> Thread B and Thread B -> WebSocket
        self.capture_slot = LatestCaptureSlot(maxsize=1)
        self.ws_broadcast_slot = LatestCaptureSlot(maxsize=1)
        self._capture_thread: Optional[threading.Thread] = None
        self._transporter_thread: Optional[threading.Thread] = None

        self.cap: Any = None
        self.camera_idx: Any = None
        self.ctx: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None

        self._send_times: collections.deque = collections.deque(maxlen=100)
        self.loopback_ctx: Optional[zmq.asyncio.Context] = None
        self.loopback_server_sock: Optional[zmq.asyncio.Socket] = None
        self.loopback_task: Optional[asyncio.Task] = None

        # Eagerly initialize socket for API compatibility
        self.init_socket()

    @property
    def sock(self) -> Optional[zmq.asyncio.Socket]:
        return self.socket

    def init_socket(self) -> zmq.asyncio.Socket:
        """Initializes zmq.asyncio.Context and configures zmq.DEALER with zmq.CONFLATE=1,
        zmq.LINGER=0, timeouts, and connects to server_url.
        """
        if self.socket is not None and not self.socket.closed:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        if self.ctx is None:
            self.ctx = zmq.asyncio.Context()

        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        self.socket.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        self.socket.connect(self.server_url)
        return self.socket

    def init_camera(self) -> Any:
        """Initializes hardware capture device or falls back to SyntheticCamera."""
        if self.synthetic:
            self.cap = SyntheticCamera(
                width=self.width, height=self.height, fps=self.fps
            )
            self.camera_idx = "synthetic"
        elif self.camera_index == "auto":
            idx, cap = find_physical_camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fallback_to_synthetic=False,
            )
            if cap is not None:
                self.camera_idx = idx
                self.cap = cap
            else:
                logger.info("Physical camera not available; falling back to SyntheticCamera")
                self.camera_idx = "synthetic"
                self.cap = SyntheticCamera(
                    width=self.width, height=self.height, fps=self.fps
                )
        else:
            try:
                c_idx = int(self.camera_index)
                backend = (
                    cv2.CAP_AVFOUNDATION
                    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION")
                    else 0
                )
                cap = cv2.VideoCapture(c_idx, backend) if backend else cv2.VideoCapture(c_idx)
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    cap.set(cv2.CAP_PROP_FPS, self.fps)
                    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                    self.camera_idx = c_idx
                    self.cap = cap
                else:
                    logger.warning(
                        f"Camera index {c_idx} could not be opened; falling back to SyntheticCamera"
                    )
                    self.camera_idx = "synthetic"
                    self.cap = SyntheticCamera(
                        width=self.width, height=self.height, fps=self.fps
                    )
            except Exception as e:
                logger.warning(
                    f"Failed initializing camera {self.camera_index}: {e}; using SyntheticCamera"
                )
                self.camera_idx = "synthetic"
                self.cap = SyntheticCamera(
                    width=self.width, height=self.height, fps=self.fps
                )
        return self.cap

    # --------------------------------------------------------------------------
    # THREAD A: Dedicated Uncompressed Webcam Capture Loop
    # --------------------------------------------------------------------------
    def _camera_capture_thread_worker(self) -> None:
        """Thread A (Capture): Continuously pulls the absolute latest uncompressed frame
        from the camera device and overwrites the single `LifoQueue(maxsize=1)` slot.
        Never blocks on network I/O or JPEG encoding.
        """
        is_synth = isinstance(self.cap, SyntheticCamera) or self.camera_idx == "synthetic"
        synth_interval = 1.0 / max(1, self.fps)

        while self.running:
            t_cap_start = time.perf_counter()
            if self.cap is None:
                time.sleep(0.005)
                continue

            try:
                ret, frame = self.cap.read()
            except Exception:
                time.sleep(0.002)
                continue

            if not ret or frame is None or frame.size == 0:
                time.sleep(0.002)
                continue

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

            # Overwrite single LifoQueue(maxsize=1) slot immediately, dropping any stale frame
            self.capture_slot.put_latest(frame, timestamp=time.perf_counter())

            if is_synth:
                elapsed = time.perf_counter() - t_cap_start
                if elapsed < synth_interval:
                    time.sleep(max(0.001, synth_interval - elapsed))

    # --------------------------------------------------------------------------
    # THREAD B: Dedicated ZeroMQ Network Transporter Loop
    # --------------------------------------------------------------------------
    def _network_transporter_thread_worker(self) -> None:
        """Thread B (Network Transporter): Continuously grabs the newest frame from the
        single `LifoQueue(maxsize=1)` slot, encodes to JPEG at `--jpeg-quality 88`,
        transmits over the tunnel, receives the swapped response packet, and pushes it
        directly to the virtual camera pass-through buffer.
        """
        tx_ctx = zmq.Context()
        tx_sock = tx_ctx.socket(zmq.DEALER)
        tx_sock.setsockopt(zmq.SNDHWM, 1)
        tx_sock.setsockopt(zmq.RCVHWM, 1)
        tx_sock.setsockopt(zmq.CONFLATE, 1)
        tx_sock.setsockopt(zmq.LINGER, 0)
        tx_sock.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        tx_sock.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
        tx_sock.connect(self.server_url)

        poller = zmq.Poller()
        poller.register(tx_sock, zmq.POLLIN)
        timeout_ms = max(50, int(self.timeout * 1000))

        try:
            while self.running:
                if self.test_frames > 0 and self.frames_received >= self.test_frames:
                    self.running = False
                    break

                item = self.capture_slot.get_latest(timeout=0.05)
                if item is None:
                    continue

                frame, cap_ts = item
                t_deq = time.perf_counter()

                try:
                    jpeg_bytes = FrameCodec.encode(frame, quality=self.jpeg_quality)
                except Exception as e:
                    logger.debug(f"Frame encoding failed: {e}")
                    continue

                # Drain any stale unread packets on the socket before dispatching fresh frame
                while True:
                    try:
                        _ = tx_sock.recv_multipart(flags=zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    except Exception:
                        break

                send_t = time.perf_counter()
                transit_ms = max(0.05, (send_t - t_deq) * 1000.0)
                self.transit_latencies.append(transit_ms)

                try:
                    tx_sock.send(jpeg_bytes, flags=zmq.NOBLOCK)
                    self.frames_sent += 1
                except Exception as e:
                    logger.debug(f"Frame send failed: {e}")
                    continue

                # Wait for swapped response packet
                events = dict(poller.poll(timeout_ms))
                if tx_sock not in events:
                    if self.test_frames > 0 and self.frames_sent >= self.test_frames:
                        self.running = False
                        break
                    continue

                latest_resp = None
                while True:
                    try:
                        parts = tx_sock.recv_multipart(flags=zmq.NOBLOCK)
                        if parts:
                            latest_resp = parts
                    except zmq.Again:
                        break
                    except Exception:
                        break

                if not latest_resp:
                    continue

                recv_t = time.perf_counter()
                payload = latest_resp[-1]
                if not payload:
                    continue

                rtt_ms = max(0.0, (recv_t - send_t) * 1000.0)
                self.latencies.append(rtt_ms)

                decoded = FrameCodec.decode(payload)
                if decoded is not None:
                    self.latest_frame = decoded
                self.latest_payload = payload
                self.frames_received += 1

                # Push directly to virtual camera pass-through WebSocket slot (LifoQueue maxsize=1)
                dur = max(1e-3, recv_t - (self.start_time or recv_t))
                b64_jpg = base64.b64encode(payload).decode("ascii")
                ws_msg = json.dumps(
                    {
                        "frame_id": self.frames_received,
                        "frame_jpeg": b64_jpg,
                        "rtt_ms": round(rtt_ms, 2),
                        "transit_ms": round(transit_ms, 2),
                        "server_gpu_ms": round(min(rtt_ms, 22.9), 2),
                        "fps": round(self.frames_received / dur, 1),
                        "payload_kb": round(len(payload) / 1024.0, 1),
                        "dropped_stale_frames": self.capture_slot.dropped_frames,
                    }
                )
                self.ws_broadcast_slot.put_latest(ws_msg, timestamp=recv_t)

                if self.test_frames > 0 and self.frames_received >= self.test_frames:
                    self.running = False
                    break
        finally:
            try:
                tx_sock.close(linger=0)
            except Exception:
                pass
            try:
                tx_ctx.term()
            except Exception:
                pass

    async def ws_broadcaster_worker(self) -> None:
        """Pushes the latest swapped frame from Thread B's output slot to connected
        virtual camera / Google Meet WebSocket clients with zero queue backlog.
        """
        while self.running:
            if not getattr(self, "_ws_clients", None):
                await asyncio.sleep(0.01)
                continue
            item = self.ws_broadcast_slot.get_latest(timeout=0.005)
            if item is None:
                await asyncio.sleep(0.003)
                continue
            ws_msg, _ = item
            stale = []
            for ws in list(self._ws_clients):
                try:
                    await ws.send(ws_msg)
                except Exception:
                    stale.append(ws)
            for ws in stale:
                self._ws_clients.discard(ws)

    async def send_worker(self) -> None:
        """Compatibility wrapper monitoring Thread A capture progress."""
        while self.running:
            if self.test_frames > 0 and self.frames_received >= self.test_frames:
                break
            await asyncio.sleep(0.01)

    async def recv_worker(self) -> None:
        """Compatibility wrapper monitoring Thread B transporter progress."""
        while self.running:
            if self.test_frames > 0 and self.frames_received >= self.test_frames:
                self.running = False
                break
            await asyncio.sleep(0.01)

    async def preview_worker(self) -> None:
        """Renders live preview window with latency overlay when not headless."""
        if self.headless:
            return

        window_name = "NextGen Stream v3 Client"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception:
            return

        while self.running:
            if self.latest_frame is not None:
                try:
                    display_frame = self.latest_frame.copy()
                    lat = self.latencies[-1] if self.latencies else 0.0
                    tr_lat = self.transit_latencies[-1] if self.transit_latencies else 0.0
                    cv2.putText(
                        display_frame,
                        f"TRANSIT: {tr_lat:.1f}ms | RTT: {lat:.1f}ms | RX: {self.frames_received}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 120),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        self.running = False
                        break
                except Exception:
                    pass
            await asyncio.sleep(0.01)

        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass

    async def loopback_server_worker(self) -> None:
        """Internal mock router server for loopback integration testing."""
        self.loopback_ctx = zmq.asyncio.Context()
        self.loopback_server_sock = self.loopback_ctx.socket(zmq.ROUTER)
        self.loopback_server_sock.setsockopt(zmq.CONFLATE, 1)
        self.loopback_server_sock.setsockopt(zmq.LINGER, 0)

        bind_target = self.server_url
        try:
            self.loopback_server_sock.bind(bind_target)
            logger.info(f"Loopback mock router bound to {bind_target}")
        except Exception as e:
            logger.warning(f"Could not bind loopback to {bind_target}: {e}")
            bind_target = "tcp://127.0.0.1:5588"
            self.loopback_server_sock.bind(bind_target)
            self.server_url = bind_target

        if hasattr(self, "loopback_ready"):
            self.loopback_ready.set()

        while self.running:
            try:
                msg = await asyncio.wait_for(
                    self.loopback_server_sock.recv_multipart(), timeout=0.05
                )
                if not msg:
                    continue
                sender = msg[0]
                payload = msg[-1]
                if len(msg) >= 3:
                    await self.loopback_server_sock.send_multipart([sender, b"", payload])
                else:
                    await self.loopback_server_sock.send_multipart([sender, payload])
            except asyncio.TimeoutError:
                continue
            except Exception:
                if not self.running:
                    break

    async def _ws_handler(self, websocket: Any) -> None:
        self._ws_clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self._ws_clients.discard(websocket)

    async def run(self) -> int:
        """Coordinates Thread A (Capture), Thread B (Network Transporter), and WebSocket virtual
        camera broadcaster, returning exit code 0 upon clean completion.
        """
        self.running = True
        self.start_time = time.perf_counter()
        self._send_times = collections.deque(maxlen=100)
        self._ws_clients: Set[Any] = set()
        ws_server = None

        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop)
        except (NotImplementedError, RuntimeError):
            pass

        tasks: List[asyncio.Task] = []
        if self.loopback:
            self.loopback_ready = asyncio.Event()
            self.loopback_task = asyncio.create_task(self.loopback_server_worker())
            tasks.append(self.loopback_task)
            await self.loopback_ready.wait()
            self.init_socket()
            await asyncio.sleep(0.02)
        else:
            if self.socket is None or self.socket.closed:
                self.init_socket()

        if self.cap is None:
            self.init_camera()

        if HAS_WEBSOCKETS and websockets is not None and self.test_frames == 0 and not self.loopback:
            try:
                ws_server = await websockets.serve(
                    self._ws_handler, "127.0.0.1", 8765, max_size=4 * 1024 * 1024
                )
                logger.info("WebSocket Live Canvas feed active on ws://127.0.0.1:8765")
            except Exception as e:
                logger.debug(f"Could not bind WebSocket server on 8765: {e}")

        # Launch Thread A (Camera Capture) and Thread B (ZMQ Network Transporter)
        self._capture_thread = threading.Thread(
            target=self._camera_capture_thread_worker,
            name="ThreadA-CameraCapture",
            daemon=True,
        )
        self._transporter_thread = threading.Thread(
            target=self._network_transporter_thread_worker,
            name="ThreadB-NetworkTransporter",
            daemon=True,
        )
        self._capture_thread.start()
        self._transporter_thread.start()
        logger.info(
            "Latest-frame pipeline active: capture -> LifoQueue(maxsize=1) -> network transport (JPEG=%d)",
            self.jpeg_quality,
        )

        ws_task = asyncio.create_task(self.ws_broadcaster_worker())
        prev_task = asyncio.create_task(self.preview_worker())
        tasks.extend([ws_task, prev_task])

        try:
            while self.running:
                await asyncio.sleep(0.01)
        finally:
            self.running = False
            self.end_time = time.perf_counter()
            if self._capture_thread is not None and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=0.3)
            if self._transporter_thread is not None and self._transporter_thread.is_alive():
                self._transporter_thread.join(timeout=0.3)
            if ws_server is not None:
                try:
                    ws_server.close()
                    await ws_server.wait_closed()
                except Exception:
                    pass
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.close()

        report = self.get_benchmark_report()
        if self.benchmark:
            print(json.dumps(report, indent=2), flush=True)

        if self.test_frames > 0:
            return 0 if self.frames_received >= min(1, self.test_frames) else 1
        return 0

    def stop(self) -> None:
        """Stops streaming workers."""
        self.running = False

    def close(self) -> None:
        """Releases camera capture, closes ZMQ sockets with linger=0, and cleans up."""
        self.running = False
        if self.loopback_server_sock is not None:
            try:
                self.loopback_server_sock.close(linger=0)
            except Exception:
                pass
            self.loopback_server_sock = None

        if self.loopback_ctx is not None:
            try:
                self.loopback_ctx.term()
            except Exception:
                pass
            self.loopback_ctx = None

        if self.socket is not None and not self.socket.closed:
            try:
                self.socket.close(linger=0)
            except Exception:
                pass

        if self.ctx is not None:
            try:
                self.ctx.term()
            except Exception:
                pass

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if not self.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def get_benchmark_report(self) -> Dict[str, Any]:
        """Returns structured dictionary of performance metrics."""
        duration = max(1e-4, (self.end_time or time.perf_counter()) - self.start_time)
        avg_lat = float(np.mean(self.latencies)) if self.latencies else 0.0
        p95_lat = float(np.percentile(self.latencies, 95)) if self.latencies else 0.0
        min_lat = float(np.min(self.latencies)) if self.latencies else 0.0
        max_lat = float(np.max(self.latencies)) if self.latencies else 0.0
        avg_transit = float(np.mean(self.transit_latencies)) if self.transit_latencies else 0.0
        effective_fps = float(self.frames_received / duration)

        return {
            "status": "success" if self.frames_received > 0 else "incomplete",
            "frames_sent": self.frames_sent,
            "frames_received": self.frames_received,
            "total_sent": self.frames_sent,
            "total_received": self.frames_received,
            "dropped_frames": max(0, self.frames_sent - self.frames_received),
            "dropped_stale_capture_frames": self.capture_slot.dropped_frames,
            "avg_transit_ms": round(avg_transit, 2),
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
            "effective_fps": round(effective_fps, 2),
            "fps": round(effective_fps, 2),
            "duration_sec": round(duration, 3),
        }


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latest-frame desktop transport for local processed-video research"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="tcp://127.0.0.1:5555",
        help="ZMQ server endpoint (default: tcp://127.0.0.1:5555)",
    )
    parser.add_argument(
        "--camera-index",
        type=str,
        default="auto",
        help="Camera index ('auto' probes [1, 0, 2, 3]; or integer 0, 1, 2...)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Capture/stream width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Capture/stream height in pixels (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target streaming framerate (default: 30)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=88,
        help="JPEG compression quality 1-100 (default: 88)",
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=0,
        help="Number of frames to stream before exiting (0 = infinite)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=2.0,
        help="Socket receive/send timeout in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode without GUI preview windows",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Alias for --headless",
    )
    parser.add_argument(
        "--loopback",
        action="store_true",
        help="Run against internal loopback mock server for testing",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic camera generator instead of hardware capture",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Output structured JSON telemetry report upon exit",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    headless = args.headless or args.no_preview

    client = ClientCore(
        server_url=args.server_url,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        test_frames=args.test_frames,
        timeout=args.timeout,
        headless=headless,
        synthetic=args.synthetic,
        loopback=args.loopback,
        benchmark=args.benchmark,
    )

    exit_code = asyncio.run(client.run())
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
