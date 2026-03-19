"""Lip ROI extraction and preprocessing."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    mp = None
    _MEDIAPIPE_IMPORT_ERROR = exc
else:
    _MEDIAPIPE_IMPORT_ERROR = None


LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

REFERENCE_EYES = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
    ],
    dtype=np.float32,
)


@dataclass
class PreprocessStats:
    success: int = 0
    failed: int = 0
    skipped: int = 0


def _mediapipe_install_hint() -> str:
    return (
        "MediaPipe FaceMesh is unavailable. On Linux this is usually a "
        "mediapipe/protobuf compatibility issue. Reinstall with:\n"
        "  python -m pip uninstall -y mediapipe protobuf\n"
        "  python -m pip install 'protobuf==3.20.3' 'mediapipe==0.10.9'\n"
        "Then verify with:\n"
        "  python -c \"import sys, mediapipe as mp; print(sys.executable); "
        "print(mp.__version__); "
        "print(hasattr(mp, 'solutions')); "
        "from mediapipe.python.solutions.face_mesh import FaceMesh; print('ok')\""
    )


def _resolve_face_mesh_class():
    if mp is None:  # pragma: no cover
        raise ImportError(_mediapipe_install_hint()) from _MEDIAPIPE_IMPORT_ERROR

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "face_mesh"):
        return solutions.face_mesh.FaceMesh

    import_errors: list[Exception] = []

    try:
        from mediapipe.python.solutions.face_mesh import FaceMesh

        return FaceMesh
    except Exception as exc:  # pragma: no cover
        import_errors.append(exc)

    try:
        from mediapipe.python.solutions import face_mesh as face_mesh_module

        return face_mesh_module.FaceMesh
    except Exception as exc:  # pragma: no cover
        import_errors.append(exc)

    raise ImportError(
        f"{_mediapipe_install_hint()}\n"
        f"mediapipe module path: {getattr(mp, '__file__', '<unknown>')}\n"
        f"last import error: {import_errors[-1] if import_errors else _MEDIAPIPE_IMPORT_ERROR}"
    )


class LipROIExtractor:
    """Extract stabilized grayscale lip crops from a video."""

    def __init__(self, target_size: int = 96, canvas_multiplier: int = 4) -> None:
        face_mesh_class = _resolve_face_mesh_class()

        self.target_size = target_size
        self.canvas_multiplier = canvas_multiplier
        self.canvas_size = target_size * canvas_multiplier
        self.mouth_center = np.array(
            [self.canvas_size // 2, int(self.canvas_size * 0.55)],
            dtype=np.int32,
        )
        scale = target_size / 96.0
        self.canonical_eyes = REFERENCE_EYES * canvas_multiplier * scale
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        self.face_mesh = face_mesh_class(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._last_transform: np.ndarray | None = None
        self._failure_streak = 0

    def close(self) -> None:
        self.face_mesh.close()

    def _landmark_px(self, landmarks, index: int, width: int, height: int) -> np.ndarray:
        point = landmarks[index]
        return np.array([point.x * width, point.y * height], dtype=np.float32)

    @staticmethod
    def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
        if src.shape != (2, 2) or dst.shape != (2, 2):
            return None

        src_center = src.mean(axis=0)
        dst_center = dst.mean(axis=0)
        src_vec = src[1] - src[0]
        dst_vec = dst[1] - dst[0]
        src_norm = float(np.linalg.norm(src_vec))
        dst_norm = float(np.linalg.norm(dst_vec))
        if src_norm < 1e-6 or dst_norm < 1e-6:
            return None

        scale = dst_norm / src_norm
        src_angle = math.atan2(float(src_vec[1]), float(src_vec[0]))
        dst_angle = math.atan2(float(dst_vec[1]), float(dst_vec[0]))
        theta = dst_angle - src_angle
        rotation = scale * np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)],
            ],
            dtype=np.float32,
        )
        translation = dst_center - rotation @ src_center
        return np.concatenate([rotation, translation[:, None]], axis=1)

    def _compute_transform(self, frame: np.ndarray) -> np.ndarray | None:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = (
            self._landmark_px(landmarks, LEFT_EYE_INNER, width, height)
            + self._landmark_px(landmarks, LEFT_EYE_OUTER, width, height)
        ) / 2.0
        right_eye = (
            self._landmark_px(landmarks, RIGHT_EYE_INNER, width, height)
            + self._landmark_px(landmarks, RIGHT_EYE_OUTER, width, height)
        ) / 2.0
        src = np.stack([left_eye, right_eye], axis=0)
        return self._similarity_transform(src, self.canonical_eyes)

    def _compute_transform_with_fallback(self, frame: np.ndarray) -> np.ndarray | None:
        transform = self._compute_transform(frame)
        if transform is not None:
            return transform

        if self._failure_streak <= 3:
            upsampled = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            upsampled_transform = self._compute_transform(upsampled)
            if upsampled_transform is not None:
                adjusted = upsampled_transform.copy()
                adjusted[:, :2] *= 2.0
                return adjusted
        return None

    def _crop_from_transform(self, frame: np.ndarray, transform: np.ndarray) -> np.ndarray:
        warped = cv2.warpAffine(
            frame,
            transform,
            (self.canvas_size, self.canvas_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        half = self.target_size // 2
        x_center, y_center = int(self.mouth_center[0]), int(self.mouth_center[1])
        x1, x2 = x_center - half, x_center + half
        y1, y2 = y_center - half, y_center + half
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.canvas_size, x2), min(self.canvas_size, y2)
        crop = warped[y1:y2, x1:x2]
        if crop.shape[:2] != (self.target_size, self.target_size):
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def extract(self, video_path: str | Path) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {video_path}")

        frames: list[np.ndarray] = []
        self._last_transform = None
        self._failure_streak = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            transform = self._compute_transform_with_fallback(frame)
            if transform is None:
                self._failure_streak += 1
                if self._last_transform is None:
                    frames.append(np.zeros((self.target_size, self.target_size), dtype=np.uint8))
                    continue
                transform = self._last_transform
            else:
                self._failure_streak = 0
                self._last_transform = transform

            frames.append(self._crop_from_transform(frame, transform))

        capture.release()
        if not frames:
            raise ValueError(f"No frames extracted from {video_path}")
        return np.stack(frames, axis=0).astype(np.float16) / 255.0


def mirror_video_to_roi_path(video_path: str | Path, split_root: str | Path, output_root: str | Path) -> Path:
    video_path = Path(video_path)
    split_root = Path(split_root)
    output_root = Path(output_root)
    return output_root / video_path.relative_to(split_root).with_suffix(".npy")


def preprocess_split(
    data_root: str | Path,
    output_root: str | Path,
    split: str,
    target_size: int = 96,
    skip_existing: bool = True,
) -> PreprocessStats:
    split_root = Path(data_root) / split
    roi_root = Path(output_root) / split
    roi_root.mkdir(parents=True, exist_ok=True)

    extractor = LipROIExtractor(target_size=target_size)
    stats = PreprocessStats()
    videos = sorted(split_root.rglob("*.mp4"))

    try:
        for video_path in tqdm(videos, desc=f"Preprocess {split}", unit="video"):
            roi_path = mirror_video_to_roi_path(video_path, split_root, roi_root)
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            if skip_existing and roi_path.exists():
                stats.skipped += 1
                continue

            try:
                roi = extractor.extract(video_path)
                np.save(str(roi_path), roi)
                stats.success += 1
            except Exception:
                np.save(
                    str(roi_path),
                    np.zeros((1, target_size, target_size), dtype=np.float16),
                )
                stats.failed += 1
    finally:
        extractor.close()

    stats_path = roi_root / "_stats.json"
    stats_path.write_text(json.dumps(asdict(stats), indent=2))
    return stats
