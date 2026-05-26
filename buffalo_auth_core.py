from __future__ import annotations

import random
import shutil
import sqlite3
import importlib.util
import site
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.exists():
    site.addsitedir(str(LOCAL_DEPS))

LOCAL_INSIGHTFACE_SOURCE = Path(__file__).resolve().parent / ".vendor" / "insightface"
if LOCAL_INSIGHTFACE_SOURCE.exists():
    site.addsitedir(str(LOCAL_INSIGHTFACE_SOURCE))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
except ImportError:
    FaceAnalysis = None
    face_align = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LIVENESS_CHECKPOINT = PROJECT_ROOT / "face_liveness_vit" / "model.pt"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "app_data" / "auth.db"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "app_data" / "uploads"
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "app_data" / "profiles"
DEFAULT_INSIGHTFACE_ROOT = PROJECT_ROOT / ".insightface"
BUNDLED_BUFFALO_DIR = PROJECT_ROOT / "buffalo"
DEFAULT_HAAR_EYE_CASCADE = LOCAL_DEPS / "cv2" / "data" / "haarcascade_eye.xml"


class _SimpleCompose:
    def __init__(self, funcs):
        self.funcs = funcs

    def __call__(self, image):
        for func in self.funcs:
            image = func(image)
        return image


def _resize_short_side(image: Image.Image, target_size: int) -> Image.Image:
    width, height = image.size
    if width == 0 or height == 0:
        raise RuntimeError("Encountered an empty image while resizing")
    scale = target_size / min(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _center_crop(image: Image.Image, crop_size: int) -> Image.Image:
    width, height = image.size
    left = max(0, (width - crop_size) // 2)
    top = max(0, (height - crop_size) // 2)
    right = left + min(crop_size, width)
    bottom = top + min(crop_size, height)
    image = image.crop((left, top, right, bottom))
    if image.size != (crop_size, crop_size):
        image = image.resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    return image


def _to_grayscale_rgb(image: Image.Image) -> Image.Image:
    return image.convert("L").convert("RGB")


def _to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


@dataclass
class AuthConfig:
    liveness_checkpoint: Path = DEFAULT_LIVENESS_CHECKPOINT
    database_path: Path = DEFAULT_DATABASE_PATH
    upload_dir: Path = DEFAULT_UPLOAD_DIR
    profile_dir: Path = DEFAULT_PROFILE_DIR
    insightface_model_name: str = "buffalo_l"
    insightface_root: str | Path | None = DEFAULT_INSIGHTFACE_ROOT
    det_size: tuple[int, int] = (448, 448)
    scan_frame_stride: int = 3
    max_detected_faces: int = 6
    max_embedding_faces: int = 3
    min_detected_faces: int = 2
    random_seed: int = 42
    liveness_spoof_threshold: float = 0.32
    min_live_ratio: float = 0.40
    match_threshold: float = 0.55
    min_match_ratio: float = 0.40
    use_median_score: bool = True
    challenge_min_frames: int = 3
    challenge_min_open_eye_ratio: float = 0.06
    challenge_close_ratio: float = 0.90
    challenge_reopen_ratio: float = 0.55
    challenge_pose_min_frames: int = 4
    challenge_turn_threshold: float = 0.08
    registration_allow_update: bool = False


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Encoder(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=2, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)


class LivenessViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=2,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=d_model)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, d_model))
        self.pos_drop = nn.Dropout(dropout)
        self.encoder = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers, mlp_ratio=mlp_ratio, dropout=dropout)
        self.head = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.pos_drop(x)
        x = self.encoder(x)
        return self.head(x[:, 0])


class AuthEngine:
    def __init__(self, config: AuthConfig | None = None):
        self.config = config or AuthConfig()
        self.config.liveness_checkpoint = Path(self.config.liveness_checkpoint)
        self.config.database_path = Path(self.config.database_path)
        self.config.upload_dir = Path(self.config.upload_dir)
        self.config.profile_dir = Path(self.config.profile_dir)
        self.config.upload_dir.mkdir(parents=True, exist_ok=True)
        self.config.profile_dir.mkdir(parents=True, exist_ok=True)
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = self._get_device()
        self.face_app: FaceAnalysis | None = None
        self.liveness_model: LivenessViT | None = None
        self.liveness_transform: Any | None = None
        self.eye_cascade = None
        self._init_db()

    def _get_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        return device

    def _insightface_root_path(self) -> Path | None:
        if self.config.insightface_root is None:
            return None
        return Path(self.config.insightface_root).expanduser()

    @staticmethod
    def _missing_runtime_dependencies() -> list[str]:
        missing: list[str] = []
        if FaceAnalysis is None or face_align is None:
            missing.append("insightface")
        if importlib.util.find_spec("onnxruntime") is None:
            missing.append("onnxruntime")
        return missing

    def _ensure_local_insightface_pack(self, root_path: Path) -> None:
        model_dir = root_path / "models" / self.config.insightface_model_name
        if self.config.insightface_model_name == "buffalo_l" and BUNDLED_BUFFALO_DIR.exists():
            model_dir.mkdir(parents=True, exist_ok=True)
            for source_path in BUNDLED_BUFFALO_DIR.glob("*.onnx"):
                target_path = model_dir / source_path.name
                if not target_path.exists():
                    shutil.copy2(source_path, target_path)

    @staticmethod
    def _resolve_eye_cascade_path() -> Path:
        if cv2 is not None and hasattr(cv2, "data"):
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
            if cascade_path.exists():
                return cascade_path
        return DEFAULT_HAAR_EYE_CASCADE

    def _ensure_eye_cascade(self):
        if cv2 is None:
            raise RuntimeError("Blink challenge verification requires opencv-python-headless to be installed.")
        if self.eye_cascade is None:
            cascade_path = self._resolve_eye_cascade_path()
            if not cascade_path.exists():
                raise RuntimeError("OpenCV eye cascade file was not found; blink verification cannot start.")
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if cascade.empty():
                raise RuntimeError("OpenCV eye cascade failed to load; blink verification cannot start.")
            self.eye_cascade = cascade
        return self.eye_cascade

    def _init_db(self) -> None:
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profile_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    frame_index INTEGER,
                    crop_path TEXT,
                    embedding_dim INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    username TEXT,
                    ip_address TEXT,
                    detail TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS unknown_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_kind TEXT NOT NULL,
                    input_path TEXT NOT NULL,
                    run_dir TEXT NOT NULL,
                    preview_crop_path TEXT,
                    live_ratio REAL,
                    mean_spoof REAL,
                    session_score REAL,
                    match_ratio REAL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _ensure_models(self) -> None:
        missing = self._missing_runtime_dependencies()
        if missing:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                "Authentication runtime dependencies are not installed. "
                f"Missing: {missing_str}. "
                "Install them before using registration or authentication."
            )

        if self.face_app is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            kwargs: dict[str, Any] = {"name": self.config.insightface_model_name, "providers": providers}
            root_path = self._insightface_root_path()
            if root_path is not None:
                self._ensure_local_insightface_pack(root_path)
                kwargs["root"] = str(root_path)
            self.face_app = FaceAnalysis(**kwargs)
            self.face_app.prepare(ctx_id=0 if self.device.type == "cuda" else -1, det_size=self.config.det_size)

        if self.liveness_model is None or self.liveness_transform is None:
            model = LivenessViT(
                img_size=224,
                patch_size=16,
                d_model=128,
                nhead=4,
                num_layers=2,
                num_classes=2,
            ).to(self.device)
            checkpoint = torch.load(self.config.liveness_checkpoint, map_location=self.device, weights_only=False)
            state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self.liveness_model = model
            self.liveness_transform = _SimpleCompose(
                [
                    lambda image: _resize_short_side(image, int(224 * 1.14)),
                    lambda image: _center_crop(image, 224),
                    _to_grayscale_rgb,
                    _to_normalized_tensor,
                ]
            )
        self._ensure_eye_cascade()

    @staticmethod
    def _pick_largest_face(faces):
        if not faces:
            return None
        return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm == 0:
            raise RuntimeError("Encountered zero-norm embedding")
        return embedding / norm

    @staticmethod
    def _get_aligned_face(frame_bgr, face_obj, out_size=224):
        if hasattr(face_obj, "kps") and face_obj.kps is not None:
            aligned_bgr = face_align.norm_crop(frame_bgr, landmark=face_obj.kps, image_size=out_size)
            aligned_rgb = aligned_bgr[:, :, ::-1]
            return Image.fromarray(aligned_rgb)

        x1, y1, x2, y2 = face_obj.bbox.astype(int)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            return None
        crop_rgb = crop_bgr[:, :, ::-1]
        crop_image = Image.fromarray(crop_rgb)
        crop_image = crop_image.resize((out_size, out_size), Image.Resampling.BILINEAR)
        crop_rgb = np.asarray(crop_image)
        return Image.fromarray(crop_rgb)

    def _extract_face_record(self, frame_bgr, frame_idx: int, crops_dir: Path) -> dict[str, Any] | None:
        assert self.face_app is not None
        faces = self.face_app.get(frame_bgr)
        face = self._pick_largest_face(faces)
        if face is None:
            return None

        aligned = self._get_aligned_face(frame_bgr, face, out_size=224)
        if aligned is None or not hasattr(face, "embedding") or face.embedding is None:
            return None

        crop_path = crops_dir / f"frame_{frame_idx:06d}.jpg"
        aligned.save(crop_path, quality=95)
        return {
            "frame_index": frame_idx,
            "crop_path": str(crop_path),
            "face_image": aligned,
            # FaceAnalysis already returns the recognition embedding from buffalo_l.
            "embedding": self._normalize_embedding(face.embedding),
        }

    def _estimate_eye_openness(self, frame_bgr) -> float | None:
        self._ensure_models()
        eye_cascade = self._ensure_eye_cascade()
        assert self.face_app is not None

        faces = self.face_app.get(frame_bgr)
        face = self._pick_largest_face(faces)
        if face is None:
            return None

        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        face_roi = frame_bgr[y1:y2, x1:x2]
        if face_roi.size == 0:
            return None

        upper_half = face_roi[: max(1, face_roi.shape[0] // 2), :]
        gray = cv2.cvtColor(upper_half, cv2.COLOR_BGR2GRAY)
        min_eye = max(12, min(gray.shape[:2]) // 8)
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(min_eye, min_eye),
        )
        if len(eyes) == 0:
            return 0.0

        ranked_eyes = sorted(eyes, key=lambda rect: rect[2] * rect[3], reverse=True)
        selected: list[tuple[int, int, int, int]] = []
        for rect in ranked_eyes:
            rx, _, rw, _ = rect
            center_x = rx + (rw / 2)
            if all(abs(center_x - (other[0] + (other[2] / 2))) > (0.2 * gray.shape[1]) for other in selected):
                selected.append(tuple(int(v) for v in rect))
            if len(selected) == 2:
                break
        if not selected:
            return 0.0

        openness_values = [rect[3] / max(rect[2], 1) for rect in selected]
        return float(sum(openness_values) / len(openness_values))

    def _verify_blink_challenge(self, frame_paths: list[Path], required_blinks: int) -> dict[str, Any]:
        if len(frame_paths) < self.config.challenge_min_frames:
            raise RuntimeError(
                f"Blink challenge needs at least {self.config.challenge_min_frames} frames; only {len(frame_paths)} were captured."
            )

        openness_values: list[float] = []
        for frame_path in frame_paths:
            frame_bgr = cv2.imread(str(frame_path)) if cv2 is not None else None
            if frame_bgr is None:
                continue
            openness = self._estimate_eye_openness(frame_bgr)
            if openness is not None:
                openness_values.append(float(openness))

        if len(openness_values) < self.config.challenge_min_frames:
            raise RuntimeError("Could not verify a blink from the live frame sequence. Keep the face centered and try again.")

        baseline = max(openness_values)
        if baseline < self.config.challenge_min_open_eye_ratio:
            raise RuntimeError("Blink verification was inconclusive because the eyes were not clear enough in the capture.")

        close_threshold = baseline * self.config.challenge_close_ratio
        reopen_threshold = baseline * self.config.challenge_reopen_ratio
        blink_count = 0
        closed_run = 0
        in_closed_state = False

        for openness in openness_values:
            if openness <= close_threshold:
                closed_run += 1
                in_closed_state = True
                continue
            if in_closed_state and closed_run >= 1 and openness >= reopen_threshold:
                blink_count += 1
                closed_run = 0
                in_closed_state = False
                continue
            if openness >= reopen_threshold:
                closed_run = 0
                in_closed_state = False

        verified = blink_count >= required_blinks
        return {
            "action": "blink",
            "count": int(blink_count),
            "required": int(required_blinks),
            "verified": bool(verified),
            "valid_frames": len(openness_values),
            "baseline": float(baseline),
            "close_threshold": float(close_threshold),
            "reopen_threshold": float(reopen_threshold),
        }

    def _estimate_pose_turn_score(self, frame_bgr) -> float | None:
        self._ensure_models()
        assert self.face_app is not None

        faces = self.face_app.get(frame_bgr)
        face = self._pick_largest_face(faces)
        if face is None or not hasattr(face, "kps") or face.kps is None:
            return None

        kps = np.asarray(face.kps, dtype=np.float32)
        if kps.shape[0] < 5:
            return None

        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        eye_mid_x = float((left_eye[0] + right_eye[0]) / 2.0)
        eye_distance = float(abs(right_eye[0] - left_eye[0]))
        if eye_distance <= 1e-6:
            return None
        # Positive means nose shifts toward subject's right side in the image.
        return float((nose[0] - eye_mid_x) / eye_distance)

    def _verify_pose_challenge(self, frame_paths: list[Path], action: str) -> dict[str, Any]:
        scores: list[float] = []
        for frame_path in frame_paths:
            frame_bgr = cv2.imread(str(frame_path)) if cv2 is not None else None
            if frame_bgr is None:
                continue
            score = self._estimate_pose_turn_score(frame_bgr)
            if score is not None:
                scores.append(float(score))

        if len(scores) < self.config.challenge_pose_min_frames:
            raise RuntimeError("Could not verify enough face poses from the live frame sequence. Keep your face visible and try again.")

        peak_left = abs(min(scores))
        peak_right = max(scores)
        if action == "turn_left":
            achieved = peak_left
            verified = peak_left >= self.config.challenge_turn_threshold
        else:
            achieved = peak_right
            verified = peak_right >= self.config.challenge_turn_threshold

        return {
            "action": action,
            "count": 1 if verified else 0,
            "required": 1,
            "verified": bool(verified),
            "valid_frames": len(scores),
            "peak_left": float(peak_left),
            "peak_right": float(peak_right),
            "achieved": float(achieved),
        }

    def _collect_face_records(self, video_path: Path, run_dir: Path) -> list[dict[str, Any]]:
        self._ensure_models()
        assert self.face_app is not None
        if cv2 is None:
            raise RuntimeError("Video processing requires opencv-python-headless to be installed.")

        crops_dir = run_dir / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        records: list[dict[str, Any]] = []
        frame_idx = 0
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                if frame_idx % self.config.scan_frame_stride != 0:
                    frame_idx += 1
                    continue

                record = self._extract_face_record(frame_bgr, frame_idx, crops_dir)
                if record is not None:
                    records.append(record)
                frame_idx += 1
                if len(records) >= self.config.max_detected_faces:
                    break
        finally:
            cap.release()

        return records

    def _collect_face_records_from_images(self, image_files: list[Any], run_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
        self._ensure_models()
        assert self.face_app is not None

        frames_dir = run_dir / "frames"
        crops_dir = run_dir / "crops"
        frames_dir.mkdir(parents=True, exist_ok=True)
        crops_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        frame_paths: list[Path] = []
        for frame_idx, image_file in enumerate(image_files):
            suffix = Path(getattr(image_file, "filename", "") or f"frame_{frame_idx:06d}.jpg").suffix or ".jpg"
            image_path = frames_dir / f"frame_{frame_idx:06d}{suffix}"
            image_file.save(image_path)
            frame_paths.append(image_path)
            try:
                frame_rgb = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            frame_bgr = np.asarray(frame_rgb)[:, :, ::-1].copy()
            record = self._extract_face_record(frame_bgr, frame_idx, crops_dir)
            if record is not None:
                records.append(record)
            if len(records) >= self.config.max_detected_faces:
                break

        return records, frame_paths

    @torch.no_grad()
    def _run_liveness(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_models()
        assert self.liveness_model is not None
        assert self.liveness_transform is not None

        spoof_probs: list[float] = []
        live_flags: list[bool] = []
        for record in records:
            x = self.liveness_transform(record["face_image"]).unsqueeze(0).to(self.device)
            logits = self.liveness_model(x)
            probs = torch.softmax(logits, dim=1)
            spoof_prob = float(probs[0, 1].item())
            is_live = spoof_prob < self.config.liveness_spoof_threshold
            record["spoof_prob"] = spoof_prob
            record["is_live"] = is_live
            spoof_probs.append(spoof_prob)
            live_flags.append(is_live)

        if not spoof_probs:
            raise RuntimeError("No face crops available for liveness inference")

        mean_spoof = float(sum(spoof_probs) / len(spoof_probs))
        live_ratio = float(sum(live_flags) / len(live_flags))
        session_live = (mean_spoof < self.config.liveness_spoof_threshold) and (live_ratio >= self.config.min_live_ratio)
        return {
            "mean_spoof": mean_spoof,
            "live_ratio": live_ratio,
            "session_live": bool(session_live),
        }

    def _build_authenticity_assessment(self, liveness: dict[str, Any]) -> dict[str, Any]:
        session_live = bool(liveness["session_live"])
        mean_spoof = float(liveness["mean_spoof"])
        live_ratio = float(liveness["live_ratio"])
        human_confidence = max(0.0, min(1.0, min(1.0 - mean_spoof, live_ratio)))
        suspicion_confidence = max(0.0, min(1.0, max(mean_spoof, 1.0 - live_ratio)))
        if session_live:
            return {
                "classification": "human",
                "label": "Likely Human",
                "short_label": "HUMAN",
                "flagged": False,
                "message": (
                    "The session passed liveness screening and looks consistent with a real human capture."
                ),
                "confidence_hint": human_confidence,
                "mean_spoof": mean_spoof,
                "live_ratio": live_ratio,
            }
        return {
            "classification": "ai_suspected",
            "label": "AI / Deepfake Suspected",
            "short_label": "SUSPECTED",
            "flagged": True,
            "message": (
                "The session failed liveness screening and should be treated as suspicious. It may be a spoof, replay, or AI-generated deepfake attempt."
            ),
            "confidence_hint": suspicion_confidence,
            "mean_spoof": mean_spoof,
            "live_ratio": live_ratio,
        }

    def _validate_challenge(self, frame_paths: list[Path], challenge: dict[str, Any] | None) -> dict[str, Any]:
        if not challenge:
            raise RuntimeError("Missing capture challenge. Refresh the page and try again.")
        action = str(challenge.get("action", "blink")).strip().lower()
        if action == "blink":
            try:
                required_blinks = max(1, int(challenge.get("required_blinks", 1)))
            except (TypeError, ValueError):
                required_blinks = 1
            result = self._verify_blink_challenge(frame_paths, required_blinks)
            if not result["verified"]:
                raise RuntimeError(
                    f"Blink verification failed. Please blink at least {result['required']} time(s) during live capture."
                )
            return result
        if action in {"turn_left", "turn_right"}:
            result = self._verify_pose_challenge(frame_paths, action)
            if not result["verified"]:
                direction_label = "left" if action == "turn_left" else "right"
                raise RuntimeError(f"Head-turn verification failed. Please turn your face to the {direction_label} during capture.")
            return result
        raise RuntimeError("Unsupported capture challenge. Refresh the page and try again.")

    def log_security_event(
        self,
        event_type: str,
        outcome: str,
        username: str | None = None,
        ip_address: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._init_db()
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute(
                """
                INSERT INTO audit_events (event_type, outcome, username, ip_address, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (event_type, outcome, username, ip_address, detail, now),
            )
            conn.commit()

    def list_audit_events(self, limit: int = 100) -> list[dict[str, Any]]:
        self._init_db()
        safe_limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self.config.database_path) as conn:
            rows = conn.execute(
                """
                SELECT id, event_type, outcome, username, ip_address, detail, created_at
                FROM audit_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "event_type": row[1],
                "outcome": row[2],
                "username": row[3],
                "ip_address": row[4],
                "detail": row[5],
                "created_at": row[6],
            }
            for row in rows
        ]

    def record_unknown_attempt(
        self,
        *,
        input_kind: str,
        input_path: str,
        run_dir: str,
        preview_crop_path: str | None,
        live_ratio: float,
        mean_spoof: float,
        session_score: float,
        match_ratio: float,
    ) -> int:
        self._init_db()
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.config.database_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO unknown_attempts (
                    input_kind, input_path, run_dir, preview_crop_path,
                    live_ratio, mean_spoof, session_score, match_ratio, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    input_kind,
                    input_path,
                    run_dir,
                    preview_crop_path,
                    float(live_ratio),
                    float(mean_spoof),
                    float(session_score),
                    float(match_ratio),
                    now,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def list_unknown_attempts(self, limit: int = 100) -> list[dict[str, Any]]:
        self._init_db()
        safe_limit = max(1, min(int(limit), 500))
        with sqlite3.connect(self.config.database_path) as conn:
            rows = conn.execute(
                """
                SELECT id, input_kind, input_path, run_dir, preview_crop_path,
                       live_ratio, mean_spoof, session_score, match_ratio, created_at
                FROM unknown_attempts
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "input_kind": row[1],
                "input_path": row[2],
                "run_dir": row[3],
                "preview_crop_path": row[4],
                "live_ratio": float(row[5]) if row[5] is not None else None,
                "mean_spoof": float(row[6]) if row[6] is not None else None,
                "session_score": float(row[7]) if row[7] is not None else None,
                "match_ratio": float(row[8]) if row[8] is not None else None,
                "created_at": row[9],
            }
            for row in rows
        ]

    def _sample_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        live_records = [record for record in records if record.get("is_live")]
        candidates = live_records if live_records else records
        if len(candidates) < self.config.min_detected_faces:
            raise RuntimeError(
                f"Only {len(candidates)} usable face crops available; need at least {self.config.min_detected_faces}"
            )
        sample_count = min(self.config.max_embedding_faces, len(candidates))
        rng = random.Random(self.config.random_seed)
        return rng.sample(candidates, sample_count)

    def _make_run_dir(self, prefix: str, username: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_username = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in username).strip("_") or "user"
        run_dir = self.config.upload_dir / f"{prefix}_{safe_username}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _save_video(self, file_storage, run_dir: Path) -> Path:
        suffix = Path(getattr(file_storage, "filename", "upload.mp4") or "upload.mp4").suffix or ".mp4"
        video_path = run_dir / f"input{suffix}"
        file_storage.save(video_path)
        return video_path

    def _prepare_submission(
        self,
        username: str,
        mode: str,
        file_storage=None,
        image_files: list[Any] | None = None,
    ) -> tuple[Path, list[dict[str, Any]], str, str, list[Path]]:
        run_dir = self._make_run_dir(mode, username)
        webcam_frames = [image_file for image_file in (image_files or []) if getattr(image_file, "filename", "")]
        if file_storage is not None and getattr(file_storage, "filename", ""):
            input_path = self._save_video(file_storage, run_dir)
            records = self._collect_face_records(input_path, run_dir)
            input_kind = "video"
            frame_paths: list[Path] = []
        elif webcam_frames:
            records, frame_paths = self._collect_face_records_from_images(webcam_frames, run_dir)
            input_path = run_dir / "frames"
            input_kind = "webcam"
        else:
            raise RuntimeError("Upload a video or capture webcam frames.")

        return run_dir, records, str(input_path), input_kind, frame_paths

    def _upsert_user(self, username: str) -> int:
        self._init_db()
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.config.database_path) as conn:
            row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if row:
                if not self.config.registration_allow_update:
                    raise RuntimeError(
                        f"Username '{username}' is already registered. Use a different username or add a dedicated update flow."
                    )
                conn.execute("UPDATE users SET updated_at = ? WHERE id = ?", (now, row[0]))
                conn.commit()
                return int(row[0])
            cursor = conn.execute(
                "INSERT INTO users (username, created_at, updated_at) VALUES (?, ?, ?)",
                (username, now, now),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def _store_profile_embeddings(self, user_id: int, sampled_records: list[dict[str, Any]]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.config.database_path) as conn:
            conn.execute("DELETE FROM profile_embeddings WHERE user_id = ?", (user_id,))
            for record in sampled_records:
                emb = np.asarray(record["embedding"], dtype=np.float32)
                conn.execute(
                    """
                    INSERT INTO profile_embeddings (
                        user_id, frame_index, crop_path, embedding_dim, embedding_blob, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        int(record["frame_index"]),
                        str(record["crop_path"]),
                        int(emb.shape[0]),
                        sqlite3.Binary(emb.tobytes()),
                        now,
                    ),
                )
            conn.commit()

    def _load_profile_embeddings(self, username: str) -> tuple[int, np.ndarray]:
        self._init_db()
        with sqlite3.connect(self.config.database_path) as conn:
            user_row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if not user_row:
                raise RuntimeError(f"No registered user found for '{username}'")
            rows = conn.execute(
                "SELECT embedding_dim, embedding_blob FROM profile_embeddings WHERE user_id = ? ORDER BY id",
                (user_row[0],),
            ).fetchall()
        if not rows:
            raise RuntimeError(f"No stored embeddings found for '{username}'")
        embeddings = []
        for dim, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != dim:
                raise RuntimeError("Stored embedding has inconsistent dimension")
            embeddings.append(emb)
        gallery = np.stack(embeddings, axis=0)
        gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        return int(user_row[0]), gallery.astype(np.float32)

    def _load_registered_galleries(self) -> list[dict[str, Any]]:
        self._init_db()
        with sqlite3.connect(self.config.database_path) as conn:
            rows = conn.execute(
                """
                SELECT u.id, u.username, pe.embedding_dim, pe.embedding_blob
                FROM users u
                JOIN profile_embeddings pe ON pe.user_id = u.id
                ORDER BY u.username, pe.id
                """
            ).fetchall()

        galleries_by_user: dict[int, dict[str, Any]] = {}
        for user_id, username, dim, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            if emb.size != dim:
                raise RuntimeError(f"Stored embedding has inconsistent dimension for '{username}'")

            user_key = int(user_id)
            bucket = galleries_by_user.setdefault(
                user_key,
                {
                    "user_id": user_key,
                    "username": username,
                    "embeddings": [],
                },
            )
            bucket["embeddings"].append(emb)

        galleries: list[dict[str, Any]] = []
        for bucket in galleries_by_user.values():
            gallery = np.stack(bucket["embeddings"], axis=0)
            gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
            galleries.append(
                {
                    "user_id": bucket["user_id"],
                    "username": bucket["username"],
                    "gallery": gallery.astype(np.float32),
                }
            )

        galleries.sort(key=lambda item: item["username"].lower())
        return galleries

    def _score_identity_candidates(
        self,
        video_embeddings: np.ndarray,
        galleries: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        candidate_scores: list[dict[str, Any]] = []
        for candidate in galleries:
            similarity_to_gallery = video_embeddings @ candidate["gallery"].T
            per_frame_best = similarity_to_gallery.max(axis=1)
            session_score = float(np.median(per_frame_best) if self.config.use_median_score else np.mean(per_frame_best))
            match_ratio = float(np.mean(per_frame_best >= self.config.match_threshold))
            identity_match = bool(
                session_score >= self.config.match_threshold and match_ratio >= self.config.min_match_ratio
            )
            candidate_scores.append(
                {
                    "user_id": candidate["user_id"],
                    "username": candidate["username"],
                    "session_score": session_score,
                    "match_ratio": match_ratio,
                    "identity_match": identity_match,
                    "per_frame_best": per_frame_best,
                }
            )

        if not candidate_scores:
            raise RuntimeError("No registered users available for authentication.")

        candidate_scores.sort(
            key=lambda item: (item["session_score"], item["match_ratio"], item["username"].lower()),
            reverse=True,
        )
        return candidate_scores[0], candidate_scores

    def register_user(
        self,
        username: str,
        file_storage=None,
        image_files: list[Any] | None = None,
        challenge: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not username.strip():
            raise RuntimeError("Username is required")
        run_dir, records, input_path, input_kind, frame_paths = self._prepare_submission(
            username,
            "register",
            file_storage=file_storage,
            image_files=image_files,
        )
        blink = self._validate_challenge(frame_paths, challenge)
        if len(records) < self.config.min_detected_faces:
            raise RuntimeError(
                f"Only {len(records)} face crops detected; need at least {self.config.min_detected_faces}"
            )
        liveness = self._run_liveness(records)
        authenticity = self._build_authenticity_assessment(liveness)
        if not liveness["session_live"]:
            raise RuntimeError(
                "Registration requires a live face capture. "
                f"Liveness failed: mean_spoof={liveness['mean_spoof']:.4f}, "
                f"live_ratio={liveness['live_ratio']:.2%}, "
                f"threshold={self.config.liveness_spoof_threshold:.2f}, "
                f"min_live_ratio={self.config.min_live_ratio:.2%}."
            )
        sampled_records = self._sample_records(records)
        user_id = self._upsert_user(username)
        self._store_profile_embeddings(user_id, sampled_records)

        sampled_embeddings = np.stack([record["embedding"] for record in sampled_records], axis=0)
        mean_embedding = sampled_embeddings.mean(axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        profile_path = self.config.profile_dir / f"{username}.npz"
        np.savez_compressed(
            profile_path,
            username=username,
            user_id=user_id,
            embeddings=sampled_embeddings,
            mean_embedding=mean_embedding,
        )

        return {
            "mode": "register",
            "username": username,
            "user_id": user_id,
            "input_kind": input_kind,
            "input_path": input_path,
            "video_path": input_path,
            "run_dir": str(run_dir),
            "detected_faces": len(records),
            "sampled_faces": len(sampled_records),
            "blink": blink,
            "liveness": liveness,
            "authenticity": authenticity,
            "profile_path": str(profile_path),
            "saved": True,
            "comparison": None,
            "frame_results": [],
        }

    def authenticate_user(
        self,
        file_storage=None,
        image_files: list[Any] | None = None,
        challenge: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        galleries = self._load_registered_galleries()
        if not galleries:
            raise RuntimeError("No registered users available. Register at least one user before authentication.")

        run_dir, records, input_path, input_kind, frame_paths = self._prepare_submission(
            "visual_only",
            "authenticate",
            file_storage=file_storage,
            image_files=image_files,
        )
        blink = self._validate_challenge(frame_paths, challenge)
        if len(records) < self.config.min_detected_faces:
            raise RuntimeError(
                f"Only {len(records)} face crops detected; need at least {self.config.min_detected_faces}"
            )
        liveness = self._run_liveness(records)
        authenticity = self._build_authenticity_assessment(liveness)
        sampled_records = self._sample_records(records)

        video_embeddings = np.stack([record["embedding"] for record in sampled_records], axis=0).astype(np.float32)
        best_match, ranked_candidates = self._score_identity_candidates(video_embeddings, galleries)
        per_frame_best = best_match["per_frame_best"]
        session_score = float(best_match["session_score"])
        match_ratio = float(best_match["match_ratio"])
        identity_match = bool(best_match["identity_match"])
        authenticated = bool(liveness["session_live"] and identity_match)
        resolved_username = best_match["username"] if authenticated else None
        resolved_user_id = best_match["user_id"] if authenticated else None

        attack_flagged = bool(authenticity["flagged"])
        unknown_attempt_id = None
        unknown_person_alert = False
        if not authenticated and not attack_flagged and bool(liveness["session_live"]):
            preview_crop_path = str(sampled_records[0]["crop_path"]) if sampled_records else None
            unknown_attempt_id = self.record_unknown_attempt(
                input_kind=input_kind,
                input_path=input_path,
                run_dir=str(run_dir),
                preview_crop_path=preview_crop_path,
                live_ratio=float(liveness["live_ratio"]),
                mean_spoof=float(liveness["mean_spoof"]),
                session_score=session_score,
                match_ratio=match_ratio,
            )
            unknown_person_alert = True
        if authenticated:
            security_message = (
                f"Authenticated as {best_match['username']} after the live human session passed identity thresholds."
            )
        elif authenticity["flagged"]:
            security_message = (
                "Authentication failed because the capture did not pass the liveness and anti-spoof screening. "
                "The attempt has been blocked for review."
            )
        else:
            security_message = (
                "Authentication failed because the session looked live but did not match any enrolled identity strongly enough."
            )

        frame_results = []
        for record, similarity in zip(sampled_records, per_frame_best.tolist()):
            frame_results.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "crop_path": str(record["crop_path"]),
                    "spoof_prob": float(record["spoof_prob"]),
                    "is_live": bool(record["is_live"]),
                    "authenticity_label": "Human" if bool(record["is_live"]) else "AI Suspected",
                    "similarity": float(similarity),
                    "matched": bool(similarity >= self.config.match_threshold),
                }
            )

        return {
            "mode": "authenticate",
            "username": resolved_username,
            "user_id": resolved_user_id,
            "input_kind": input_kind,
            "input_path": input_path,
            "video_path": input_path,
            "run_dir": str(run_dir),
            "detected_faces": len(records),
            "sampled_faces": len(sampled_records),
            "blink": blink,
            "liveness": liveness,
            "authenticity": authenticity,
            "closest_match": {
                "user_id": resolved_user_id,
                "username": resolved_username,
                "session_score": session_score,
                "match_ratio": match_ratio,
                "identity_match": identity_match,
            },
            "comparison": {
                "session_score": session_score,
                "match_ratio": match_ratio,
                "match_threshold": self.config.match_threshold,
                "min_match_ratio": self.config.min_match_ratio,
                "authenticated": authenticated,
                "identity_match": identity_match,
                "liveness_passed": bool(liveness["session_live"]),
            },
            "candidate_rankings": (
                [
                    {
                        "user_id": candidate["user_id"],
                        "username": candidate["username"],
                        "session_score": float(candidate["session_score"]),
                        "match_ratio": float(candidate["match_ratio"]),
                        "identity_match": bool(candidate["identity_match"]),
                    }
                    for candidate in ranked_candidates[:3]
                ]
                if authenticated
                else []
            ),
            "frame_results": frame_results,
            "authenticated": authenticated,
            "attack_flagged": attack_flagged,
            "unknown_person_alert": unknown_person_alert,
            "unknown_attempt_id": unknown_attempt_id,
            "security_message": security_message,
        }

    def list_users(self) -> list[dict[str, Any]]:
        self._init_db()
        with sqlite3.connect(self.config.database_path) as conn:
            rows = conn.execute(
                """
                SELECT u.id, u.username, u.created_at, u.updated_at, COUNT(pe.id) AS embedding_count
                FROM users u
                LEFT JOIN profile_embeddings pe ON pe.user_id = u.id
                GROUP BY u.id, u.username, u.created_at, u.updated_at
                ORDER BY u.username
                """
            ).fetchall()
        return [
            {
                "id": int(row[0]),
                "username": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "embedding_count": int(row[4]),
            }
            for row in rows
        ]

