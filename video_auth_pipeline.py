from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm.auto import tqdm
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LIVENESS_CHECKPOINT = PROJECT_ROOT / "face_liveness_vit" / "model.pt"
DEFAULT_EMBEDDING_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "celeba_embedding_best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline_runs"
DEFAULT_INSIGHTFACE_ROOT = PROJECT_ROOT / ".insightface"
BUNDLED_BUFFALO_DIR = PROJECT_ROOT / "buffalo"


@dataclass
class PipelineConfig:
    video_path: str
    liveness_checkpoint: str = str(DEFAULT_LIVENESS_CHECKPOINT)
    embedding_checkpoint: str = str(DEFAULT_EMBEDDING_CHECKPOINT)
    profile_path: str | None = None
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    insightface_model_name: str = "buffalo_l"
    insightface_root: str | None = str(DEFAULT_INSIGHTFACE_ROOT)
    det_size: tuple[int, int] = (640, 640)
    scan_frame_stride: int = 3
    max_detected_faces: int = 48
    max_embedding_faces: int = 12
    min_detected_faces: int = 6
    liveness_spoof_threshold: float = 0.1199
    min_live_ratio: float = 0.60
    match_threshold: float = 0.60
    min_match_ratio: float = 0.50
    use_median_score: bool = True
    random_seed: int = 42


@dataclass
class FrameRecord:
    frame_index: int
    crop_path: str
    spoof_prob: float
    is_live: bool
    similarity: float | None
    matched: bool | None


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


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
        num_patches = self.patch_embed.num_patches
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
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.pos_drop(x)
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


class FaceEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim, num_classes, use_pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        normalized_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        logits = self.classifier(embedding)
        return normalized_embedding, logits


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def prepare_output_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "crops").mkdir(exist_ok=True)
    return run_dir


def ensure_local_insightface_pack(root_path: Path, model_name: str) -> None:
    model_dir = root_path / "models" / model_name
    if model_name == "buffalo_l" and BUNDLED_BUFFALO_DIR.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        for source_path in BUNDLED_BUFFALO_DIR.glob("*.onnx"):
            target_path = model_dir / source_path.name
            if not target_path.exists():
                shutil.copy2(source_path, target_path)


def get_face_app(config: PipelineConfig, device: torch.device) -> FaceAnalysis:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
    kwargs: dict[str, Any] = {"name": config.insightface_model_name, "providers": providers}
    if config.insightface_root:
        root_path = Path(config.insightface_root).expanduser()
        ensure_local_insightface_pack(root_path, config.insightface_model_name)
        kwargs["root"] = str(root_path)
    face_app = FaceAnalysis(**kwargs)
    face_app.prepare(ctx_id=0 if device.type == "cuda" else -1, det_size=config.det_size)
    return face_app


def get_aligned_face(frame_bgr, face_obj, out_size=224):
    if hasattr(face_obj, "kps") and face_obj.kps is not None:
        aligned_bgr = face_align.norm_crop(frame_bgr, landmark=face_obj.kps, image_size=out_size)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(aligned_rgb)

    x1, y1, x2, y2 = face_obj.bbox.astype(int)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None
    crop_bgr = cv2.resize(crop_bgr, (out_size, out_size))
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def pick_largest_face(faces):
    if not faces:
        return None
    return sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)[0]


def load_liveness_model(checkpoint_path: Path, device: torch.device) -> tuple[LivenessViT, transforms.Compose]:
    model = LivenessViT(
        img_size=224,
        patch_size=16,
        d_model=128,
        nhead=4,
        num_layers=2,
        num_classes=2,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(int(224 * 1.14)),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform


def load_embedding_model(checkpoint_path: Path, device: torch.device) -> tuple[FaceEmbeddingCNN, transforms.Compose, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embedding_dim = checkpoint.get("embedding_dim", 256)
    image_size = checkpoint.get("image_size", 224)
    num_classes = checkpoint.get("num_classes", 10177)

    model = FaceEmbeddingCNN(embedding_dim=embedding_dim, num_classes=num_classes, use_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    meta = {
        "embedding_dim": embedding_dim,
        "image_size": image_size,
        "num_classes": num_classes,
        "source_epoch": checkpoint.get("epoch"),
    }
    return model, transform, meta


def collect_candidate_faces(
    video_path: Path,
    face_app: FaceAnalysis,
    config: PipelineConfig,
    crops_dir: Path,
) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    records: list[dict[str, Any]] = []
    frame_idx = 0
    progress = tqdm(desc="scan video", dynamic_ncols=True)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % config.scan_frame_stride != 0:
                frame_idx += 1
                continue

            faces = face_app.get(frame_bgr)
            face = pick_largest_face(faces)
            if face is not None:
                aligned = get_aligned_face(frame_bgr, face, out_size=224)
                if aligned is not None:
                    crop_path = crops_dir / f"frame_{frame_idx:06d}.jpg"
                    aligned.save(crop_path)
                    records.append(
                        {
                            "frame_index": frame_idx,
                            "face_image": aligned,
                            "crop_path": str(crop_path),
                        }
                    )

            progress.update(1)
            progress.set_postfix({"faces": len(records), "frame": frame_idx})
            if len(records) >= config.max_detected_faces:
                break
            frame_idx += 1
    finally:
        cap.release()
        progress.close()

    return records


@torch.no_grad()
def run_liveness(
    model: LivenessViT,
    transform: transforms.Compose,
    face_records: list[dict[str, Any]],
    config: PipelineConfig,
    device: torch.device,
) -> dict[str, Any]:
    spoof_probs: list[float] = []
    live_flags: list[bool] = []

    progress = tqdm(face_records, desc="liveness", dynamic_ncols=True)
    for record in progress:
        x = transform(record["face_image"]).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        spoof_prob = probs[0, 1].item()
        is_live = spoof_prob < config.liveness_spoof_threshold
        record["spoof_prob"] = spoof_prob
        record["is_live"] = is_live
        spoof_probs.append(spoof_prob)
        live_flags.append(is_live)
        progress.set_postfix({"spoof": f"{spoof_prob:.4f}", "live": is_live})
    progress.close()

    if not spoof_probs:
        raise RuntimeError("No face crops available for liveness inference")

    mean_spoof = sum(spoof_probs) / len(spoof_probs)
    live_ratio = sum(live_flags) / len(live_flags)
    session_live = (mean_spoof < config.liveness_spoof_threshold) and (live_ratio >= config.min_live_ratio)
    return {
        "mean_spoof": mean_spoof,
        "live_ratio": live_ratio,
        "session_live": session_live,
    }


def sample_records_for_embedding(face_records: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    live_records = [record for record in face_records if record.get("is_live")]
    candidates = live_records if live_records else face_records
    if len(candidates) < config.min_detected_faces:
        raise RuntimeError(
            f"Only {len(candidates)} usable face crops available; need at least {config.min_detected_faces}"
        )

    sample_count = min(config.max_embedding_faces, len(candidates))
    rng = random.Random(config.random_seed)
    return rng.sample(candidates, sample_count)


@torch.no_grad()
def embed_faces(
    model: FaceEmbeddingCNN,
    transform: transforms.Compose,
    sampled_records: list[dict[str, Any]],
    device: torch.device,
) -> torch.Tensor:
    tensors = [transform(record["face_image"]).unsqueeze(0) for record in sampled_records]
    batch = torch.cat(tensors, dim=0).to(device)
    embeddings, _ = model(batch)
    return embeddings.cpu()


def load_profile(profile_path: Path) -> dict[str, Any]:
    profile = torch.load(profile_path, map_location="cpu", weights_only=False)
    profile["embeddings"] = profile["embeddings"].float()
    if "mean_embedding" in profile:
        profile["mean_embedding"] = profile["mean_embedding"].float()
    return profile


def compare_with_profile(
    video_embeddings: torch.Tensor,
    profile: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    gallery = profile["embeddings"]
    similarity_to_gallery = video_embeddings @ gallery.T
    per_frame_best = similarity_to_gallery.max(dim=1).values

    if "mean_embedding" in profile:
        mean_embedding = profile["mean_embedding"].unsqueeze(0)
        per_frame_best = torch.maximum(per_frame_best, (video_embeddings @ mean_embedding.T).squeeze(1))

    session_score = per_frame_best.median().item() if config.use_median_score else per_frame_best.mean().item()
    match_ratio = (per_frame_best >= config.match_threshold).float().mean().item()

    return {
        "session_score": session_score,
        "match_ratio": match_ratio,
        "per_frame_best": per_frame_best.tolist(),
        "authenticated": session_score >= config.match_threshold and match_ratio >= config.min_match_ratio,
    }


def serialize_frame_records(records: list[FrameRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_seed)

    device = get_device()
    video_path = Path(config.video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    run_dir = prepare_output_dir(Path(config.output_dir))
    crops_dir = run_dir / "crops"

    liveness_model, liveness_transform = load_liveness_model(Path(config.liveness_checkpoint), device)
    embedding_model, embedding_transform, embedding_meta = load_embedding_model(Path(config.embedding_checkpoint), device)
    face_app = get_face_app(config, device)

    detected_records = collect_candidate_faces(video_path, face_app, config, crops_dir)
    if len(detected_records) < config.min_detected_faces:
        raise RuntimeError(
            f"Only {len(detected_records)} face crops detected; need at least {config.min_detected_faces}"
        )

    liveness_summary = run_liveness(liveness_model, liveness_transform, detected_records, config, device)
    sampled_records = sample_records_for_embedding(detected_records, config)
    video_embeddings = embed_faces(embedding_model, embedding_transform, sampled_records, device)

    comparison = None
    final_decision = None
    if config.profile_path:
        profile = load_profile(Path(config.profile_path))
        comparison = compare_with_profile(video_embeddings, profile, config)
        final_decision = bool(liveness_summary["session_live"] and comparison["authenticated"])

    frame_rows: list[FrameRecord] = []
    similarity_map: dict[int, float] = {}
    matched_map: dict[int, bool] = {}
    if comparison is not None:
        for idx, record in enumerate(sampled_records):
            similarity_map[id(record)] = float(comparison["per_frame_best"][idx])
            matched_map[id(record)] = similarity_map[id(record)] >= config.match_threshold

    for record in sampled_records:
        frame_rows.append(
            FrameRecord(
                frame_index=int(record["frame_index"]),
                crop_path=str(record["crop_path"]),
                spoof_prob=float(record["spoof_prob"]),
                is_live=bool(record["is_live"]),
                similarity=similarity_map.get(id(record)),
                matched=matched_map.get(id(record)),
            )
        )

    embeddings_path = run_dir / "video_embeddings.pt"
    torch.save(
        {
            "embeddings": video_embeddings,
            "frame_indices": [record.frame_index for record in frame_rows],
            "config": asdict(config),
            "embedding_meta": embedding_meta,
        },
        embeddings_path,
    )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path),
        "output_dir": str(run_dir),
        "detected_faces": len(detected_records),
        "sampled_faces": len(sampled_records),
        "liveness": liveness_summary,
        "comparison": comparison,
        "final_decision": final_decision,
        "profile_path": config.profile_path,
        "frame_results": serialize_frame_records(frame_rows),
        "embeddings_path": str(embeddings_path),
        "embedding_meta": embedding_meta,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the video face authentication pipeline on one input video.")
    parser.add_argument("video_path", help="Path to the input video")
    parser.add_argument("--profile-path", default=None, help="Optional profile .pt file for matching")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for run outputs")
    parser.add_argument("--scan-frame-stride", type=int, default=3, help="Process every Nth frame while scanning the video")
    parser.add_argument("--max-detected-faces", type=int, default=48, help="Stop scanning once this many face crops are collected")
    parser.add_argument("--max-embedding-faces", type=int, default=12, help="Number of random face crops to embed")
    parser.add_argument("--min-detected-faces", type=int, default=6, help="Minimum usable face crops required to continue")
    parser.add_argument("--match-threshold", type=float, default=0.60, help="Cosine similarity threshold for a match")
    parser.add_argument("--min-match-ratio", type=float, default=0.50, help="Minimum fraction of sampled frames that must match")
    parser.add_argument("--liveness-threshold", type=float, default=0.1199, help="Maximum spoof probability for a live face")
    parser.add_argument("--min-live-ratio", type=float, default=0.60, help="Minimum live-frame ratio for the session")
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        video_path=args.video_path,
        profile_path=args.profile_path,
        output_dir=args.output_dir,
        scan_frame_stride=args.scan_frame_stride,
        max_detected_faces=args.max_detected_faces,
        max_embedding_faces=args.max_embedding_faces,
        min_detected_faces=args.min_detected_faces,
        match_threshold=args.match_threshold,
        min_match_ratio=args.min_match_ratio,
        liveness_spoof_threshold=args.liveness_threshold,
        min_live_ratio=args.min_live_ratio,
    )


def print_summary(summary: dict[str, Any]) -> None:
    print("\n===== PIPELINE SUMMARY =====")
    print("video               :", summary["video_path"])
    print("output dir          :", summary["output_dir"])
    print("detected faces      :", summary["detected_faces"])
    print("sampled faces       :", summary["sampled_faces"])
    print("session live        :", summary["liveness"]["session_live"])
    print("mean spoof prob     :", f'{summary["liveness"]["mean_spoof"]:.4f}')
    print("live ratio          :", f'{summary["liveness"]["live_ratio"]:.2%}')

    if summary["comparison"] is None:
        print("comparison          : skipped (no profile provided)")
        print("final decision      : skipped")
    else:
        print("session similarity  :", f'{summary["comparison"]["session_score"]:.4f}')
        print("match ratio         :", f'{summary["comparison"]["match_ratio"]:.2%}')
        print("final decision      :", summary["final_decision"])

    print("embeddings path     :", summary["embeddings_path"])
    print("summary json        :", str(Path(summary["output_dir"]) / "summary.json"))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    summary = run_pipeline(config)
    print_summary(summary)


if __name__ == "__main__":
    main()
