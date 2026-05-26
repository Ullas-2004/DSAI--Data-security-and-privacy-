from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from buffalo_auth_core import AuthConfig, AuthEngine


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "video"
DEFAULT_SWEEP_ROOT = PROJECT_ROOT / "app_data" / "threshold_sweep"


@dataclass(frozen=True)
class ClipEntry:
    individual_id: str
    label: str
    clip_name: str
    path: Path
    partner_id: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep authentication thresholds on the labeled real/spoof dataset.")
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT), help="Directory containing real/, spoof/, and manifest CSVs.")
    parser.add_argument(
        "--work-root",
        default=str(DEFAULT_SWEEP_ROOT),
        help="Directory for extracted crops, caches, and sweep reports.",
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute per-clip features even if a cache exists.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top configurations to print.")
    return parser.parse_args()


def load_dataset(video_root: Path) -> list[ClipEntry]:
    manifest_path = video_root / "clip_manifest.csv"
    entries: list[ClipEntry] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_path = Path(row["path"]) if row.get("path") else None
            resolved_path = row_path if row_path and row_path.exists() else (video_root / row["label"] / row["clip_name"])
            entries.append(
                ClipEntry(
                    individual_id=row["individual_id"],
                    label=row["label"],
                    clip_name=row["clip_name"],
                    path=resolved_path,
                    partner_id=row["partner_id"] or None,
                )
            )
    return sorted(entries, key=lambda item: (item.individual_id, item.label, item.clip_name))


def build_engine(work_root: Path) -> AuthEngine:
    config = AuthConfig(
        database_path=work_root / "scratch_auth.db",
        upload_dir=work_root / "uploads",
        profile_dir=work_root / "profiles",
    )
    return AuthEngine(config)


def cache_path(cache_dir: Path, clip_name: str) -> Path:
    return cache_dir / f"{Path(clip_name).stem}.pt"


@torch.no_grad()
def collect_clip_features(engine: AuthEngine, entry: ClipEntry, cache_dir: Path, rebuild_cache: bool) -> dict[str, Any]:
    target = cache_path(cache_dir, entry.clip_name)
    if target.exists() and not rebuild_cache:
        return torch.load(target, map_location="cpu", weights_only=False)

    run_dir = cache_dir / "runs" / Path(entry.clip_name).stem
    run_dir.mkdir(parents=True, exist_ok=True)

    records = engine._collect_face_records(entry.path, run_dir)
    engine._ensure_models()
    assert engine.liveness_model is not None
    assert engine.liveness_transform is not None

    cached_records: list[dict[str, Any]] = []
    for record in records:
        x = engine.liveness_transform(record["face_image"]).unsqueeze(0).to(engine.device)
        logits = engine.liveness_model(x)
        probs = torch.softmax(logits, dim=1)
        cached_records.append(
            {
                "frame_index": int(record["frame_index"]),
                "crop_path": str(record["crop_path"]),
                "embedding": np.asarray(record["embedding"], dtype=np.float32),
                "spoof_prob": float(probs[0, 1].item()),
            }
        )

    payload = {
        "individual_id": entry.individual_id,
        "label": entry.label,
        "clip_name": entry.clip_name,
        "path": str(entry.path),
        "partner_id": entry.partner_id,
        "detected_faces": len(cached_records),
        "records": cached_records,
    }
    torch.save(payload, target)
    return payload


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return vectors / norms


def make_session_state(
    clip_features: dict[str, Any],
    liveness_threshold: float,
    min_live_ratio: float,
    min_detected_faces: int,
    max_embedding_faces: int,
    random_seed: int,
) -> dict[str, Any]:
    records = clip_features["records"]
    if not records:
        return {
            "valid": False,
            "session_live": False,
            "detected_faces": 0,
            "usable_faces": 0,
            "live_ratio": 0.0,
            "mean_spoof": 1.0,
            "sampled_embeddings": None,
        }

    spoof_probs = np.asarray([record["spoof_prob"] for record in records], dtype=np.float32)
    live_mask = spoof_probs < liveness_threshold
    live_ratio = float(live_mask.mean())
    mean_spoof = float(spoof_probs.mean())
    live_records = [record for record, is_live in zip(records, live_mask.tolist()) if is_live]
    candidates = live_records if live_records else records
    session_live = bool(mean_spoof < liveness_threshold and live_ratio >= min_live_ratio)

    if len(candidates) < min_detected_faces:
        return {
            "valid": False,
            "session_live": session_live,
            "detected_faces": len(records),
            "usable_faces": len(candidates),
            "live_ratio": live_ratio,
            "mean_spoof": mean_spoof,
            "sampled_embeddings": None,
        }

    sample_count = min(max_embedding_faces, len(candidates))
    rng = random.Random(random_seed)
    sampled_records = rng.sample(candidates, sample_count)
    sampled_embeddings = np.stack([np.asarray(record["embedding"], dtype=np.float32) for record in sampled_records], axis=0)

    return {
        "valid": True,
        "session_live": session_live,
        "detected_faces": len(records),
        "usable_faces": len(candidates),
        "live_ratio": live_ratio,
        "mean_spoof": mean_spoof,
        "sampled_embeddings": normalize_rows(sampled_embeddings),
    }


def compare_sessions(
    profile_state: dict[str, Any],
    query_state: dict[str, Any],
    use_median_score: bool,
    match_threshold: float,
    min_match_ratio: float,
) -> dict[str, Any]:
    if not profile_state["valid"] or not query_state["valid"]:
        return {
            "accepted": False,
            "session_score": None,
            "match_ratio": 0.0,
            "profile_valid": profile_state["valid"],
            "query_valid": query_state["valid"],
        }

    query_embeddings = query_state["sampled_embeddings"]
    gallery = profile_state["sampled_embeddings"]
    similarity_to_gallery = query_embeddings @ gallery.T
    per_frame_best = similarity_to_gallery.max(axis=1)
    session_score = float(np.median(per_frame_best) if use_median_score else np.mean(per_frame_best))
    match_ratio = float(np.mean(per_frame_best >= match_threshold))
    accepted = bool(
        query_state["session_live"]
        and session_score >= match_threshold
        and match_ratio >= min_match_ratio
    )
    return {
        "accepted": accepted,
        "session_score": session_score,
        "match_ratio": match_ratio,
        "profile_valid": True,
        "query_valid": True,
    }


def rate(values: list[bool]) -> float:
    return float(sum(bool(value) for value in values) / len(values)) if values else 0.0


def build_pair_sets(entries: list[ClipEntry]) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    real_entries = [entry for entry in entries if entry.label == "real"]
    spoof_by_individual: dict[str, list[ClipEntry]] = defaultdict(list)
    for entry in entries:
        if entry.label == "spoof":
            spoof_by_individual[entry.individual_id].append(entry)

    genuine_pairs: list[tuple[str, str]] = []
    impostor_pairs: list[tuple[str, str]] = []
    spoof_pairs: list[tuple[str, str]] = []

    for profile_entry in real_entries:
        for query_entry in real_entries:
            if query_entry.clip_name == profile_entry.clip_name:
                continue
            if query_entry.individual_id == profile_entry.individual_id:
                genuine_pairs.append((profile_entry.clip_name, query_entry.clip_name))
            else:
                impostor_pairs.append((profile_entry.clip_name, query_entry.clip_name))
        for spoof_entry in spoof_by_individual.get(profile_entry.individual_id, []):
            spoof_pairs.append((profile_entry.clip_name, spoof_entry.clip_name))

    return genuine_pairs, impostor_pairs, spoof_pairs


def evaluate_configuration(
    entries_by_name: dict[str, ClipEntry],
    features_by_name: dict[str, dict[str, Any]],
    genuine_pairs: list[tuple[str, str]],
    impostor_pairs: list[tuple[str, str]],
    spoof_pairs: list[tuple[str, str]],
    *,
    liveness_threshold: float,
    min_live_ratio: float,
    match_threshold: float,
    min_match_ratio: float,
    use_median_score: bool,
    min_detected_faces: int,
    max_embedding_faces: int,
    random_seed: int,
) -> dict[str, Any]:
    session_states = {
        clip_name: make_session_state(
            clip_features=features_by_name[clip_name],
            liveness_threshold=liveness_threshold,
            min_live_ratio=min_live_ratio,
            min_detected_faces=min_detected_faces,
            max_embedding_faces=max_embedding_faces,
            random_seed=random_seed,
        )
        for clip_name in features_by_name
    }

    real_clips = [entry.clip_name for entry in entries_by_name.values() if entry.label == "real"]
    spoof_clips = [entry.clip_name for entry in entries_by_name.values() if entry.label == "spoof"]

    enrollment_success_rate = rate([session_states[clip_name]["valid"] for clip_name in real_clips])
    real_session_live_rate = rate([session_states[clip_name]["session_live"] for clip_name in real_clips])
    spoof_session_reject_rate = rate([not session_states[clip_name]["session_live"] for clip_name in spoof_clips])

    genuine_accepts: list[bool] = []
    impostor_rejects: list[bool] = []
    spoof_rejects: list[bool] = []

    for profile_name, query_name in genuine_pairs:
        result = compare_sessions(
            session_states[profile_name],
            session_states[query_name],
            use_median_score=use_median_score,
            match_threshold=match_threshold,
            min_match_ratio=min_match_ratio,
        )
        genuine_accepts.append(result["accepted"])

    for profile_name, query_name in impostor_pairs:
        result = compare_sessions(
            session_states[profile_name],
            session_states[query_name],
            use_median_score=use_median_score,
            match_threshold=match_threshold,
            min_match_ratio=min_match_ratio,
        )
        impostor_rejects.append(not result["accepted"])

    for profile_name, query_name in spoof_pairs:
        result = compare_sessions(
            session_states[profile_name],
            session_states[query_name],
            use_median_score=use_median_score,
            match_threshold=match_threshold,
            min_match_ratio=min_match_ratio,
        )
        spoof_rejects.append(not result["accepted"])

    genuine_accept_rate = rate(genuine_accepts)
    impostor_reject_rate = rate(impostor_rejects)
    spoof_reject_rate = rate(spoof_rejects)

    objective_components = [
        enrollment_success_rate,
        real_session_live_rate,
        genuine_accept_rate,
        impostor_reject_rate,
        spoof_reject_rate,
    ]
    objective = float(sum(objective_components) / len(objective_components))
    worst_case_rate = float(min(objective_components))

    return {
        "liveness_threshold": liveness_threshold,
        "min_live_ratio": min_live_ratio,
        "match_threshold": match_threshold,
        "min_match_ratio": min_match_ratio,
        "use_median_score": use_median_score,
        "objective": objective,
        "worst_case_rate": worst_case_rate,
        "enrollment_success_rate": enrollment_success_rate,
        "real_session_live_rate": real_session_live_rate,
        "spoof_session_reject_rate": spoof_session_reject_rate,
        "genuine_accept_rate": genuine_accept_rate,
        "impostor_reject_rate": impostor_reject_rate,
        "spoof_reject_rate": spoof_reject_rate,
        "genuine_pairs": len(genuine_pairs),
        "impostor_pairs": len(impostor_pairs),
        "spoof_pairs": len(spoof_pairs),
    }


def sort_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        results,
        key=lambda row: (
            row["objective"],
            row["worst_case_rate"],
            row["spoof_reject_rate"],
            row["genuine_accept_rate"],
            row["real_session_live_rate"],
            row["enrollment_success_rate"],
        ),
        reverse=True,
    )


def sweep_grid(
    entries_by_name: dict[str, ClipEntry],
    features_by_name: dict[str, dict[str, Any]],
    genuine_pairs: list[tuple[str, str]],
    impostor_pairs: list[tuple[str, str]],
    spoof_pairs: list[tuple[str, str]],
    *,
    liveness_thresholds: list[float],
    min_live_ratios: list[float],
    match_thresholds: list[float],
    min_match_ratios: list[float],
    use_median_options: list[bool],
    min_detected_faces: int,
    max_embedding_faces: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = (
        len(liveness_thresholds)
        * len(min_live_ratios)
        * len(match_thresholds)
        * len(min_match_ratios)
        * len(use_median_options)
    )
    index = 0
    for liveness_threshold in liveness_thresholds:
        for min_live_ratio in min_live_ratios:
            for match_threshold in match_thresholds:
                for min_match_ratio in min_match_ratios:
                    for use_median_score in use_median_options:
                        index += 1
                        if index % 100 == 0 or index == total:
                            print(f"[sweep] {index}/{total}")
                        results.append(
                            evaluate_configuration(
                                entries_by_name,
                                features_by_name,
                                genuine_pairs,
                                impostor_pairs,
                                spoof_pairs,
                                liveness_threshold=liveness_threshold,
                                min_live_ratio=min_live_ratio,
                                match_threshold=match_threshold,
                                min_match_ratio=min_match_ratio,
                                use_median_score=use_median_score,
                                min_detected_faces=min_detected_faces,
                                max_embedding_faces=max_embedding_faces,
                                random_seed=random_seed,
                            )
                        )
    return sort_results(results)


def clip_feature_summary(features_by_name: dict[str, dict[str, Any]], destination: Path) -> None:
    rows: list[dict[str, Any]] = []
    for clip_name, clip_features in sorted(features_by_name.items()):
        spoof_probs = [float(record["spoof_prob"]) for record in clip_features["records"]]
        rows.append(
            {
                "clip_name": clip_name,
                "individual_id": clip_features["individual_id"],
                "label": clip_features["label"],
                "partner_id": clip_features["partner_id"] or "",
                "detected_faces": clip_features["detected_faces"],
                "mean_spoof_prob": float(np.mean(spoof_probs)) if spoof_probs else None,
                "min_spoof_prob": float(np.min(spoof_probs)) if spoof_probs else None,
                "max_spoof_prob": float(np.max(spoof_probs)) if spoof_probs else None,
            }
        )

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def refine_values(center: float, step: float, lower: float, upper: float) -> list[float]:
    values = []
    current = center - (2 * step)
    while current <= center + (2 * step) + 1e-9:
        clipped = min(max(current, lower), upper)
        rounded = round(clipped, 4)
        if rounded not in values:
            values.append(rounded)
        current += step
    return sorted(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    video_root = Path(args.video_root)
    work_root = Path(args.work_root)
    cache_dir = work_root / "clip_cache"
    report_dir = work_root / "reports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    entries = load_dataset(video_root)
    entries_by_name = {entry.clip_name: entry for entry in entries}
    genuine_pairs, impostor_pairs, spoof_pairs = build_pair_sets(entries)

    engine = build_engine(work_root)
    features_by_name: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries, start=1):
        print(f"[features] {index}/{len(entries)} {entry.clip_name}")
        features_by_name[entry.clip_name] = collect_clip_features(engine, entry, cache_dir, args.rebuild_cache)

    clip_feature_summary(features_by_name, report_dir / "clip_feature_summary.csv")

    coarse_results = sweep_grid(
        entries_by_name,
        features_by_name,
        genuine_pairs,
        impostor_pairs,
        spoof_pairs,
        liveness_thresholds=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20],
        min_live_ratios=[0.30, 0.40, 0.50, 0.60, 0.70],
        match_thresholds=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        min_match_ratios=[0.30, 0.40, 0.50, 0.60, 0.70],
        use_median_options=[True, False],
        min_detected_faces=engine.config.min_detected_faces,
        max_embedding_faces=engine.config.max_embedding_faces,
        random_seed=engine.config.random_seed,
    )
    write_csv(report_dir / "coarse_results.csv", coarse_results)
    coarse_best = coarse_results[0]

    refined_results = sweep_grid(
        entries_by_name,
        features_by_name,
        genuine_pairs,
        impostor_pairs,
        spoof_pairs,
        liveness_thresholds=refine_values(coarse_best["liveness_threshold"], 0.01, 0.08, 0.24),
        min_live_ratios=refine_values(coarse_best["min_live_ratio"], 0.05, 0.20, 0.90),
        match_thresholds=refine_values(coarse_best["match_threshold"], 0.025, 0.30, 0.85),
        min_match_ratios=refine_values(coarse_best["min_match_ratio"], 0.05, 0.20, 0.90),
        use_median_options=[True, False],
        min_detected_faces=engine.config.min_detected_faces,
        max_embedding_faces=engine.config.max_embedding_faces,
        random_seed=engine.config.random_seed,
    )
    write_csv(report_dir / "refined_results.csv", refined_results)

    best = refined_results[0]
    summary = {
        "dataset": {
            "video_root": str(video_root),
            "real_clips": sum(1 for entry in entries if entry.label == "real"),
            "spoof_clips": sum(1 for entry in entries if entry.label == "spoof"),
            "genuine_pairs": len(genuine_pairs),
            "impostor_pairs": len(impostor_pairs),
            "spoof_pairs": len(spoof_pairs),
        },
        "best": best,
        "top_results": refined_results[: args.top_k],
        "report_dir": str(report_dir),
    }
    (report_dir / "best_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBest configuration")
    for key in [
        "liveness_threshold",
        "min_live_ratio",
        "match_threshold",
        "min_match_ratio",
        "use_median_score",
        "objective",
        "worst_case_rate",
        "enrollment_success_rate",
        "real_session_live_rate",
        "genuine_accept_rate",
        "impostor_reject_rate",
        "spoof_reject_rate",
    ]:
        print(f"{key:24}: {best[key]}")

    print(f"\nReports written to: {report_dir}")


if __name__ == "__main__":
    main()
