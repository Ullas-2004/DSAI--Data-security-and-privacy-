from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from threshold_sweep_eval import compare_sessions, make_session_state, sort_results


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "video"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "app_data" / "threshold_sweep" / "clip_cache"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "app_data" / "threshold_sweep" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds on exact real->spoof clip pairs.")
    parser.add_argument("--video-root", default=str(DEFAULT_VIDEO_ROOT), help="Directory containing clip_manifest.csv.")
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Directory containing cached per-clip features from threshold_sweep_eval.py.",
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for exact-pair sweep reports.",
    )
    return parser.parse_args()


def load_manifest(video_root: Path) -> list[dict[str, str]]:
    manifest_path = video_root / "clip_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def extract_sequence(clip_name: str) -> str:
    stem = Path(clip_name).stem
    return stem.split("_")[-1]


def build_exact_pairs(manifest_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    real_by_individual: dict[str, list[dict[str, str]]] = {}
    spoof_by_individual: dict[str, list[dict[str, str]]] = {}

    for row in manifest_rows:
        if row["label"] == "real":
            real_by_individual.setdefault(row["individual_id"], []).append(row)
        elif row["label"] == "spoof":
            spoof_by_individual.setdefault(row["individual_id"], []).append(row)

    for rows in real_by_individual.values():
        rows.sort(key=lambda row: row["clip_name"])
    for rows in spoof_by_individual.values():
        rows.sort(key=lambda row: row["clip_name"])

    pairs: list[dict[str, str]] = []
    for individual_id, spoof_rows in sorted(spoof_by_individual.items()):
        real_rows = real_by_individual.get(individual_id, [])
        if not real_rows:
            continue

        real_by_sequence = {extract_sequence(row["clip_name"]): row for row in real_rows}
        unused_real = real_rows.copy()

        for spoof_row in spoof_rows:
            sequence = extract_sequence(spoof_row["clip_name"])
            real_row = real_by_sequence.get(sequence)
            if real_row is None:
                if len(real_rows) == 1:
                    real_row = real_rows[0]
                else:
                    real_row = unused_real[0]
            if real_row in unused_real:
                unused_real.remove(real_row)
            pairs.append(
                {
                    "individual_id": individual_id,
                    "real_clip": real_row["clip_name"],
                    "spoof_clip": spoof_row["clip_name"],
                    "spoof_partner_id": spoof_row["partner_id"],
                }
            )
    return pairs


def load_features(cache_dir: Path, clip_name: str) -> dict[str, Any]:
    cache_path = cache_dir / f"{Path(clip_name).stem}.pt"
    return torch.load(cache_path, map_location="cpu", weights_only=False)


def evaluate_exact_pairs(
    pairs: list[dict[str, str]],
    features_by_name: dict[str, dict[str, Any]],
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
            clip_features=features,
            liveness_threshold=liveness_threshold,
            min_live_ratio=min_live_ratio,
            min_detected_faces=min_detected_faces,
            max_embedding_faces=max_embedding_faces,
            random_seed=random_seed,
        )
        for clip_name, features in features_by_name.items()
    }

    profile_valid = []
    spoof_rejects = []
    details = []

    for pair in pairs:
        profile_state = session_states[pair["real_clip"]]
        query_state = session_states[pair["spoof_clip"]]
        result = compare_sessions(
            profile_state,
            query_state,
            use_median_score=use_median_score,
            match_threshold=match_threshold,
            min_match_ratio=min_match_ratio,
        )
        profile_ok = bool(profile_state["valid"])
        spoof_rejected = bool(not result["accepted"])
        profile_valid.append(profile_ok)
        spoof_rejects.append(spoof_rejected)
        details.append(
            {
                "individual_id": pair["individual_id"],
                "real_clip": pair["real_clip"],
                "spoof_clip": pair["spoof_clip"],
                "profile_valid": profile_ok,
                "query_session_live": bool(query_state["session_live"]),
                "accepted": bool(result["accepted"]),
                "spoof_rejected": spoof_rejected,
                "session_score": result["session_score"],
                "match_ratio": result["match_ratio"],
            }
        )

    profile_valid_rate = sum(profile_valid) / len(profile_valid) if profile_valid else 0.0
    spoof_reject_rate = sum(spoof_rejects) / len(spoof_rejects) if spoof_rejects else 0.0
    objective = (profile_valid_rate + spoof_reject_rate) / 2.0

    return {
        "liveness_threshold": liveness_threshold,
        "min_live_ratio": min_live_ratio,
        "match_threshold": match_threshold,
        "min_match_ratio": min_match_ratio,
        "use_median_score": use_median_score,
        "objective": objective,
        "profile_valid_rate": profile_valid_rate,
        "spoof_reject_rate": spoof_reject_rate,
        "pair_count": len(pairs),
        "details": details,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sweep(
    pairs: list[dict[str, str]],
    features_by_name: dict[str, dict[str, Any]],
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
    results = []
    for liveness_threshold in liveness_thresholds:
        for min_live_ratio in min_live_ratios:
            for match_threshold in match_thresholds:
                for min_match_ratio in min_match_ratios:
                    for use_median_score in use_median_options:
                        results.append(
                            evaluate_exact_pairs(
                                pairs,
                                features_by_name,
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
    return sorted(
        results,
        key=lambda row: (row["objective"], row["profile_valid_rate"], row["spoof_reject_rate"]),
        reverse=True,
    )


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    cache_dir = Path(args.cache_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(video_root)
    pairs = build_exact_pairs(manifest_rows)
    pair_manifest_path = report_dir / "exact_pair_manifest.csv"
    write_csv(pair_manifest_path, pairs)

    clip_names = sorted({pair["real_clip"] for pair in pairs} | {pair["spoof_clip"] for pair in pairs})
    features_by_name = {clip_name: load_features(cache_dir, clip_name) for clip_name in clip_names}

    results = sweep(
        pairs,
        features_by_name,
        liveness_thresholds=[0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.21, 0.22],
        min_live_ratios=[0.20, 0.30, 0.40, 0.50, 0.60],
        match_thresholds=[0.40, 0.45, 0.50, 0.55, 0.60],
        min_match_ratios=[0.20, 0.30, 0.40, 0.50, 0.60],
        use_median_options=[True, False],
        min_detected_faces=6,
        max_embedding_faces=12,
        random_seed=42,
    )

    result_rows = []
    for row in results:
        result_rows.append(
            {
                "liveness_threshold": row["liveness_threshold"],
                "min_live_ratio": row["min_live_ratio"],
                "match_threshold": row["match_threshold"],
                "min_match_ratio": row["min_match_ratio"],
                "use_median_score": row["use_median_score"],
                "objective": row["objective"],
                "profile_valid_rate": row["profile_valid_rate"],
                "spoof_reject_rate": row["spoof_reject_rate"],
                "pair_count": row["pair_count"],
            }
        )
    write_csv(report_dir / "exact_pair_results.csv", result_rows)

    best = results[0]
    (report_dir / "exact_pair_best.json").write_text(
        json.dumps(
            {
                "pair_count": len(pairs),
                "pairs": pairs,
                "best": {
                    key: best[key]
                    for key in [
                        "liveness_threshold",
                        "min_live_ratio",
                        "match_threshold",
                        "min_match_ratio",
                        "use_median_score",
                        "objective",
                        "profile_valid_rate",
                        "spoof_reject_rate",
                        "pair_count",
                    ]
                },
                "best_details": best["details"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Exact pair count:", len(pairs))
    print("Best exact-pair configuration:")
    for key in [
        "liveness_threshold",
        "min_live_ratio",
        "match_threshold",
        "min_match_ratio",
        "use_median_score",
        "objective",
        "profile_valid_rate",
        "spoof_reject_rate",
    ]:
        print(f"{key:22}: {best[key]}")


if __name__ == "__main__":
    main()
