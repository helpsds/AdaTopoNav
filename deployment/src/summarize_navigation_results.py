#!/usr/bin/env python3
"""Aggregate trial CSV files and report topological-map size statistics."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

import torch


def optional_mean(rows, field, successful_only=False):
    values = []
    for row in rows:
        if successful_only and row.get("success") != "1":
            continue
        value = row.get(field, "")
        if value != "":
            values.append(float(value))
    return "" if not values else f"{mean(values):.6f}"


def summarize_trials(input_path: Path, output_path: Path) -> None:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["environment"], row["method"])].append(row)

    fields = [
        "environment",
        "method",
        "trials",
        "successes",
        "sr",
        "collision_free_successes",
        "collision_free_sr",
        "mean_collisions",
        "mean_success_time",
        "mean_success_distance",
        "mean_success_spl",
        "mean_final_goal_distance",
        "mean_integrated_abs_angular_velocity",
        "mean_rotation_in_place_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (environment, method), method_rows in sorted(grouped.items()):
            trials = len(method_rows)
            successes = sum(int(row["success"]) for row in method_rows)
            collision_free = sum(
                int(row.get("collision_free_success", "0")) for row in method_rows
            )
            writer.writerow(
                {
                    "environment": environment,
                    "method": method,
                    "trials": trials,
                    "successes": successes,
                    "sr": f"{successes / trials:.6f}",
                    "collision_free_successes": collision_free,
                    "collision_free_sr": f"{collision_free / trials:.6f}",
                    "mean_collisions": optional_mean(method_rows, "collisions"),
                    "mean_success_time": optional_mean(
                        method_rows, "elapsed_seconds", successful_only=True
                    ),
                    "mean_success_distance": optional_mean(
                        method_rows, "travelled_distance", successful_only=True
                    ),
                    "mean_success_spl": optional_mean(
                        method_rows, "spl", successful_only=True
                    ),
                    "mean_final_goal_distance": optional_mean(
                        method_rows, "final_goal_distance"
                    ),
                    "mean_integrated_abs_angular_velocity": optional_mean(
                        method_rows, "integrated_abs_angular_velocity"
                    ),
                    "mean_rotation_in_place_seconds": optional_mean(
                        method_rows, "rotation_in_place_seconds"
                    ),
                }
            )


def file_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def summarize_map(map_root: Path, name: str):
    image_dir = map_root / "images" / name
    images = sorted(image_dir.glob("*.png")) if image_dir.exists() else []
    vectors = map_root / f"{name}_vectors.pt"
    poses = map_root / f"{name}_poses.pt"
    edges = map_root / f"{name}_edges.pt"
    edge_count = ""
    average_degree = ""
    if edges.exists():
        tensor = torch.load(edges, map_location="cpu")
        edge_count = int(tensor.reshape(-1, 2).shape[0])
        if images:
            average_degree = f"{2.0 * edge_count / len(images):.6f}"
    image_bytes = sum(path.stat().st_size for path in images)
    feature_bytes = file_bytes(vectors)
    pose_bytes = file_bytes(poses)
    edge_bytes = file_bytes(edges)
    return {
        "map": name,
        "nodes": len(images),
        "edges": edge_count,
        "average_degree": average_degree,
        "image_bytes": image_bytes,
        "feature_bytes": feature_bytes,
        "pose_bytes": pose_bytes,
        "edge_bytes": edge_bytes,
        "total_bytes": image_bytes + feature_bytes + pose_bytes + edge_bytes,
    }


def summarize_maps(map_root: Path, names, output_path: Path) -> None:
    fields = [
        "map",
        "nodes",
        "edges",
        "average_degree",
        "image_bytes",
        "feature_bytes",
        "pose_bytes",
        "edge_bytes",
        "total_bytes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in names:
            writer.writerow(summarize_map(map_root, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--map-summary", required=True)
    args = parser.parse_args()
    trial_path = Path(args.trials).resolve()
    summary_path = Path(args.summary).resolve()
    map_summary_path = Path(args.map_summary).resolve()
    os.makedirs(summary_path.parent, exist_ok=True)
    summarize_trials(trial_path, summary_path)
    summarize_maps(Path(args.map_root).resolve(), args.maps, map_summary_path)


if __name__ == "__main__":
    main()
