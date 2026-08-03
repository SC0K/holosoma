"""Run resumable G1-to-G1 box augmentation for a manifest of trajectories."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from holosoma_retargeting.examples.g1_box_augment import augment_g1_box


DEFAULT_MANIFEST = project_root / "manifests/omni_g1_box_augmentation.txt"
DEFAULT_SCALES = (0.8, 0.9, 1.1, 1.2)


def load_manifest(manifest_path: Path) -> list[str]:
    """Load unique, non-comment manifest entries while preserving order."""
    entries: list[str] = []
    seen: set[str] = set()
    for raw_line in manifest_path.read_text().splitlines():
        entry = raw_line.strip()
        if not entry or entry.startswith("#") or entry in seen:
            continue
        entries.append(entry)
        seen.add(entry)
    return entries


def resolve_manifest_entry(dataset_root: Path, entry: str) -> Path:
    """Resolve an entry, including OmniRetarget's robot/ → robot-object/ rename."""
    direct = dataset_root / entry
    candidates = [direct]
    entry_path = Path(entry)
    if entry_path.parts and entry_path.parts[0] == "robot":
        candidates.append(dataset_root / "robot-object" / Path(*entry_path.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = "\n  - ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Manifest entry not found: {entry}\nTried:\n  - {attempted}")


def scaled_output_path(output_dir: Path, source_path: Path, scale: float) -> Path:
    """Create a collision-free name that retains the complete source stem."""
    scale_name = f"{scale:.2f}_{scale:.2f}_{scale:.2f}"
    return output_dir / f"{source_path.stem}_g1_scale_{scale_name}.npz"


def valid_existing_output(path: Path) -> bool:
    """Return whether output uses the corrected native-layout pipeline."""
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                "qpos" in data
                and data["qpos"].ndim == 2
                and len(data["qpos"]) > 0
                and "qpos_layout" in data
                and str(data["qpos_layout"].item()) == "native"
                and "box_augmentation_version" in data
                and int(data["box_augmentation_version"].item()) >= 2
            )
    except Exception:
        return False


def run_one(task: tuple[str, float, str, int | None, bool, float, str]) -> dict[str, object]:
    """Worker entry point for one motion/scale pair."""
    source_string, scale, output_string, max_frames, no_foot_sticking, foot_threshold, input_layout = task
    source_path = Path(source_string)
    output_path = Path(output_string)
    log_path = output_path.parent / "_logs" / f"{output_path.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    try:
        used_foot_sticking_fallback = False
        with log_path.open("w") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                try:
                    _, object_urdf = augment_g1_box(
                        source_path,
                        (scale, scale, scale),
                        output_path,
                        max_frames=max_frames,
                        no_foot_sticking=no_foot_sticking,
                        foot_velocity_threshold=foot_threshold,
                        input_layout=input_layout,
                    )
                except RuntimeError as error:
                    if no_foot_sticking or "CVXPY solve failed" not in str(error):
                        raise
                    used_foot_sticking_fallback = True
                    print(
                        f"Foot-sticking solve failed ({error}); retrying the motion "
                        "without foot-sticking constraints."
                    )
                    _, object_urdf = augment_g1_box(
                        source_path,
                        (scale, scale, scale),
                        output_path,
                        max_frames=max_frames,
                        no_foot_sticking=True,
                        foot_velocity_threshold=foot_threshold,
                        input_layout=input_layout,
                    )
        return {
            "status": "completed",
            "source": str(source_path),
            "scale": scale,
            "output": str(output_path),
            "object_urdf": str(object_urdf),
            "log": str(log_path),
            "seconds": time.time() - start,
            "used_foot_sticking_fallback": used_foot_sticking_fallback,
        }
    except Exception as error:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as log_file:
            log_file.write("\n" + traceback.format_exc())
        return {
            "status": "failed",
            "source": str(source_path),
            "scale": scale,
            "output": str(output_path),
            "log": str(log_path),
            "error": repr(error),
            "seconds": time.time() - start,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--max-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--limit", type=int, help="Optionally process only the first N manifest entries")
    parser.add_argument("--no-foot-sticking", action="store_true")
    parser.add_argument("--foot-velocity-threshold", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--input-layout",
        choices=("auto", "native", "omniretarget"),
        default="omniretarget",
        help="Manifest input layout (this OmniRetarget dataset uses omniretarget)",
    )
    arguments = parser.parse_args()

    dataset_root = arguments.dataset_root.resolve()
    manifest_path = arguments.manifest.resolve()
    output_dir = (
        arguments.output_dir.resolve()
        if arguments.output_dir
        else dataset_root / "robot-object-box-scale-augmented"
    )
    scales = tuple(float(scale) for scale in arguments.scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--scales must contain positive values")
    if arguments.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    entries = load_manifest(manifest_path)
    if arguments.limit is not None:
        if arguments.limit <= 0:
            raise ValueError("--limit must be positive")
        entries = entries[: arguments.limit]
    sources = [resolve_manifest_entry(dataset_root, entry) for entry in entries]

    tasks: list[tuple[str, float, str, int | None, bool, float, str]] = []
    skipped = 0
    for source in sources:
        with np.load(source, allow_pickle=False) as data:
            if "qpos" not in data or data["qpos"].ndim != 2 or data["qpos"].shape[1] != 43:
                raise ValueError(f"Incompatible qpos in {source}: expected shape (T, 43)")
        for scale in scales:
            output = scaled_output_path(output_dir, source, scale)
            if not arguments.overwrite and valid_existing_output(output):
                skipped += 1
                continue
            tasks.append(
                (
                    str(source),
                    scale,
                    str(output),
                    arguments.max_frames,
                    arguments.no_foot_sticking,
                    arguments.foot_velocity_threshold,
                    arguments.input_layout,
                )
            )

    print(f"Manifest entries: {len(entries)}")
    print(f"Scales: {scales}")
    print(f"Total expected outputs: {len(entries) * len(scales)}")
    print(f"Already complete: {skipped}")
    print(f"Pending: {len(tasks)}")
    print(f"Output directory: {output_dir}")
    if arguments.dry_run or not tasks:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    completed = failed = 0
    start = time.time()
    with ProcessPoolExecutor(
        max_workers=arguments.max_workers,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(run_one, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1
            print(
                f"[{index}/{len(tasks)}] {result['status']}: "
                f"{Path(str(result['source'])).name} scale={result['scale']} "
                f"({float(result['seconds']):.1f}s)"
            )
            if result["status"] == "failed":
                print(f"  {result['error']} — log: {result['log']}")

    summary = {
        "manifest": str(manifest_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "scales": scales,
        "manifest_entries": len(entries),
        "skipped": skipped,
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": time.time() - start,
        "results": results,
    }
    summary_path = output_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary: {summary_path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
