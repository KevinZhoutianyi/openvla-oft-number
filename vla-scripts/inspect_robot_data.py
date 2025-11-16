#!/usr/bin/env python3
"""
inspect_robot_data.py

Utility to inspect RobotTwin / OpenVLA style HDF5 datasets and report what
fields are present, how many timesteps exist, and which modalities/instructions
are available. The goal is to quickly understand dataset coverage without
storing any interpretation of task success (which is evaluated online).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

# Ensure prismatic modules remain importable if we ever need them here.
sys.path.insert(0, str(Path(__file__).parent.parent))


INSTRUCTION_KEYS = ("seen", "unseen", "language_instruction", "language", "task")
ACTION_KEYS = ("action", "actions", "relative_action")
IMAGE_SUFFIX = "_image"


def format_bytes(num_bytes: int) -> str:
    """Return human-readable file size."""
    if num_bytes < 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect RobotTwin/OpenVLA datasets (HDF5, RAW, RLDS) and summarize structure and modalities."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to a directory containing HDF5 files, or a single HDF5 file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optionally limit the number of HDF5 files inspected (useful for very large folders).",
    )
    parser.add_argument(
        "--per-file-details",
        type=int,
        default=3,
        help="Number of per-file rows to display (set to 0 to hide per-file breakdown).",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "hdf5", "raw", "rlds"],
        default="auto",
        help="Explicitly set the dataset format. Defaults to automatic detection.",
    )
    parser.add_argument(
        "--show-all-shapes",
        action="store_true",
        help="Show every unique shape encountered for each dataset key (default shows up to 5).",
    )
    return parser.parse_args()


def detect_format(data_path: Path, user_choice: str) -> str:
    """Detect dataset format based on filesystem structure."""
    if user_choice != "auto":
        return user_choice

    if data_path.is_dir():
        dataset_info_path = data_path / "dataset_info.json"
        if dataset_info_path.exists():
            return "rlds"
        # Look one and two levels down for dataset_info.json to infer RLDS root directory
        for child in data_path.iterdir():
            if not child.is_dir():
                continue
            if (child / "dataset_info.json").exists():
                return "rlds"
            for grandchild in child.iterdir():
                if grandchild.is_dir() and (grandchild / "dataset_info.json").exists():
                    return "rlds"
        # Otherwise assume HDF5-based (raw or processed)
        hdf5_files = build_file_list(data_path)
        if not hdf5_files:
            raise ValueError(f"Cannot detect dataset format under {data_path}")
        first_file = hdf5_files[0]
        return "raw" if is_raw_hdf5(first_file) else "hdf5"

    if data_path.is_file() and data_path.suffix.lower() in {".h5", ".hdf5"}:
        return "raw" if is_raw_hdf5(data_path) else "hdf5"

    raise ValueError(f"Unsupported path or unable to detect format: {data_path}")


def is_raw_hdf5(file_path: Path) -> bool:
    """Heuristic to determine if an HDF5 file follows the raw LIBERO-style structure."""
    try:
        with h5py.File(file_path, "r") as f:
            data_group = f.get("data")
            if isinstance(data_group, h5py.Group):
                demo_keys = [key for key in data_group.keys() if key.startswith("demo")]
                return len(demo_keys) > 0
    except Exception:
        return False
    return False


def preview_instruction(f: h5py.File) -> Tuple[Optional[str], Optional[str]]:
    """Return first available instruction key and sample text."""
    for key in INSTRUCTION_KEYS:
        if key in f:
            try:
                dataset = f[key]
                if isinstance(dataset, h5py.Dataset):
                    if dataset.ndim == 0:
                        value = dataset[()]
                    else:
                        value = dataset[0]
                else:
                    continue
                if isinstance(value, bytes):
                    return key, value.decode("utf-8", errors="ignore")
                if isinstance(value, np.ndarray):
                    if value.dtype == object and len(value) > 0:
                        candidate = value[0]
                        if isinstance(candidate, bytes):
                            return key, candidate.decode("utf-8", errors="ignore")
                        return key, str(candidate)
                    return key, str(value)
                return key, str(value)
            except Exception:
                continue
    return None, None


def format_shape(shape: Iterable[int]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def infer_modalities(dataset_names: Iterable[str]) -> Dict[str, Any]:
    names = set(dataset_names)
    cameras = sorted([name for name in names if name.endswith(IMAGE_SUFFIX)])
    has_actions = any(key in names for key in ACTION_KEYS)
    has_proprio = "proprio" in names
    return {
        "cameras": cameras,
        "has_actions": has_actions,
        "has_proprio": has_proprio,
    }


def analyze_file(file_path: Path) -> Dict[str, Any]:
    file_info: Dict[str, Any] = {
        "path": str(file_path),
        "datasets": [],
        "num_steps": None,
        "instruction": None,
        "modalities": {},
    }

    with h5py.File(file_path, "r") as f:
        dataset_names: List[str] = []
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                dataset_names.append(key)
                file_info["datasets"].append(
                    {
                        "name": key,
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "compression": obj.compression or "None",
                    }
                )

        file_info["modalities"] = infer_modalities(dataset_names)

        # Approximate episode length from primary action keys.
        for action_key in ACTION_KEYS:
            if action_key in f and isinstance(f[action_key], h5py.Dataset):
                file_info["num_steps"] = int(f[action_key].shape[0])
                break

        instruction_key, instruction_value = preview_instruction(f)
        if instruction_key:
            file_info["instruction"] = {
                "key": instruction_key,
                "sample": instruction_value,
            }

    return file_info


def analyze_raw_file(file_path: Path) -> Dict[str, Any]:
    """Inspect a raw LIBERO-style HDF5 file with nested demo groups."""
    info: Dict[str, Any] = {
        "path": str(file_path),
        "num_demos": 0,
        "demo_lengths": [],
        "dataset_shapes": defaultdict(Counter),
        "dataset_dtypes": defaultdict(set),
    }

    with h5py.File(file_path, "r") as f:
        data_group = f.get("data")
        if not isinstance(data_group, h5py.Group):
            raise ValueError(f"{file_path} does not appear to be a RAW-format file (missing 'data' group).")

        for demo_name, demo_group in data_group.items():
            if not isinstance(demo_group, h5py.Group):
                continue
            info["num_demos"] += 1

            actions = demo_group.get("actions")
            if isinstance(actions, h5py.Dataset):
                info["demo_lengths"].append(actions.shape[0])

            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    key = name  # relative path inside demo
                    shape = tuple(obj.shape)
                    info["dataset_shapes"][key][shape] += 1
                    info["dataset_dtypes"][key].add(str(obj.dtype))

            demo_group.visititems(visitor)

    return info


def build_file_list(data_path: Path) -> List[Path]:
    if data_path.is_file():
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")

    candidates: List[Path] = []
    for suffix in ("*.h5", "*.hdf5", "*.H5", "*.HDF5"):
        candidates.extend(sorted(data_path.glob(suffix)))
    return sorted(set(candidates))


def aggregate_hdf5_stats(file_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    dataset_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "count": 0,
        "dtypes": set(),
        "shapes": set(),
    })
    total_steps = 0

    for info in file_infos:
        num_steps = info.get("num_steps")
        if isinstance(num_steps, int):
            total_steps += num_steps

        for dataset in info.get("datasets", []):
            name = dataset["name"]
            dataset_stats[name]["count"] += 1
            dataset_stats[name]["dtypes"].add(dataset["dtype"])
            dataset_stats[name]["shapes"].add(tuple(dataset["shape"]))

    return {
        "dataset_stats": dataset_stats,
        "total_steps": total_steps,
    }


def aggregate_raw_stats(file_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_demos = 0
    all_lengths: List[int] = []
    dataset_shapes: Dict[str, Counter] = defaultdict(Counter)
    dataset_dtypes: Dict[str, set] = defaultdict(set)

    for info in file_infos:
        total_demos += info["num_demos"]
        all_lengths.extend(info["demo_lengths"])
        for key, shape_counter in info["dataset_shapes"].items():
            for shape, count in shape_counter.items():
                dataset_shapes[key][shape] += count
        for key, dtypes in info["dataset_dtypes"].items():
            dataset_dtypes[key].update(dtypes)

    return {
        "total_demos": total_demos,
        "action_lengths": all_lengths,
        "dataset_shapes": dataset_shapes,
        "dataset_dtypes": dataset_dtypes,
    }


def print_hdf5_summary(
    data_path: Path,
    file_infos: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    total_files = len(file_infos)
    print("=" * 80)
    print(f"Dataset inspection summary for: {data_path}")
    print("=" * 80)
    print(f"Files inspected: {total_files}")

    total_steps = aggregate["total_steps"]
    if total_steps:
        print(f"Approximate total action timesteps: {total_steps:,}")

    dataset_stats = aggregate["dataset_stats"]
    if dataset_stats:
        print("\nDataset fields discovered:")
        for name in sorted(dataset_stats.keys()):
            entry = dataset_stats[name]
            count = entry["count"]
            dtype_list = ", ".join(sorted(entry["dtypes"]))
            shape_set = entry["shapes"]
            shape_strings = [format_shape(shape) for shape in sorted(shape_set)]
            if not args.show_all_shapes and len(shape_strings) > 5:
                displayed = ", ".join(shape_strings[:5]) + f", ... (+{len(shape_strings) - 5} more)"
            else:
                displayed = ", ".join(shape_strings)
            print(f"  - {name}: present in {count} file(s); dtype(s): {dtype_list}; shape(s): {displayed}")

    if args.per_file_details and file_infos:
        print("\nPer-file overview (first {}):".format(min(args.per_file_details, len(file_infos))))
        for info in file_infos[: args.per_file_details]:
            print(f"- {info['path']}")
            if info.get("num_steps") is not None:
                print(f"    steps: {info['num_steps']}")
            modalities = info.get("modalities", {})
            cameras = modalities.get("cameras") or []
            if cameras:
                print(f"    cameras: {', '.join(cameras)}")
            print(f"    has_actions: {modalities.get('has_actions', False)}")
            if info.get("instruction"):
                inst = info["instruction"]
                sample_text = inst['sample']
                if sample_text is not None and len(sample_text) > 80:
                    sample_text = sample_text[:77] + "..."
                print(f"    instruction key: {inst['key']} (sample: {sample_text})")
            dataset_names = [dataset["name"] for dataset in info.get("datasets", [])]
            print(f"    datasets: {', '.join(dataset_names)}")


def print_raw_summary(
    data_path: Path,
    file_infos: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    total_files = len(file_infos)
    print("=" * 80)
    print(f"RAW dataset inspection summary for: {data_path}")
    print("=" * 80)
    print(f"Files inspected: {total_files}")
    print(f"Total demonstrations: {aggregate['total_demos']}")

    action_lengths = aggregate["action_lengths"]
    if action_lengths:
        min_len = min(action_lengths)
        max_len = max(action_lengths)
        mean_len = statistics.mean(action_lengths)
        print(f"Action length (timesteps): min={min_len}, mean={mean_len:.1f}, max={max_len}")

    dataset_shapes = aggregate["dataset_shapes"]
    if dataset_shapes:
        print("\nDataset entries discovered (relative to each demo group):")
        for key in sorted(dataset_shapes.keys()):
            shape_counter = dataset_shapes[key]
            dtype_list = ", ".join(sorted(aggregate["dataset_dtypes"].get(key, [])))
            shape_items = [f"{shape}: {count}" for shape, count in shape_counter.most_common()]
            if not args.show_all_shapes and len(shape_items) > 5:
                displayed = ", ".join(shape_items[:5]) + f", ... (+{len(shape_items) - 5} more)"
            else:
                displayed = ", ".join(shape_items)
            print(f"  - {key}: {displayed}; dtype(s): {dtype_list}")

    if args.per_file_details and file_infos:
        print("\nPer-file overview (first {}):".format(min(args.per_file_details, len(file_infos))))
        for info in file_infos[: args.per_file_details]:
            print(f"- {info['path']}")
            num_demos = info["num_demos"]
            print(f"    demos: {num_demos}")
            if info["demo_lengths"]:
                min_len = min(info["demo_lengths"])
                max_len = max(info["demo_lengths"])
                mean_len = statistics.mean(info["demo_lengths"])
                print(f"    length stats: min={min_len}, mean={mean_len:.1f}, max={max_len}")
            dataset_keys = sorted(info["dataset_shapes"].keys())
            if dataset_keys:
                preview = ", ".join(dataset_keys[:8])
                if len(dataset_keys) > 8:
                    preview += f", ... (+{len(dataset_keys) - 8} more)"
                print(f"    datasets: {preview}")


def build_rlds_dataset_list(data_path: Path) -> List[Path]:
    if data_path.is_file():
        raise ValueError("Expected directory path for RLDS datasets.")

    dataset_dirs: List[Path] = []
    direct_candidate = data_path / "dataset_info.json"
    if direct_candidate.exists():
        dataset_dirs.append(data_path)
        return dataset_dirs

    for child in sorted(data_path.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "dataset_info.json"
        if candidate.exists():
            dataset_dirs.append(child)
            continue
        for grandchild in sorted(child.iterdir()):
            if grandchild.is_dir() and (grandchild / "dataset_info.json").exists():
                dataset_dirs.append(grandchild)
    return dataset_dirs


def extract_feature_keys(features: Dict[str, Any]) -> List[str]:
    keys: set = set()

    def recurse(node: Any, prefix: Optional[str] = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"feature", "dict", "tuple"}:
                    recurse(value, prefix)
                else:
                    full_key = f"{prefix}.{key}" if prefix else key
                    keys.add(full_key)
                    recurse(value, full_key)
        elif isinstance(node, list):
            for idx, value in enumerate(node):
                recurse(value, prefix)

    recurse(features)
    return sorted(keys)


def analyze_rlds_dataset(dataset_dir: Path) -> Dict[str, Any]:
    dataset_info_path = dataset_dir / "dataset_info.json"
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found under {dataset_dir}")

    with open(dataset_info_path, "r") as f:
        info_json = json.load(f)

    dataset_name = info_json.get("name") or dataset_dir.parent.name
    version = info_json.get("version") or dataset_dir.name
    config_name = info_json.get("configName")

    splits_summary = []
    total_examples = 0
    for split_name, split_info in sorted(info_json.get("splits", {}).items()):
        num_examples = split_info.get("numExamples")
        if num_examples:
            total_examples += num_examples
        num_shards = split_info.get("numShards")
        shard_lengths = split_info.get("shardLengths", [])
        if num_shards is None and shard_lengths:
            num_shards = len(shard_lengths)
        splits_summary.append(
            {
                "name": split_name,
                "num_examples": num_examples,
                "num_shards": num_shards,
            }
        )

    feature_keys = extract_feature_keys(info_json.get("features", {}))

    statistics_path = dataset_dir / "statistics.json"
    statistics_keys: Optional[List[str]] = None
    if statistics_path.exists():
        with open(statistics_path, "r") as f:
            statistics_data = json.load(f)
        statistics_keys = sorted(statistics_data.keys())

    tfrecord_files = list(dataset_dir.glob("*.tfrecord*"))
    total_size_bytes = sum(file.stat().st_size for file in tfrecord_files)

    return {
        "path": str(dataset_dir),
        "dataset_name": dataset_name,
        "version": version,
        "config_name": config_name,
        "splits": splits_summary,
        "feature_keys": feature_keys,
        "statistics_keys": statistics_keys,
        "num_tfrecords": len(tfrecord_files),
        "total_size_bytes": total_size_bytes,
        "total_examples": total_examples,
    }


def aggregate_rlds_stats(dataset_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_examples = 0
    total_size_bytes = 0
    dataset_names = []

    for info in dataset_infos:
        if info["total_examples"]:
            total_examples += info["total_examples"]
        total_size_bytes += info["total_size_bytes"]
        dataset_names.append(info["dataset_name"])

    return {
        "total_examples": total_examples,
        "total_size_bytes": total_size_bytes,
        "dataset_names": dataset_names,
    }


def print_rlds_summary(
    data_path: Path,
    dataset_infos: List[Dict[str, Any]],
    aggregate: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    print("=" * 80)
    print(f"RLDS dataset inspection summary for: {data_path}")
    print("=" * 80)
    print(f"Datasets inspected: {len(dataset_infos)}")

    total_examples = aggregate["total_examples"]
    if total_examples:
        print(f"Total examples across datasets: {total_examples:,}")
    total_size = aggregate["total_size_bytes"]
    print(f"Total TFRecord size: {format_bytes(total_size)}")

    if dataset_infos:
        print("\nDatasets:")
        for info in dataset_infos:
            config_str = f" (config: {info['config_name']})" if info["config_name"] else ""
            print(f"  - {info['dataset_name']} v{info['version']}{config_str}")

    if args.per_file_details and dataset_infos:
        print("\nDataset details (first {}):".format(min(args.per_file_details, len(dataset_infos))))
        for info in dataset_infos[: args.per_file_details]:
            print(f"- {info['path']}")
            config_str = f", config={info['config_name']}" if info["config_name"] else ""
            print(f"    name={info['dataset_name']}, version={info['version']}{config_str}")
            print(f"    tfrecords: {info['num_tfrecords']} files ({format_bytes(info['total_size_bytes'])})")
            if info["splits"]:
                split_preview = ", ".join(
                    f"{split['name']}:{split.get('num_examples', 'unknown')} (shards={split.get('num_shards', 'n/a')})"
                    for split in info["splits"]
                )
                print(f"    splits: {split_preview}")
            if info["feature_keys"]:
                feature_preview = ", ".join(info["feature_keys"][:12])
                if len(info["feature_keys"]) > 12:
                    feature_preview += f", ... (+{len(info['feature_keys']) - 12} more)"
                print(f"    feature keys: {feature_preview}")
            if info["statistics_keys"]:
                print(f"    statistics keys: {', '.join(info['statistics_keys'][:10])}")
def main() -> None:
    args = parse_args()
    data_format = detect_format(args.data_path, args.format)

    if data_format in {"hdf5", "raw"}:
        all_files = build_file_list(args.data_path)
        if not all_files:
            print(f"No HDF5 files found under {args.data_path}")
            return
        files = all_files[: args.max_files] if args.max_files is not None else all_files

        file_infos: List[Dict[str, Any]] = []
        for file_path in files:
            try:
                if data_format == "raw":
                    info = analyze_raw_file(file_path)
                else:
                    info = analyze_file(file_path)
                file_infos.append(info)
            except Exception as exc:
                print(f"[WARN] Failed to analyze {file_path}: {exc}")

        if not file_infos:
            print("No files could be analyzed successfully.")
            return

        if data_format == "raw":
            aggregate = aggregate_raw_stats(file_infos)
            print_raw_summary(args.data_path, file_infos, aggregate, args)
        else:
            aggregate = aggregate_hdf5_stats(file_infos)
            print_hdf5_summary(args.data_path, file_infos, aggregate, args)

    elif data_format == "rlds":
        dataset_dirs = build_rlds_dataset_list(args.data_path)
        if not dataset_dirs:
            print(f"No RLDS dataset_info.json files found under {args.data_path}")
            return
        dataset_dirs = dataset_dirs[: args.max_files] if args.max_files is not None else dataset_dirs

        dataset_infos: List[Dict[str, Any]] = []
        for dataset_dir in dataset_dirs:
            try:
                info = analyze_rlds_dataset(dataset_dir)
                dataset_infos.append(info)
            except Exception as exc:
                print(f"[WARN] Failed to analyze RLDS dataset at {dataset_dir}: {exc}")

        if not dataset_infos:
            print("No RLDS datasets could be analyzed successfully.")
            return

        aggregate = aggregate_rlds_stats(dataset_infos)
        print_rlds_summary(args.data_path, dataset_infos, aggregate, args)

    else:
        raise ValueError(f"Unsupported data format: {data_format}")


if __name__ == "__main__":
    main()
