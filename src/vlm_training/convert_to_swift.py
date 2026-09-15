"""Convert annotations.json to ms-swift JSONL stage files."""
import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load data_config.yaml and resolve output_dir to absolute path."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        # config lives at <repo_root>/config/; go up one extra level to reach the repo root.
        output_dir = (config_path.parent.parent / output_dir).resolve()
    cfg["output_dir"] = str(output_dir)
    return cfg


def load_annotations(annotations_path: Path, output_dir: str) -> list[dict]:
    """Load annotations.json, resolve image paths, validate stage field exists."""
    annotations_path = Path(annotations_path)
    if not annotations_path.exists():
        print(f"ERROR: annotations.json not found at {annotations_path}")
        raise SystemExit(1)
    with open(annotations_path) as f:
        records = json.load(f)
    base = Path(output_dir)
    resolved = []
    for rec in records:
        if "stage" not in rec:
            print(f"ERROR: record missing 'stage' field: {rec}")
            raise SystemExit(1)
        img = Path(rec["image_path"])
        if not img.is_absolute():
            img = (base / img).resolve()
        resolved.append({**rec, "image_path": str(img)})
    return resolved


def group_by_stage(records: list[dict]) -> dict[int, list[dict]]:
    """Group records by stage integer, return dict sorted by stage number."""
    groups: dict[int, list] = {}
    for rec in records:
        s = int(rec["stage"])
        groups.setdefault(s, []).append(rec)
    return dict(sorted(groups.items()))


def split_stage(records: list[dict], val_ratio: float, seed: int) -> tuple[list, list]:
    """Randomly split records into train and val with a fixed seed."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def to_swift_record(rec: dict, system_prompt: str) -> dict:
    """Convert one annotation record to ms-swift VLM conversation format.

    Note: ms-swift uses '<image>' token in the content string and a separate
    'images' list. This is different from the HuggingFace format used in
    inference.py (which passes {"type": "image", "image": path} directly).
    ms-swift handles the conversion internally during training.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"<image>Question: {rec['question']}"},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": [rec["image_path"]],
    }


def write_stage_files(
    records: list[dict],
    stages_dir: Path,
    val_split_ratio: float,
    random_seed: int,
    system_prompt: str,
) -> None:
    """Group records by stage, split each, write train/val JSONL files."""
    stages_dir = Path(stages_dir)
    stages_dir.mkdir(parents=True, exist_ok=True)
    groups = group_by_stage(records)
    if not groups:
        print("ERROR: No records found after grouping by stage.")
        raise SystemExit(1)
    for stage_num, stage_records in groups.items():
        train_recs, val_recs = split_stage(stage_records, val_split_ratio, random_seed)
        for split_name, split_recs in [("train", train_recs), ("val", val_recs)]:
            out_path = stages_dir / f"stage_{stage_num}_{split_name}.jsonl"
            with open(out_path, "w") as f:
                for rec in split_recs:
                    f.write(json.dumps(to_swift_record(rec, system_prompt)) + "\n")
        print(
            f"  Stage {stage_num}: {len(train_recs)} train, {len(val_recs)} val"
            f" → {stages_dir}/stage_{stage_num}_{{train,val}}.jsonl"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert annotations.json to ms-swift JSONL stage files.")
    parser.add_argument("--config", default="config/data_config.yaml", help="Path to data_config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)

    version = cfg.get("version", "v1")
    output_dir = cfg["output_dir"]
    annotations_path = Path(output_dir) / "annotations.json"
    stages_dir = Path(output_dir).parent / "stages"

    print(f"Loading annotations from {annotations_path} ...")
    records = load_annotations(annotations_path, output_dir)
    print(f"Loaded {len(records)} records.")

    write_stage_files(
        records,
        stages_dir=stages_dir,
        val_split_ratio=cfg.get("val_split_ratio", 0.1),
        random_seed=cfg.get("random_seed", 42),
        system_prompt=cfg.get("system_prompt", ""),
    )
    print(f"Done. Stage files written to {stages_dir}/")


if __name__ == "__main__":
    main()
