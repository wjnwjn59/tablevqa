"""Orchestrate multi-stage LoRA fine-tuning via ms-swift swift sft CLI."""
import argparse
import datetime
import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def load_train_config(config_path: Path) -> dict[str, Any]:
    """Load train_config.yaml and resolve output_dir to absolute path."""
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        # config lives at <repo_root>/config/; go up one extra level to reach the repo root.
        output_dir = (config_path.parent.parent / output_dir).resolve()
    cfg["output_dir"] = str(output_dir)
    return cfg


def resolve_model(cfg: dict, model_name: str | None) -> dict:
    """Return the model entry matching model_name, or active_model if None."""
    name = model_name or cfg.get("active_model")
    for m in cfg["models"]:
        if m["name"] == name:
            return m
    print(f"ERROR: model '{name}' not found in config. Available: {[m['name'] for m in cfg['models']]}")
    raise SystemExit(1)


def discover_stages(stages_dir: Path, val_dataset_flag: str) -> list[dict]:
    """Discover stage_N_train.jsonl files, return sorted list of stage dicts."""
    stages_dir = Path(stages_dir)
    train_files = list(stages_dir.glob("stage_*_train.jsonl"))
    if not train_files:
        print(f"ERROR: No stage_*_train.jsonl files found in {stages_dir}")
        raise SystemExit(1)

    def stage_num(p: Path) -> int:
        m = re.search(r"stage_(\d+)_train\.jsonl", p.name)
        return int(m.group(1))

    stages = []
    for tf in sorted(train_files, key=stage_num):
        n = stage_num(tf)
        vf = stages_dir / f"stage_{n}_val.jsonl"
        if not vf.exists():
            print(f"WARNING: No val file for stage {n} — training without validation for this stage.")
        stages.append({
            "stage_num": n,
            "train_file": str(tf.resolve()),
            "val_file": str(vf.resolve()) if vf.exists() else None,
            "val_dataset_flag": val_dataset_flag,
        })
    return stages


def build_swift_command(
    cfg: dict,
    model: dict,
    stage: dict,
    prev_checkpoint: Path | None,
    output_dir: Path,
) -> list[str]:
    """Build the swift sft CLI command for one stage."""
    lora = cfg["lora"]
    training = cfg["training"]

    model_path = str(prev_checkpoint) if prev_checkpoint else model["model_id"]

    cmd = [
        "swift", "sft",
        "--model", model_path,
        "--model_type", model["model_type"],
        "--tuner_type", "lora",
        "--dataset", stage["train_file"],
        "--lora_rank", str(lora["rank"]),
        "--lora_alpha", str(lora["alpha"]),
        "--lora_dropout", str(lora["dropout"]),
        "--lora_target_modules", str(lora["target_modules"]),
        "--torch_dtype", training["torch_dtype"],
        "--num_train_epochs", str(training["num_train_epochs"]),
        "--per_device_train_batch_size", str(training["per_device_train_batch_size"]),
        "--gradient_accumulation_steps", str(training["gradient_accumulation_steps"]),
        "--learning_rate", str(training["learning_rate"]),
        "--lr_scheduler_type", training["lr_scheduler_type"],
        "--warmup_ratio", str(training["warmup_ratio"]),
        "--save_strategy", training["save_strategy"],
        "--save_total_limit", str(training["save_total_limit"]),
        "--output_dir", str(output_dir),
    ]

    if stage["val_file"]:
        cmd += [f"--{stage['val_dataset_flag']}", stage["val_file"]]

    return cmd


def find_checkpoint(stage_output_dir: Path) -> Path:
    """Locate the single checkpoint subdirectory inside a stage output dir.

    ms-swift writes checkpoints to a subdirectory named checkpoint-<step> or last,
    not the output root itself. Selection strategy:
      1. Prefer subdirs matching checkpoint-* or last (ms-swift naming convention).
      2. Fall back to mtime (most recently modified) if no pattern match.
    This avoids picking auxiliary directories (e.g. tensorboard/) that ms-swift
    may write after the checkpoint, which would defeat a pure mtime heuristic.
    """
    stage_output_dir = Path(stage_output_dir)
    if not stage_output_dir.exists():
        print(f"ERROR: Stage output directory not found: {stage_output_dir}")
        raise SystemExit(1)
    subdirs = [p for p in stage_output_dir.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"ERROR: No checkpoint subdirectory found in {stage_output_dir}")
        raise SystemExit(1)
    # Prefer ms-swift checkpoint naming: checkpoint-<N> or last.
    ckpt_dirs = [p for p in subdirs if p.name.startswith("checkpoint-") or p.name == "last"]
    candidates = ckpt_dirs if ckpt_dirs else subdirs
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_training(cmd: list[str], log_file: Path) -> None:
    """Run swift sft subprocess, stream output to console AND append to log file.

    Uses Popen with line-by-line reading so stdout is shown in real time
    while also being captured into the log. stderr is redirected to stdout
    so both streams appear in the same log.
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(cmd)}")
    print(f"Log: {log_file}")
    with open(log_file, "a") as log:
        log.write(f"\n{'='*60}\n{' '.join(cmd)}\n{'='*60}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        print(f"ERROR: swift sft failed with exit code {proc.returncode}")
        raise SystemExit(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrate multi-stage LoRA training via swift sft.")
    parser.add_argument("--config", default="config/train_config.yaml")
    parser.add_argument("--model", default=None, help="Model name; overrides active_model in config")
    parser.add_argument("--stages-dir", default=None, help="Override path to stage JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    # Relative to process cwd; set --config explicitly if running from outside the repo root.
    config_path = Path(args.config)
    cfg = load_train_config(config_path)
    model = resolve_model(cfg, args.model)

    version = cfg["data_version"]
    base_output = Path(cfg["output_dir"])

    if args.stages_dir:
        stages_dir = Path(args.stages_dir)
    else:
        # Default assumes config lives at <repo_root>/config/train_config.yaml.
        # config_path.parent = <repo_root>/config, .parent again = <repo_root>.
        # Use --stages-dir to override if your layout differs.
        repo_root = config_path.parent.parent
        stages_dir = repo_root / "data" / version / "stages"

    if "val_dataset_flag" not in cfg["training"]:
        print("WARNING: 'val_dataset_flag' not set in train_config.yaml; defaulting to 'val_dataset'. "
              "Set this to 'eval_dataset' for ms-swift versions that require it.")
    val_flag = cfg["training"].get("val_dataset_flag", "val_dataset")
    stages = discover_stages(stages_dir, val_flag)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = base_output / model["name"] / version / f"train_{timestamp}.log"

    print(f"Model: {model['name']} | Version: {version} | Stages: {len(stages)}")

    prev_checkpoint = None
    for stage in stages:
        n = stage["stage_num"]
        stage_output_dir = base_output / model["name"] / version / f"stage_{n}"
        cmd = build_swift_command(cfg, model, stage, prev_checkpoint, stage_output_dir)

        if args.dry_run:
            print(f"\n[DRY RUN] Stage {n}:")
            print("  " + " \\\n    ".join(cmd))
            # Simulate a checkpoint for chaining in dry-run
            prev_checkpoint = stage_output_dir / "checkpoint-0"
        else:
            print(f"\n--- Stage {n}/{len(stages)} ---")
            run_training(cmd, log_file)
            prev_checkpoint = find_checkpoint(stage_output_dir)
            print(f"Stage {n} complete. Checkpoint: {prev_checkpoint}")

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
