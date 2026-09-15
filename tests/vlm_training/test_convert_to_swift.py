import pytest
import yaml
import json
from pathlib import Path
import tempfile, os
import subprocess, sys

# We import the module under test once it exists
# Tests run against the public functions directly


def write_config(tmp_path, overrides=None):
    """Helper: write a minimal data_config.yaml and return its path.

    Places config at tmp_path/config/data_config.yaml so that
    config_path.parent.parent == tmp_path, matching the real repo layout
    where config lives at <repo_root>/config/.
    """
    cfg = {
        "version": "v1",
        "output_dir": "data/v1/raw/",
        "val_split_ratio": 0.2,
        "random_seed": 0,
        "system_prompt": "Answer concisely.",
    }
    if overrides:
        cfg.update(overrides)
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "data_config.yaml"
    config_file.write_text(yaml.dump(cfg))
    return config_file


def test_load_config_resolves_output_dir_to_absolute(tmp_path):
    """output_dir relative to config file location becomes absolute."""
    from src.vlm_training.convert_to_swift import load_config
    config_file = write_config(tmp_path, {"output_dir": "data/v1/raw/"})
    cfg = load_config(config_file)
    assert Path(cfg["output_dir"]).is_absolute()
    assert cfg["output_dir"] == str(tmp_path / "data/v1/raw")


def test_load_config_absolute_output_dir_unchanged(tmp_path):
    """Absolute output_dir is returned as-is."""
    from src.vlm_training.convert_to_swift import load_config
    abs_dir = str(tmp_path / "abs/raw")
    config_file = write_config(tmp_path, {"output_dir": abs_dir})
    cfg = load_config(config_file)
    assert cfg["output_dir"] == abs_dir


def test_load_config_preserves_other_fields(tmp_path):
    from src.vlm_training.convert_to_swift import load_config
    config_file = write_config(tmp_path)
    cfg = load_config(config_file)
    assert cfg["val_split_ratio"] == 0.2
    assert cfg["random_seed"] == 0
    assert cfg["system_prompt"] == "Answer concisely."


def write_annotations(raw_dir: Path, records: list) -> Path:
    path = raw_dir / "annotations.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records))
    return path


def test_load_annotations_resolves_bare_filename(tmp_path):
    """Bare filename image_path is resolved to absolute using output_dir."""
    from src.vlm_training.convert_to_swift import load_annotations
    raw_dir = tmp_path / "data/v1/raw"
    records = [{"image_path": "img.png", "question": "Q?", "answer": "A", "stage": 1}]
    write_annotations(raw_dir, records)
    result = load_annotations(raw_dir / "annotations.json", str(raw_dir))
    assert result[0]["image_path"] == str(raw_dir / "img.png")


def test_load_annotations_keeps_absolute_path(tmp_path):
    """Absolute image_path is left unchanged."""
    from src.vlm_training.convert_to_swift import load_annotations
    raw_dir = tmp_path / "data/v1/raw"
    abs_path = str(tmp_path / "other/img.png")
    records = [{"image_path": abs_path, "question": "Q?", "answer": "A", "stage": 1}]
    write_annotations(raw_dir, records)
    result = load_annotations(raw_dir / "annotations.json", str(raw_dir))
    assert result[0]["image_path"] == abs_path


def test_load_annotations_missing_file_raises(tmp_path):
    from src.vlm_training.convert_to_swift import load_annotations
    with pytest.raises(SystemExit):
        load_annotations(tmp_path / "missing.json", str(tmp_path))


def test_load_annotations_missing_stage_field_raises(tmp_path):
    """Mixed list: one valid record then one without 'stage' — must still raise."""
    from src.vlm_training.convert_to_swift import load_annotations
    raw_dir = tmp_path / "raw"
    records = [
        {"image_path": "img0.png", "question": "Q?", "answer": "A", "stage": 1},
        {"image_path": "img1.png", "question": "Q?", "answer": "A"},  # missing stage
    ]
    write_annotations(raw_dir, records)
    with pytest.raises(SystemExit):
        load_annotations(raw_dir / "annotations.json", str(raw_dir))


def make_records(stages_counts: dict) -> list:
    """Build synthetic annotation records. stages_counts = {stage_int: count}."""
    records = []
    for stage, count in stages_counts.items():
        for i in range(count):
            records.append({
                "image_path": f"/abs/img_{stage}_{i}.png",
                "question": f"Q{i}?",
                "answer": f"A{i}",
                "task_type": "Hierarchical Header",
                "difficulty": 1,
                "stage": stage,
            })
    return records


def test_group_by_stage_returns_sorted_stages(tmp_path):
    from src.vlm_training.convert_to_swift import group_by_stage
    records = make_records({3: 2, 1: 3, 2: 1})
    groups = group_by_stage(records)
    assert list(groups.keys()) == [1, 2, 3]
    assert len(groups[1]) == 3
    assert len(groups[3]) == 2


def test_split_stage_produces_correct_counts(tmp_path):
    from src.vlm_training.convert_to_swift import split_stage
    records = make_records({1: 10})[:]
    train, val = split_stage(records, val_ratio=0.2, seed=42)
    assert len(train) == 8
    assert len(val) == 2
    assert set(r["image_path"] for r in train).isdisjoint(
        set(r["image_path"] for r in val)
    )


def test_split_stage_is_reproducible(tmp_path):
    from src.vlm_training.convert_to_swift import split_stage
    records = make_records({1: 20})[:]
    t1, v1 = split_stage(records, val_ratio=0.2, seed=7)
    t2, v2 = split_stage(records, val_ratio=0.2, seed=7)
    assert [r["image_path"] for r in t1] == [r["image_path"] for r in t2]


def test_to_swift_record_format(tmp_path):
    from src.vlm_training.convert_to_swift import to_swift_record
    rec = {"image_path": "/abs/img.png", "question": "What?", "answer": "42", "stage": 1}
    result = to_swift_record(rec, system_prompt="Answer concisely.")
    assert result["messages"][0] == {"role": "system", "content": "Answer concisely."}
    assert result["messages"][1] == {"role": "user", "content": "<image>Question: What?"}
    assert result["messages"][2] == {"role": "assistant", "content": "42"}
    assert result["images"] == ["/abs/img.png"]


def test_write_stage_files_creates_jsonl(tmp_path):
    from src.vlm_training.convert_to_swift import write_stage_files
    records = make_records({1: 5, 2: 5})
    stages_dir = tmp_path / "stages"
    write_stage_files(
        records,
        stages_dir=stages_dir,
        val_split_ratio=0.2,
        random_seed=0,
        system_prompt="Answer.",
    )
    assert (stages_dir / "stage_1_train.jsonl").exists()
    assert (stages_dir / "stage_1_val.jsonl").exists()
    assert (stages_dir / "stage_2_train.jsonl").exists()
    train_lines = (stages_dir / "stage_1_train.jsonl").read_text().strip().splitlines()
    val_lines = (stages_dir / "stage_1_val.jsonl").read_text().strip().splitlines()
    # 5 records * val_ratio=0.2 → n_val = max(1, round(1.0)) = 1; train = 4
    assert len(train_lines) == 4
    assert len(val_lines) == 1
    # Each line must be valid JSON with messages (system/user/assistant) and images
    for line in train_lines:
        obj = json.loads(line)
        assert "messages" in obj and "images" in obj
        roles = [m["role"] for m in obj["messages"]]
        assert roles == ["system", "user", "assistant"]


def test_cli_produces_stage_files(tmp_path):
    """End-to-end: run convert_to_swift.py as a script and check output files."""
    raw_dir = tmp_path / "data/v1/raw"
    raw_dir.mkdir(parents=True)
    # Write annotations.json
    records = make_records({1: 5, 2: 5})
    (raw_dir / "annotations.json").write_text(json.dumps(records))
    # Write config pointing at tmp_path
    cfg = {
        "version": "v1",
        "output_dir": str(raw_dir),
        "val_split_ratio": 0.2,
        "random_seed": 42,
        "system_prompt": "Answer.",
    }
    config_file = tmp_path / "data_config.yaml"
    config_file.write_text(yaml.dump(cfg))
    # Run CLI
    result = subprocess.run(
        [sys.executable, "src/vlm_training/convert_to_swift.py", "--config", str(config_file)],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parents[2]),  # repo root
    )
    assert result.returncode == 0, result.stderr
    stages_dir = raw_dir.parent / "stages"
    assert (stages_dir / "stage_1_train.jsonl").exists()
    assert (stages_dir / "stage_2_val.jsonl").exists()
