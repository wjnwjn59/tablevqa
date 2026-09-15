import pytest, yaml, json
from pathlib import Path


def write_train_config(tmp_path, overrides=None):
    cfg = {
        "data_version": "v1",
        "active_model": "modelA",
        "models": [
            {"name": "modelA", "model_id": "/mnt/modelA", "model_type": "qwen3_5-vl"},
            {"name": "modelB", "model_id": "/mnt/modelB", "model_type": "internvl3_5"},
        ],
        "output_dir": "runs/",
        "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "target_modules": "all-linear"},
        "training": {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "torch_dtype": "bfloat16",
            "val_dataset_flag": "val_dataset",
        },
    }
    if overrides:
        cfg.update(overrides)
    config_file = tmp_path / "train_config.yaml"
    config_file.write_text(yaml.dump(cfg))
    return config_file


def test_load_train_config_resolves_output_dir(tmp_path):
    from src.vlm_training.train_curriculum import load_train_config
    config_file = write_train_config(tmp_path)
    cfg = load_train_config(config_file)
    assert Path(cfg["output_dir"]).is_absolute()


def test_resolve_model_by_name(tmp_path):
    from src.vlm_training.train_curriculum import load_train_config, resolve_model
    config_file = write_train_config(tmp_path)
    cfg = load_train_config(config_file)
    model = resolve_model(cfg, "modelB")
    assert model["model_id"] == "/mnt/modelB"
    assert model["model_type"] == "internvl3_5"


def test_resolve_model_uses_active_model_by_default(tmp_path):
    from src.vlm_training.train_curriculum import load_train_config, resolve_model
    config_file = write_train_config(tmp_path)
    cfg = load_train_config(config_file)
    model = resolve_model(cfg, None)
    assert model["name"] == "modelA"


def test_resolve_model_unknown_name_raises(tmp_path):
    from src.vlm_training.train_curriculum import load_train_config, resolve_model
    config_file = write_train_config(tmp_path)
    cfg = load_train_config(config_file)
    with pytest.raises(SystemExit):
        resolve_model(cfg, "nonexistent")


def test_discover_stages_numeric_order(tmp_path):
    from src.vlm_training.train_curriculum import discover_stages
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    for n in [1, 10, 2, 9]:
        (stages_dir / f"stage_{n}_train.jsonl").touch()
        (stages_dir / f"stage_{n}_val.jsonl").touch()
    result = discover_stages(stages_dir, "val_dataset")
    assert [s["stage_num"] for s in result] == [1, 2, 9, 10]


def test_discover_stages_empty_raises(tmp_path):
    from src.vlm_training.train_curriculum import discover_stages
    stages_dir = tmp_path / "empty"
    stages_dir.mkdir()
    with pytest.raises(SystemExit):
        discover_stages(stages_dir, "val_dataset")


def make_cfg(tmp_path):
    config_file = write_train_config(tmp_path)
    from src.vlm_training.train_curriculum import load_train_config
    return load_train_config(config_file)


def test_build_command_stage1_uses_base_model(tmp_path):
    from src.vlm_training.train_curriculum import build_swift_command
    cfg = make_cfg(tmp_path)
    model = cfg["models"][0]
    stage = {"stage_num": 1, "train_file": "/data/stage_1_train.jsonl", "val_file": "/data/stage_1_val.jsonl", "val_dataset_flag": "val_dataset"}
    cmd = build_swift_command(cfg, model, stage, prev_checkpoint=None, output_dir=Path("/runs/modelA/v1/stage_1"))
    assert "--model" in cmd
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "/mnt/modelA"
    assert "--tuner_type" in cmd
    assert cmd[cmd.index("--tuner_type") + 1] == "lora"
    assert "--output_dir" in cmd


def test_build_command_stage2_uses_prev_checkpoint(tmp_path):
    from src.vlm_training.train_curriculum import build_swift_command
    cfg = make_cfg(tmp_path)
    model = cfg["models"][0]
    stage = {"stage_num": 2, "train_file": "/data/stage_2_train.jsonl", "val_file": "/data/stage_2_val.jsonl", "val_dataset_flag": "val_dataset"}
    prev_ckpt = Path("/runs/modelA/v1/stage_1/checkpoint-100")
    cmd = build_swift_command(cfg, model, stage, prev_checkpoint=prev_ckpt, output_dir=Path("/runs/modelA/v1/stage_2"))
    idx = cmd.index("--model")
    assert cmd[idx + 1] == str(prev_ckpt)
    # model_type must still be present when loading from a checkpoint
    assert "--model_type" in cmd
    # Must NOT use --resume_from_checkpoint (would restore optimizer state)
    assert "--resume_from_checkpoint" not in cmd


def test_build_command_includes_lora_args(tmp_path):
    from src.vlm_training.train_curriculum import build_swift_command
    cfg = make_cfg(tmp_path)
    model = cfg["models"][0]
    stage = {"stage_num": 1, "train_file": "/t.jsonl", "val_file": None, "val_dataset_flag": "val_dataset"}
    cmd = build_swift_command(cfg, model, stage, prev_checkpoint=None, output_dir=Path("/out"))
    assert "--lora_rank" in cmd
    assert "--lora_alpha" in cmd
    assert "--torch_dtype" in cmd
    assert "bfloat16" in cmd


def test_build_command_skips_val_when_none(tmp_path):
    from src.vlm_training.train_curriculum import build_swift_command
    cfg = make_cfg(tmp_path)
    model = cfg["models"][0]
    stage = {"stage_num": 1, "train_file": "/t.jsonl", "val_file": None, "val_dataset_flag": "val_dataset"}
    cmd = build_swift_command(cfg, model, stage, prev_checkpoint=None, output_dir=Path("/out"))
    assert "--val_dataset" not in cmd
    assert "--eval_dataset" not in cmd


def test_build_command_uses_val_dataset_flag_from_config(tmp_path):
    """val_dataset_flag=eval_dataset in config must produce --eval_dataset in command."""
    from src.vlm_training.train_curriculum import build_swift_command, load_train_config
    import copy
    config_file = write_train_config(tmp_path, overrides={
        "training": {
            "num_train_epochs": 1, "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1, "learning_rate": 1e-4,
            "lr_scheduler_type": "cosine", "warmup_ratio": 0.05,
            "save_strategy": "epoch", "save_total_limit": 1,
            "torch_dtype": "bfloat16", "val_dataset_flag": "eval_dataset",
        }
    })
    cfg = load_train_config(config_file)
    model = cfg["models"][0]
    stage = {"stage_num": 1, "train_file": "/t.jsonl", "val_file": "/v.jsonl", "val_dataset_flag": "eval_dataset"}
    cmd = build_swift_command(cfg, model, stage, prev_checkpoint=None, output_dir=Path("/out"))
    assert "--eval_dataset" in cmd
    assert "--val_dataset" not in cmd


def test_find_checkpoint_returns_single_subdir(tmp_path):
    from src.vlm_training.train_curriculum import find_checkpoint
    stage_dir = tmp_path / "stage_1"
    ckpt = stage_dir / "checkpoint-150"
    ckpt.mkdir(parents=True)
    result = find_checkpoint(stage_dir)
    assert result == ckpt


def test_find_checkpoint_missing_stage_dir_raises(tmp_path):
    from src.vlm_training.train_curriculum import find_checkpoint
    with pytest.raises(SystemExit):
        find_checkpoint(tmp_path / "nonexistent")


def test_find_checkpoint_no_subdir_raises(tmp_path):
    from src.vlm_training.train_curriculum import find_checkpoint
    stage_dir = tmp_path / "stage_1"
    stage_dir.mkdir()
    with pytest.raises(SystemExit):
        find_checkpoint(stage_dir)


def test_find_checkpoint_multiple_subdirs_returns_most_recent(tmp_path):
    """When ms-swift emits multiple subdirs, the most recently modified is returned.
    Assumption: ms-swift with save_total_limit=1 keeps one checkpoint dir;
    this test guards against auxiliary dirs (e.g. tensorboard) appearing alongside it.
    """
    import time
    from src.vlm_training.train_curriculum import find_checkpoint
    stage_dir = tmp_path / "stage_1"
    old_dir = stage_dir / "checkpoint-50"
    old_dir.mkdir(parents=True)
    time.sleep(0.01)  # ensure mtime difference
    new_dir = stage_dir / "checkpoint-100"
    new_dir.mkdir()
    result = find_checkpoint(stage_dir)
    assert result == new_dir


def test_run_training_calls_subprocess(tmp_path, monkeypatch):
    """run_training should call Popen with the given command and write output to log."""
    from src.vlm_training import train_curriculum
    import io
    calls = []
    class FakeProc:
        returncode = 0
        stdout = iter(["line1\n", "line2\n"])
        def wait(self): pass
    def fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return FakeProc()
    monkeypatch.setattr(train_curriculum.subprocess, "Popen", fake_popen)
    log_file = tmp_path / "train.log"
    train_curriculum.run_training(["swift", "sft", "--model", "x"], log_file)
    assert calls[0][:2] == ["swift", "sft"]
    assert "line1" in log_file.read_text()


def test_run_training_raises_on_nonzero(tmp_path, monkeypatch):
    from src.vlm_training import train_curriculum
    class FakeProc:
        returncode = 1
        stdout = iter([])
        def wait(self): pass
    monkeypatch.setattr(train_curriculum.subprocess, "Popen", lambda *a, **kw: FakeProc())
    log_file = tmp_path / "train.log"
    with pytest.raises(SystemExit):
        train_curriculum.run_training(["swift", "sft"], log_file)


import subprocess as _subprocess, sys

def make_stage_files(stages_dir: Path, stage_nums: list[int]):
    stages_dir.mkdir(parents=True, exist_ok=True)
    for n in stage_nums:
        (stages_dir / f"stage_{n}_train.jsonl").write_text(
            json.dumps({"messages": [], "images": []}) + "\n"
        )
        (stages_dir / f"stage_{n}_val.jsonl").write_text(
            json.dumps({"messages": [], "images": []}) + "\n"
        )


def test_cli_dry_run_prints_commands(tmp_path, monkeypatch):
    """With --dry-run flag, CLI prints swift commands without executing."""
    config_file = write_train_config(tmp_path)
    stages_dir = tmp_path / "data/v1/stages"
    make_stage_files(stages_dir, [1, 2])
    result = _subprocess.run(
        [sys.executable, "src/vlm_training/train_curriculum.py",
         "--config", str(config_file),
         "--stages-dir", str(stages_dir),
         "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert "swift" in result.stdout
    # Stage 2 command must reference the simulated stage 1 checkpoint path
    assert "checkpoint-0" in result.stdout


def test_cli_model_flag_overrides_active_model(tmp_path):
    """--model CLI flag must select modelB instead of active_model (modelA)."""
    config_file = write_train_config(tmp_path)
    stages_dir = tmp_path / "stages"
    make_stage_files(stages_dir, [1])
    result = _subprocess.run(
        [sys.executable, "src/vlm_training/train_curriculum.py",
         "--config", str(config_file),
         "--stages-dir", str(stages_dir),
         "--model", "modelB",
         "--dry-run"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parents[2]),
    )
    assert result.returncode == 0, result.stderr
    assert "/mnt/modelB" in result.stdout
    assert "/mnt/modelA" not in result.stdout
