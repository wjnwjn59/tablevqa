# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Run the full test suite (must be run from repo root — tests import `src.vlm_training.*` as a namespace package resolved via the repo root):

```bash
python3 -m pytest
```

Run a single test file or test:

```bash
python3 -m pytest tests/vlm_training/test_convert_to_swift.py
python3 -m pytest tests/vlm_training/test_train_curriculum.py::test_build_command_stage2_uses_prev_checkpoint
```

There is no `requirements.txt`/`pyproject.toml` in this repo yet — dependencies (`torch`, `transformers`, `playwright`, `pyyaml`, `pytest`) are assumed to be present in the environment already.

Convert a raw annotation set into ms-swift training JSONL:

```bash
python3 src/vlm_training/convert_to_swift.py --config config/data_config.yaml
```

Run (or dry-run) curriculum LoRA fine-tuning via ms-swift's `swift sft` CLI:

```bash
python3 src/vlm_training/train_curriculum.py --config config/train_config.yaml --model qwen3vl-2b --dry-run
python3 src/vlm_training/train_curriculum.py --config config/train_config.yaml
```

## Architecture

This project has two independent pipelines that share config-driven, path-resolution conventions but no code:

**1. Dataset generation (`src/create_big_table.py`)**
Generates synthetic financial-report tables as styled HTML (four visual variants: `visual_noise`, `charts`, `hierarchical`, `borderless_misaligned`), renders each to a PNG via Playwright (`sync_playwright`), and emits hand-authored ground-truth Q&A pairs referencing specific row IDs into a QA JSON file. This is the source of the images/questions that the inference and training pipelines consume — the four table types here correspond to the `table_types` list in `config/data_config.yaml`.

**2. VLM inference/evaluation (`src/inference.py`, `src/qwen3vl.py`, `src/intervn3_5vl.py`)**
Load a vision-language model (Qwen3-VL via `transformers.Qwen3VLForConditionalGeneration`, or InternVL3.5 via `AutoModel` + manual tile-based image preprocessing) and run single-turn image+question inference. `src/inference.py` is the batch/report-producing version (reads a QA JSON, writes `evaluation_report.csv` with substring-match accuracy); `qwen3vl.py` and `intervn3_5vl.py` are one-off smoke-test scripts with hardcoded tasks and absolute model/image paths — expect to edit constants at the top of these files directly rather than pass CLI args. The `SYSTEM_PROMPT` constant in `src/inference.py` is the authoritative prompt and must stay in sync with `system_prompt` in `config/data_config.yaml` (noted explicitly in the config comments).

**3. Curriculum LoRA training (`src/vlm_training/`)**
Two-stage pipeline driven entirely by the two YAML configs:
- `convert_to_swift.py` reads `data/<version>/raw/annotations.json` (schema: `image_path`, `question`, `answer`, `stage`, plus informational `task_type`/`difficulty`), groups records by the integer `stage` field, splits each stage train/val with a fixed seed, and writes ms-swift-format JSONL (`{"messages": [...], "images": [...]}` with a literal `<image>` token in the user message) to `data/<version>/stages/stage_<N>_{train,val}.jsonl`.
- `train_curriculum.py` discovers those stage files, then runs `swift sft` once per stage in ascending stage order, **chaining checkpoints**: stage N+1 loads `--model <checkpoint from stage N>` instead of the base model, and each stage's output goes to `runs/<model_name>/<data_version>/stage_<N>/`. Checkpoint discovery (`find_checkpoint`) prefers subdirs named `checkpoint-*` or `last` over plain mtime to avoid picking up auxiliary dirs (e.g. tensorboard logs). `--dry-run` prints the would-be `swift` commands and simulates a `checkpoint-0` path for chaining, without invoking ms-swift.

Both scripts resolve `output_dir` from the YAML as relative to the **repo root**, not the config file's own directory (config files live in `config/`, so resolution goes `config_path.parent.parent`). Model identity and `model_type` (used for the ms-swift `--model_type` flag, e.g. `qwen3_5-vl`, `internvl3_5`) are declared per-entry under `models:` in `config/train_config.yaml`; `active_model` picks the default, overridable with `--model`.

**ms-swift version sensitivity**: several config comments and one training flag (`val_dataset_flag`, defaulting to `val_dataset` but switchable to `eval_dataset`) exist because exact CLI flag names for LoRA dropout, target-module shorthand, and validation dataset differ across ms-swift versions — verify against the installed version before assuming the current flag names are correct.

## Notes

- `src/create_big_table.py` docstrings/comments are in Vietnamese; identifiers and generated output are English.
- `.gitignore` excludes `tests/`, `.claude`, `.pytest_cache`, and `docs/` — despite this, `tests/` is intentionally force-tracked in git (do not assume gitignored paths are untracked in this repo).
