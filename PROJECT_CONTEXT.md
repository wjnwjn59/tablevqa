# TableVQA — Project Context

Living notes on the research motivation and open directions for this project, meant as a starting point for future brainstorming sessions (not a finished plan). Last updated: 2026-09-15.

For the technical pipeline (how the code works, commands, file layout), see `CLAUDE.md` — this doc focuses on the "why" and where the research could go next.

## What this project is, in one paragraph

TableVQA studies how well vision-language models (currently Qwen3-VL 0.8B/2B and InternVL3.5 1B/2B) read tables from images. It has its own synthetic table generator (`src/create_big_table.py`) that renders styled HTML tables to PNGs via Playwright across four visual conditions — `visual_noise`, `charts` (in-cell sparklines/bullets/traffic-lights), `hierarchical` (multi-row headers), `borderless_misaligned` — pairs them with hand-authored ground-truth Q&A, and feeds that into inference/evaluation scripts and a curriculum-style LoRA fine-tuning pipeline (`src/vlm_training/`, driven by `swift sft`).

## Research motivation

The starting question: **existing table-VQA benchmarks are missing some dimension of difficulty that matters for how these models actually fail**, and the goal is to find that dimension, demonstrate it causes real robustness gaps in current LVLMs, and then decide what to do about it (new benchmark axis, curriculum training, or inference-time mitigation).

## Candidate directions raised so far (2026-09-15 discussion)

1. **Table size / column-count as a robustness axis.** Compare LVLM accuracy on small tables (1–2 columns) vs. very wide ones (~100 columns). Hypothesis: accuracy degrades non-trivially with size/density in a way current benchmarks don't isolate. If confirmed, this is a natural fit for **curriculum learning** — order training stages by increasing table size/complexity, which the pipeline already supports structurally (see below).

2. **Test-time scaling.** Instead of (or in addition to) retraining, explore inference-time strategies — self-consistency/multiple sampling, chain-of-thought reasoning over the table, region cropping/zooming into the relevant part of a large table — to see whether extra inference compute recovers accuracy on the hard cases, without touching model weights.

3. **Benchmark gap-finding via the existing generator.** Use `create_big_table.py` as a probing tool: generate additional/novel stressors beyond the current four types, run them against current LVLMs, and see what breaks that isn't already covered by existing table-VQA benchmarks. The output would be a candidate new benchmark axis/dataset rather than a training method.

These three aren't mutually exclusive — direction 3 could surface *what* axis to scale in directions 1/2.

## How the existing code already supports this

- The `stage` field in `annotations.json` (consumed by `convert_to_swift.py` → `train_curriculum.py`) is generic curriculum-ordering infrastructure already wired up for multi-stage LoRA training — it doesn't currently encode "difficulty by table size," but could without new pipeline code, just new data generation + stage assignment.
- `src/create_big_table.py`'s four `case_type` branches are all fixed-size (20/40/80/50 rows) with a fixed column layout per type — there's no existing `num_columns` parameter, so a size-sweep would need a new generation path (or a new `table_types` entry) rather than a config tweak.
- `src/inference.py` already reports accuracy broken down by `task_type` in `evaluation_report.csv` — extending that grouping to bucket by column count (or another new axis) would be the fastest way to get baseline numbers before committing to a training-based intervention.

## Open questions for future sessions

- What precisely defines "size/difficulty" here — column count, row count, cell density, rendered pixel width, or some combination?
- No baseline numbers exist yet in this repo for any size/robustness axis — first concrete step is likely a sweep: generate tables across a size range, run existing inference against them, and see if there's a real effect before designing curriculum stages around it.
- Is the target LVLM set staying at small models (Qwen3-VL 0.8B/2B, InternVL3.5 1B/2B), or should larger/proprietary models be added as ceiling references to know how much of the gap is "small model" vs. "fundamental table-VQA" weakness?
- If direction 3 (benchmark gap-finding) turns up a new stressor, does it replace or extend the current four `table_types`?
