# Optimization Records

Side-by-side **baseline (`_original.py`)** and **MEGA-optimized (`_mega_optimized.py`)** agent code for four GEPA-style benchmarks. Each pair lets a third party reproduce both endpoints and verify the delta MEGA achieved.

## Layout

```
optimization-records/
├── hotpotqa/
│   ├── hotpotqa_test_original.py        # GEPA-faithful baseline (DSPy/GEPA deps removed)
│   ├── hotpotqa_test_mega_optimized.py  # MEGA-optimized agent
│   └── pyproject.toml
├── hover/
│   ├── hover_test_original.py
│   ├── hover_test_mega_optimized.py
│   └── pyproject.toml
├── ifbench/
│   ├── ifbench_test_original.py
│   ├── ifbench_checkers/                # IFBench instruction graders (vendored)
│   ├── ifbench_data/                    # IFBench train/test JSONL (vendored)
│   └── pyproject.toml
└── pupa/
    ├── pupa_test_original.py
    ├── pupa_test_mega_optimized.py
    └── pyproject.toml
```

## Naming convention

| Suffix | Meaning |
|---|---|
| `_original.py` | GEPA artifact's program/metric/data exactly mirrored, with the `dspy`/`gepa` framework dependencies stripped (litellm-only). This is the starting point of optimization. |
| `_mega_optimized.py` | Final agent code emitted by MEGA's optimization loop. Same dataset / metric / model / split as `_original.py` — only the agent's prompt-and-pipeline differs. |

The split, trim, and seed in both files reproduce the GEPA artifact's
`Benchmark.__init__` + `<task>.init_dataset` byte-for-byte. The model
(`openai/gpt-4.1-mini-2025-04-14`), the metric, and the evaluator are
identical between the two files — they sit inside an explicit
"OUT-OF-SCOPE — DO NOT MODIFY" section in each docstring.

## How to run

Each folder is a self-contained `uv` project:

```bash
cd hotpotqa
uv sync                         # install deps from pyproject.toml
export OPENAI_API_KEY=...       # litellm uses OpenAI for gpt-4.1-mini
uv run python hotpotqa_test_original.py        # baseline run
uv run python hotpotqa_test_mega_optimized.py  # optimized run
```

Each `_mega_optimized.py` writes its full per-example trace to
`<task>/results/<task>_optimized.json` on completion (model, seed,
n_test, score, time, and per-example records).

## Results — full holdout test

`_mega_optimized.py` evaluated on the full GEPA test split with
`gpt-4.1-mini-2025-04-14` at `seed=1`, `max_workers=8`. Raw outputs are in
`<task>/results/<task>_optimized.json`.

| Task | Metric | n_test | Passed | Score | Wall time |
|---|---|---:|---:|---:|---:|
| **hotpotqa** | Exact Match | 300 | 218 | **0.7267** | 249 s |
| **hover** | Binary retrieval (all 3 gold docs) | 300 | 224 | **0.7467** | 433 s |
| **ifbench** | IF score (avg fraction of constraints satisfied) | 294 | 170 perfect | **0.6105** | 674 s |
| **pupa** | (quality + (1 − leakage)) / 2  via LLM judge | 221 | 209 perfect | **0.9781** | 795 s |
