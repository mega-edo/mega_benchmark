"""
IFBench Baseline — Standalone Instruction Following Agent
=========================================================

Pure agent with zero DSPy/GEPA dependencies. Reproduces the GEPA artifact's
IFBench setup (data split, metric, model) exactly; only the agent's
prompt-and-pipeline implementation is rewritten with litellm.

================================================================================
OUT-OF-SCOPE — DO NOT MODIFY (reward-hacking surface)
================================================================================

  • MODEL constant (`openai/gpt-4.1-mini-2025-04-14`).
  • Dataset loading & split: `load_ifbench()` plus JSONL files in
    `ifbench_data/`. Split, trim (150/100/300 with rng.seed(1)) and subsample
    rules MUST match GEPA `Benchmark.__init__` + `IFBench.init_dataset`.
  • Metric: `metric_with_feedback()` — the 8-variant response check and the
    instruction-following score computation are the *evaluation*, not the agent.
  • `instructions_registry` / `ifbench_checkers/` — upstream IFBench graders.
  • `evaluate()` / `evaluate_single()` — orchestration only; agent must be
    invoked through `forward()`.
  • Main-block constants: SEED, N_TRAIN, N_VAL, N_TEST.

Usage:
    uv run python3 ifbench/ifbench_test.py
"""

import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm

# Add benchmark_test/ to path so ifbench_checkers package is importable
_BENCHMARK_DIR = str(Path(__file__).parent)
if _BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARK_DIR)

from ifbench_checkers import instructions_registry  # noqa: E402


# ==================== DotDict ====================

class DotDict(dict):
    """Dict subclass with attribute access."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            )


# ==================== Dataset Loading ====================

def load_ifbench(
    seed: int = 1,
    n_train: int = 150,
    n_val: int = 100,
    n_test: int = 294,
):
    """
    Load IFBench dataset from local JSONL files.

    Replicates the split logic from IFBench.init_dataset():
    - IFBench_test.jsonl → test_set (full)
    - IFBench_train.jsonl → shuffle with seed=42, split 300 val / 300 train
    """
    data_dir = Path(__file__).parent / "ifbench_data"

    # Load test set
    test_set = []
    with open(data_dir / "IFBench_test.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            test_set.append(DotDict(**d))

    # Load train+val set
    train_val_set = []
    with open(data_dir / "IFBench_train.jsonl", "r") as f:
        for line in f:
            d = json.loads(line)
            train_val_set.append(DotDict(**d))

    # Split: ordered slice (matching IFBench.init_dataset, no shuffle)
    train_all = train_val_set[300:600]
    val_all = train_val_set[:300]
    test_all = test_set

    # Trim to 150 / 300 / 300 with rng.seed(1) (matching Benchmark.__init__)
    def trim(data, size):
        if size >= len(data):
            return data
        rng1 = random.Random(1)
        return rng1.sample(data, size)

    train_all = trim(train_all, 150)
    val_all = trim(val_all, 300)
    test_all = trim(test_all, 300)

    # Subsample with user seed
    rng = random.Random(seed)
    train_data = rng.sample(train_all, min(n_train, len(train_all)))
    val_data = rng.sample(val_all, min(n_val, len(val_all)))
    test_data = rng.sample(test_all, min(n_test, len(test_all)))

    print(f"  Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


# ==================== Metric (inlined from ifbench_metric.py) ====================

def metric_with_feedback(example, response_text):
    """
    Evaluate how well a response follows the instructions.

    Generates 8 response variants and checks each instruction against all variants.
    Returns DotDict with score (0-1) and feedback text.
    """
    r = response_text.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response_text.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")

    all_responses = [
        response_text,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]

    instruction_list = example.instruction_id_list
    is_following_list = []
    correct_feedbacks = []
    incorrect_feedbacks = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in example.kwargs[index].items() if v is not None}

        ins_text = instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=example.prompt)

        is_following = False
        for resp in all_responses:
            if resp.strip() and instruction.check_following(resp):
                is_following = True
                break

        if not is_following:
            incorrect_feedbacks.append(ins_text)
        else:
            correct_feedbacks.append(ins_text)

        is_following_list.append(is_following)

    correct_feedback_text = ""
    if len(correct_feedbacks) > 0:
        correct_feedback_text = (
            "Your response correctly followed the following instructions:\n"
            + "\n".join(correct_feedbacks)
        )

    incorrect_feedback_text = ""
    if len(incorrect_feedbacks) > 0 and len(correct_feedbacks) > 0:
        incorrect_feedback_text = (
            "However, your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )
    elif len(incorrect_feedbacks) > 0:
        incorrect_feedback_text = (
            "Your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )

    feedback_text = (correct_feedback_text + "\n" + incorrect_feedback_text).strip()

    return DotDict(
        score=sum(is_following_list) / len(is_following_list),
        feedback=feedback_text,
    )


# ==================== LLM Call Helper ====================

MODEL = "openai/gpt-4.1-mini-2025-04-14"


def _llm_call(system_prompt: str, user_prompt: str) -> str:
    """Call litellm and return raw text response (NOT JSON — IFBench needs free-form text)."""
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# ==================== The 2 Prompts ====================

GENERATE_RESPONSE_INSTRUCTION = "Respond to the query"


def generate_response(query: str, instruction: str | None = None) -> dict:
    """Stage 1: Generate initial response to the query."""
    system = instruction or GENERATE_RESPONSE_INSTRUCTION
    response = _llm_call(system, query)
    return {"response": response}


ENSURE_CORRECT_RESPONSE_INSTRUCTION = (
    "Ensure the response is correct and adheres to the given constraints. "
    "Your response will be used as the final response."
)


def ensure_correct_response(
    query: str, response: str, instruction: str | None = None
) -> dict:
    """Stage 2: Verify and correct the response for constraint adherence."""
    system = instruction or ENSURE_CORRECT_RESPONSE_INSTRUCTION
    user = f"query: {query}\n\nresponse: {response}"
    final_response = _llm_call(system, user)
    return {"final_response": final_response}


# ==================== Agent Pipeline ====================

def forward(prompt: str) -> DotDict:
    """
    Run the full 2-stage Instruction Following pipeline.

    Prompt → generate_response → ensure_correct_response → Final Response
    """
    # Stage 1: Generate initial response
    stage1 = generate_response(query=prompt)

    # Stage 2: Verify and correct
    stage2 = ensure_correct_response(query=prompt, response=stage1["response"])

    return DotDict(
        response=stage2["final_response"],
        stage1_output=stage1["response"],
        stage2_output=stage2["final_response"],
    )


# ==================== Evaluation ====================

def evaluate_single(example) -> dict:
    """Run forward pass on a single example and compute instruction following score."""
    try:
        pred = forward(example.prompt)
        result = metric_with_feedback(example, pred.response)
        return {
            "prompt": example.prompt,
            "instruction_id_list": example.instruction_id_list,
            "prediction": pred.response,
            "stage1_output": pred.stage1_output,
            "score": result.score,
            "feedback": result.feedback,
            "error": None,
        }
    except Exception as e:
        return {
            "prompt": example.get("prompt", ""),
            "instruction_id_list": example.get("instruction_id_list", []),
            "prediction": "",
            "stage1_output": "",
            "score": 0.0,
            "feedback": "",
            "error": str(e),
        }


def evaluate(dataset: list, max_workers: int = 8, label: str = "Eval") -> list[dict]:
    """Evaluate a dataset concurrently. Works with any list of examples."""
    results = []
    completed = 0
    total = len(dataset)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single, ex): i for i, ex in enumerate(dataset)}
        for future in as_completed(futures):
            result = future.result()
            results.append((futures[future], result))
            completed += 1
            if completed % 10 == 0 or completed == total:
                running_score = sum(r["score"] for _, r in results) / len(results)
                print(f"  [{label}] {completed}/{total} done, running IF score: {running_score:.3f}")

    # Sort by original index
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    perfect = sum(1 for s in scores if s >= 1.0)

    print(f"\n  {label} Results:")
    print(f"    Average IF Score: {avg_score:.4f}")
    print(f"    Perfect (1.0): {perfect}/{len(scores)}")

    return results


# ==================== Main ====================

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    SEED = 1
    N_TRAIN = 150
    N_VAL = 100
    N_TEST = 300

    print("=" * 80)
    print("INDEPENDENT IFBENCH AGENT — Standalone (no DSPy/MEGA/dspylite/GEPA)")
    print("=" * 80)
    print(f"\n  Model: {MODEL}")
    print(f"  Seed: {SEED}")
    print(f"\n  2-Stage Workflow:")
    print(f"    1. generate_response: query -> initial response")
    print(f"    2. ensure_correct_response: query + response -> corrected final response")
    print("=" * 80)

    # Load dataset
    print("\n[1/3] Loading IFBench dataset...")
    import nltk
    nltk.download("punkt_tab", quiet=True)
    train_data, val_data, test_data = load_ifbench(
        seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
    )

    # Evaluate
    print("\n[2/3] Running evaluation...")
    start_time = time.time()

    print(f"\n--- Test Set ({len(test_data)} examples) ---")
    test_results = evaluate(test_data, max_workers=8, label="Test")

    total_time = time.time() - start_time

    test_scores = [r["score"] for r in test_results]
    avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  IF Score: {avg_score:.4f}")
    print(f"  Time:     {total_time:.1f}s")
    print(f"{'=' * 80}")
