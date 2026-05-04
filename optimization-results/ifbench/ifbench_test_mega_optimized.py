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

GENERATE_RESPONSE_INSTRUCTION = """You must respond to the user's query while obeying EVERY constraint embedded in it.

Follow this 6-step protocol:

1. ENUMERATE — List every constraint you find in the query. Classify each as one of:
   [quantitative] exact counts, ranges, min/max (words, sentences, paragraphs, bullets, etc.)
   [structural] formatting requirements (lists, headings, JSON, tables, etc.)
   [positional] first word, last word, start/end of response requirements
   [keyword] must-include or must-exclude words/phrases
   [format] language, case, punctuation, markdown style
   [repetition] repeat the prompt, repeat a phrase, echo back
   [custom] any other constraint

2. PLAN — For each constraint, decide how to satisfy it. For quantitative constraints, create a numbered slot skeleton (e.g., "1. ___ 2. ___ 3. ___") to guarantee the exact count.

3. DRAFT — Write your response, filling in the skeleton.

4. SELF-VERIFY — Go through each constraint from step 1. For quantitative constraints, count token-by-token (do NOT trust your intuition — count explicitly: "1: ..., 2: ..., 3: ..."). Check every single constraint is met.

5. REPAIR — If any constraint is violated, fix it now. For off-by-one count errors, add or remove exactly the delta needed.

6. OUTPUT — Wrap your final response in sentinel markers:
<<<ANSWER>>>
(your final response here)
<<<END>>>

COUNTING ENFORCEMENT: For ANY constraint involving an exact count (number of sentences, paragraphs, words, bullets, items, etc.), you MUST:
- Create a numbered slot skeleton in the PLAN step
- Enumerate and count each item explicitly in SELF-VERIFY
- If the count is off by any delta, add or remove items to match exactly

REPETITION RULE: If asked to repeat or include verbatim text, copy it character-for-character. Do NOT paraphrase, summarize, or reword — use the exact original text."""


def _extract_answer(text: str) -> str:
    """Extract the last <<<ANSWER>>>...<<<END>>> block. Falls back to raw text if missing."""
    marker_start = "<<<ANSWER>>>"
    marker_end = "<<<END>>>"
    start_idx = text.rfind(marker_start)
    if start_idx == -1:
        return text.strip()
    end_idx = text.rfind(marker_end)
    if end_idx == -1 or end_idx <= start_idx:
        return text[start_idx + len(marker_start):].strip()
    return text[start_idx + len(marker_start):end_idx].strip()


def generate_response(query: str, instruction: str | None = None) -> dict:
    """Stage 1: Generate initial response to the query."""
    system = instruction or GENERATE_RESPONSE_INSTRUCTION
    response = _llm_call(system, query)
    return {"response": response, "raw": response}


ENSURE_CORRECT_RESPONSE_INSTRUCTION = """You are a deterministic constraint auditor. Your corrected response will be used as the FINAL output.

CRITICAL RULE: Make the MINIMUM possible changes. If a constraint is already satisfied, do NOT touch that part of the response. Only fix what is actually broken. Rewriting satisfied parts risks introducing NEW violations.

Follow this protocol exactly:

1. EXTRACT — List every constraint from the original query.

2. AUDIT — Check the draft response against EACH constraint one by one.
   - For quantitative constraints: count token-by-token. Do NOT trust the draft's claims. Count explicitly: "Item 1: ..., Item 2: ..., Item 3: ..." and state the total.
   - For keyword constraints: search for exact matches.
   - For positional constraints: check the exact position.
   - Mark each constraint as PASS or FAIL with evidence.

3. REPAIR — For each FAIL (and ONLY for FAILs):
   - Make the smallest surgical edit to fix the violation.
   - For count errors: add or remove exactly the delta needed.
   - For keyword errors: insert or remove the exact word/phrase.
   - For repetition constraints: copy the required text verbatim, character-for-character.
   - NEVER rewrite parts that already PASS — leave them exactly as-is.

4. OUTPUT — Wrap the corrected final response in sentinel markers:
<<<ANSWER>>>
(corrected final response here — no scratchpad, no explanations, ONLY the response)
<<<END>>>

COUNTING ENFORCEMENT: For ANY quantitative constraint, you MUST enumerate and count each item explicitly. If the count is wrong, fix it by adding/removing exactly the delta.

REPETITION RULE: When the query asks to repeat or include verbatim text, the response MUST contain an exact character-for-character copy. Do NOT paraphrase."""


def ensure_correct_response(
    query: str, response: str, instruction: str | None = None
) -> dict:
    """Stage 2: Verify and correct the response for constraint adherence."""
    system = instruction or ENSURE_CORRECT_RESPONSE_INSTRUCTION
    user = f"query: {query}\n\nresponse: {response}"
    final_response = _llm_call(system, user)
    return {"final_response": final_response, "raw": final_response}


# ==================== Agent Pipeline ====================

def forward(prompt: str) -> DotDict:
    """
    Run the full 2-stage Instruction Following pipeline.

    Prompt → generate_response → _extract_answer → ensure_correct_response → _extract_answer → Final
    """
    stage1 = generate_response(query=prompt)
    stage1_extracted = _extract_answer(stage1["response"])

    stage2 = ensure_correct_response(query=prompt, response=stage1_extracted)
    stage2_extracted = _extract_answer(stage2["final_response"])

    return DotDict(
        response=stage2_extracted,
        stage1_output=stage1_extracted,
        stage2_output=stage2_extracted,
        stage1_raw=stage1["raw"],
        stage2_raw=stage2["raw"],
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
    print("\n[1/2] Loading IFBench dataset...")
    import nltk
    nltk.download("punkt_tab", quiet=True)
    train_data, val_data, test_data = load_ifbench(
        seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
    )

    # Evaluate
    print("\n[2/2] Running evaluation...")
    start_time = time.time()

    print(f"\n--- Test Set ({len(test_data)} examples) ---")
    test_results = evaluate(test_data, max_workers=8, label="Test")

    total_time = time.time() - start_time

    test_scores = [r["score"] for r in test_results]
    avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0

    perfect = sum(1 for s in test_scores if s >= 1.0)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  IF Score: {avg_score:.4f}")
    print(f"  Perfect:  {perfect}/{len(test_scores)}")
    print(f"  Time:     {total_time:.1f}s")
    print(f"{'=' * 80}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = results_dir / f"ifbench_optimized_{timestamp}.json"
    summary = {
        "model": MODEL,
        "seed": SEED,
        "n_test": len(test_scores),
        "if_score": avg_score,
        "perfect": perfect,
        "time_seconds": total_time,
        "timestamp": timestamp,
        "per_example": test_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved results -> {out_path}")
