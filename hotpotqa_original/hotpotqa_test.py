"""
HotpotQA Baseline — Standalone Multi-Hop QA Agent
==================================================

Pure agent with zero DSPy/GEPA dependencies. Reproduces the GEPA artifact's
HotpotQA fullwiki setup (HuggingFace `hotpot_qa/fullwiki`, BM25s over
wiki.abstracts.2017 — same retriever as HOVER, EM metric) exactly; only
the agent's prompt-and-pipeline is rewritten with litellm.

Metric: Exact Match (EM) on the predicted answer vs. the gold answer.

================================================================================
OUT-OF-SCOPE — DO NOT MODIFY (reward-hacking surface)
================================================================================

  • MODEL constant (`openai/gpt-4.1-mini-2025-04-14`).
  • Dataset loading & split: `load_hover()` — `load_dataset("hover",
    trust_remote_code=True)`, 3-hop filter, seed=0 shuffle, ordered 40/40/20
    split, trim 150/100/300 with rng.seed(1). MUST match GEPA
    `Benchmark.__init__` + `hoverBench.init_dataset`.
  • Backend: 
    wiki.abstracts.2017 corpus, and the `gepa-artifact/.../hover/bm25s_retriever`
  • Metric: `discrete_retrieval_eval()` and `normalize_text()` (used by metric).
  • `evaluate()` / `evaluate_single()` — orchestration only; agent must be
    invoked through `forward()`.
  • Main-block constants: SEED, N_TRAIN, N_VAL, N_TEST.

Usage:
    uv run python3 hotpotqa/hotpotqa_test.py
"""

import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

import json
import re
import string
import unicodedata
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import litellm
import bm25s
import Stemmer
import ujson
from diskcache import Cache
from datasets import load_dataset


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


# ==================== BM25s Search ====================

_DATA_DIR = Path(__file__).parent / "data"

_stemmer = None
_retriever = None
_corpus = None
_initialized = False
_init_lock = threading.Lock()


def _ensure_corpus_and_index(directory: Path):
    """Download wiki.abstracts.2017 and build BM25s index if not present."""
    directory.mkdir(parents=True, exist_ok=True)
    corpus_path = directory / "wiki.abstracts.2017.jsonl"
    index_dir = directory / "bm25s_retriever"
    if corpus_path.exists() and index_dir.exists():
        return

    import urllib.request
    import tarfile

    url = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
    tar_path = directory / "wiki.abstracts.2017.tar.gz"
    print(f"  Downloading wiki.abstracts.2017 corpus...")
    urllib.request.urlretrieve(url, str(tar_path))
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(path=str(directory))

    corpus = []
    with open(str(corpus_path)) as f:
        for line in f:
            row = ujson.loads(line)
            corpus.append(f"{row['title']} | {' '.join(row['text'])}")

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(corpus_tokens)
    retriever.save(str(index_dir))


def init_retriever():
    """Thread-safe lazy initialization of BM25s retriever and corpus."""
    global _retriever, _stemmer, _corpus, _initialized
    if _initialized:
        return
    with _init_lock:
        if not _initialized:
            _ensure_corpus_and_index(_DATA_DIR)
            _retriever = bm25s.BM25.load(str(_DATA_DIR / "bm25s_retriever"))
            _stemmer = Stemmer.Stemmer("english")
            corpus_data = []
            with open(str(_DATA_DIR / "wiki.abstracts.2017.jsonl")) as f:
                for line in f:
                    row = ujson.loads(line)
                    corpus_data.append(f"{row['title']} | {' '.join(row['text'])}")
            _corpus = corpus_data
            _initialized = True


_DATA_DIR.mkdir(parents=True, exist_ok=True)
_search_cache = Cache(str(_DATA_DIR / "retriever_cache"))


@_search_cache.memoize()
def search(query: str, k: int) -> DotDict:
    """BM25s search over wiki.abstracts.2017 corpus."""
    init_retriever()
    tokens = bm25s.tokenize(query, stopwords="en", stemmer=_stemmer, show_progress=False)
    results, scores = _retriever.retrieve(tokens, k=k, n_threads=1, show_progress=False)
    run = {_corpus[doc]: float(score) for doc, score in zip(results[0], scores[0])}
    return DotDict({"passages": list(run.keys())[:k]})


# ==================== EM Metric ====================

def normalize_text(s: str) -> str:
    """Normalize text: NFD unicode, lowercase, remove articles/punctuation, fix whitespace."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em_score(prediction: str, ground_truth: str) -> bool:
    return normalize_text(prediction) == normalize_text(ground_truth)


def EM(prediction: str, answers_list: list[str]) -> bool:
    assert isinstance(answers_list, list)
    return max(em_score(prediction, ans) for ans in answers_list)


# ==================== LLM Call Helper ====================

MODEL = "openai/gpt-4.1-mini-2025-04-14"


def _llm_call(system_prompt: str, user_prompt: str) -> dict:
    """Call litellm with JSON mode and parse the response."""
    response = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": content}


# ==================== The 4 Prompts ====================

SUMMARIZE1_INSTRUCTION = (
    'Given the fields `question`, `passages`, produce the field `summary`.\n\n'
    'Respond with a JSON object containing: "reasoning" (your step-by-step '
    'chain of thought), "summary".'
)


def summarize1(question: str, passages: list[str], instruction: str | None = None) -> dict:
    """Stage 1: Summarize first retrieved docs."""
    system = instruction or SUMMARIZE1_INSTRUCTION
    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "summary"'
    passages_str = "\n".join(passages)
    user = f"Question: {question}\n\nPassages:\n{passages_str}"
    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "summary": result.get("summary", ""),
    }


CREATE_QUERY_HOP2_INSTRUCTION = (
    'Given the fields `question`, `summary_1`, produce the field `query`.\n\n'
    'Respond with a JSON object containing: "reasoning" (your step-by-step '
    'chain of thought), "query".'
)


def create_query_hop2(
    question: str, summary_1: str, instruction: str | None = None
) -> dict:
    """Stage 2: Generate query for hop2 based on question and first summary."""
    system = instruction or CREATE_QUERY_HOP2_INSTRUCTION
    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "query"'
    user = f"Question: {question}\nSummary so far: {summary_1}"
    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "query": result.get("query", ""),
    }


SUMMARIZE2_INSTRUCTION = (
    'Given the fields `question`, `context`, `passages`, produce the field `summary`.\n\n'
    'Respond with a JSON object containing: "reasoning" (your step-by-step '
    'chain of thought), "summary".'
)


def summarize2(
    question: str,
    context: str,
    passages: list[str],
    instruction: str | None = None,
) -> dict:
    """Stage 3: Summarize second hop docs with context from first hop."""
    system = instruction or SUMMARIZE2_INSTRUCTION
    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "summary"'
    passages_str = "\n".join(passages)
    user = (
        f"Question: {question}\n"
        f"Context from previous hop: {context}\n"
        f"\nPassages:\n{passages_str}"
    )
    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "summary": result.get("summary", ""),
    }


FINAL_ANSWER_INSTRUCTION = (
    'Given the fields `question`, `summary_1`, `summary_2`, produce the field `answer`.\n\n'
    'Respond with a JSON object containing: "reasoning" (your step-by-step '
    'chain of thought), "answer".'
)


def final_answer(
    question: str,
    summary_1: str,
    summary_2: str,
    instruction: str | None = None,
) -> dict:
    """Stage 4: Extract final answer from two summaries."""
    system = instruction or FINAL_ANSWER_INSTRUCTION
    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "answer"'
    user = (
        f"Question: {question}\n"
        f"Summary 1: {summary_1}\n"
        f"Summary 2: {summary_2}"
    )
    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "answer": result.get("answer", ""),
    }


# ==================== Agent Pipeline ====================

def forward(question: str, k: int = 7) -> DotDict:
    """
    Run the full 4-stage Summarize-Query pipeline.

    Question → [BM25s k=7] → summarize1 → create_query_hop2
             → [BM25s k=7] → summarize2 → final_answer → Answer
    """
    # Stage 1: Retrieve hop1 docs and summarize
    hop1_docs = search(question, k=k).passages
    summary1_result = summarize1(question=question, passages=hop1_docs)
    summary_1 = summary1_result["summary"]

    # Stage 2: Generate query for hop2
    query_result = create_query_hop2(question=question, summary_1=summary_1)
    hop2_query = query_result["query"]

    # Stage 3: Retrieve hop2 docs and summarize with context
    hop2_docs = search(hop2_query, k=k).passages
    summary2_result = summarize2(
        question=question, context=summary_1, passages=hop2_docs
    )
    summary_2 = summary2_result["summary"]

    # Stage 4: Extract final answer
    answer_result = final_answer(
        question=question, summary_1=summary_1, summary_2=summary_2
    )

    return DotDict(
        answer=answer_result["answer"],
        hop1_docs=hop1_docs,
        hop2_docs=hop2_docs,
        summary_1=summary_1,
        hop2_query=hop2_query,
        summary_2=summary_2,
        summarize1_reasoning=summary1_result["reasoning"],
        query_reasoning=query_result["reasoning"],
        summarize2_reasoning=summary2_result["reasoning"],
        answer_reasoning=answer_result["reasoning"],
    )


# ==================== Dataset Loading ====================

def load_hotpotqa(
    dataset_mode: str = "full",
    seed: int = 1,
    n_train: int = 150,
    n_val: int = 100,
    n_test: int = 300,
):
    """
    Load HotPotQA fullwiki from HuggingFace and split into train/val/test.

    Replicates the split logic from gepa-artifact's Benchmark class:
    shuffle with seed=42, split 40% test / 40% val / 20% train,
    trim to max sizes, then subsample with the given seed.
    """
    print("  Loading HotpotQA dataset from HuggingFace (fullwiki)...")
    raw = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)

    dataset_size_map = {"full": None, "lite": 500, "tiny": 200, "test": 50}
    max_size = dataset_size_map.get(dataset_mode)

    # Convert to list of DotDicts
    all_data = []
    for row in raw:
        all_data.append(DotDict(
            question=row["question"],
            answer=row["answer"],
            context=row["context"],
            supporting_facts=row["supporting_facts"],
            type=row["type"],
            level=row["level"],
        ))

    # Optional dataset_mode trim (max_testset_size in Benchmark — kept for parity but unused for "full")
    if max_size and max_size < len(all_data):
        all_data = all_data[:max_size]

    # Ordered split: 40% test, 40% val, 20% train (matching Benchmark.create_splits, no shuffle)
    total_len = len(all_data)
    test_size = int(0.4 * total_len)
    val_size = int(0.4 * total_len)
    test_all = all_data[:test_size]
    val_all = all_data[test_size : test_size + val_size]
    train_all = all_data[test_size + val_size :]

    # Trim to max set sizes with rng.seed(1) (matching Benchmark.__init__)
    def trim(data, max_n):
        if max_n >= len(data):
            return data
        rng1 = random.Random(1)
        return rng1.sample(data, max_n)

    train_all = trim(train_all, 150)
    test_all = trim(test_all, 300)
    val_all = trim(val_all, 300)

    # Subsample with user seed
    rng = random.Random(seed)
    train_data = rng.sample(train_all, min(n_train, len(train_all)))
    val_data = rng.sample(val_all, min(n_val, len(val_all)))
    test_data = rng.sample(test_all, min(n_test, len(test_all)))

    print(f"  Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


# ==================== Evaluation ====================

def evaluate_single(example: dict) -> dict:
    """Run forward pass on a single example and compute EM score."""
    try:
        pred = forward(example["question"])
        gold = example["answer"]
        answers = [gold] if isinstance(gold, str) else gold
        score = float(EM(pred.answer, answers))
        return {
            "question": example["question"],
            "gold_answer": gold,
            "prediction": pred.answer,
            "summary_1": pred.summary_1,
            "hop2_query": pred.hop2_query,
            "summary_2": pred.summary_2,
            "score": score,
            "correct": score > 0.0,
            "error": None,
        }
    except Exception as e:
        return {
            "question": example.get("question", ""),
            "gold_answer": example.get("answer", ""),
            "prediction": "",
            "summary_1": "",
            "hop2_query": "",
            "summary_2": "",
            "score": 0.0,
            "correct": False,
            "error": str(e),
        }


def evaluate(dataset: list, max_workers: int = 8, label: str = "Eval") -> list[dict]:
    """Evaluate a dataset concurrently."""
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
                running_em = sum(1 for _, r in results if r["correct"]) / len(results)
                print(f"  [{label}] {completed}/{total} done, running EM: {running_em:.3f}")

    # Sort by original index
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    scores = [r["score"] for r in results]
    em_count = sum(1 for s in scores if s > 0.0)
    avg_score = sum(scores) / len(scores) if scores else 0.0

    print(f"\n  {label} Results:")
    print(f"    Average Score: {avg_score:.4f}")
    print(f"    Exact Match: {em_count}/{len(scores)} ({em_count / len(scores):.4f})")

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
    print("HOTPOTQA MULTI-HOP AGENT — Standalone (no DSPy/MEGA/dspylite/GEPA)")
    print("=" * 80)
    print(f"\n  Model: {MODEL}")
    print(f"  Seed: {SEED}")
    print(f"\n  4-Stage Workflow (matches hotpotqa_test.py):")
    print(f"    1. summarize1:        question + passages -> summary_1")
    print(f"    2. create_query_hop2: question + summary_1 -> hop2_query")
    print(f"    3. summarize2:        question + context + passages -> summary_2")
    print(f"    4. final_answer:      question + summary_1 + summary_2 -> answer")
    print("=" * 80)

    # Load dataset
    print("\n[1/3] Loading HotpotQA dataset...")
    train_data, val_data, test_data = load_hotpotqa(
        dataset_mode="full", seed=SEED,
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
    )

    # Initialize retriever
    print("\n[2/3] Initializing BM25s retriever...")
    init_retriever()
    print(f"  Retriever loaded with {len(_corpus)} documents")

    # Evaluate
    print("\n[3/3] Running evaluation...")
    start_time = time.time()

    print(f"\n--- Test Set ({len(test_data)} examples) ---")
    test_results = evaluate(test_data, max_workers=8, label="Test")

    total_time = time.time() - start_time

    test_scores = [r["score"] for r in test_results]
    test_em = sum(1 for s in test_scores if s > 0) / len(test_scores) if test_scores else 0.0

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  Test EM:  {test_em:.4f}")
    print(f"  Time:     {total_time:.1f}s")
    print(f"{'=' * 80}")
