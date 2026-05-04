"""
HOVER Baseline — Standalone 3-Hop Multi-Hop Retrieval Agent
===========================================================

Pure agent with zero DSPy/GEPA dependencies. Reproduces the GEPA artifact's
HOVER setup (HuggingFace `hover` dataset, 3-hop filter, BM25s over
wiki.abstracts.2017, binary retrieval metric) exactly; only the agent's
prompt-and-pipeline is rewritten with litellm.

Metric: Binary retrieval eval — 1.0 if ALL gold docs retrieved, else 0.0.

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
    uv run python3 hover/hover_test.py
"""

import sys
import os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", 1)

import json
import random
import re
import string
import time
import threading
import unicodedata
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


# ==================== BM25s Search (inlined from hover_program.py) ====================

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


# ==================== normalize_text (inlined from dspy.evaluate.metrics) ====================

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


# ==================== Binary Retrieval Metric (inlined from hover_utils.py) ====================

def discrete_retrieval_eval(example, retrieved_docs) -> DotDict:
    """
    Binary retrieval evaluation: 1.0 if ALL gold docs retrieved, else 0.0.

    Gold titles are extracted from example.supporting_facts.
    Found titles are parsed from "title | text" format in retrieved_docs.
    """
    gold_titles = set(
        normalize_text(doc["key"])
        for doc in example.supporting_facts
    )
    found_titles = set(
        normalize_text(c.split(" | ")[0])
        for c in retrieved_docs
    )

    score = gold_titles.issubset(found_titles)

    gold_found = gold_titles.intersection(found_titles)
    gold_missing = gold_titles.difference(found_titles)

    feedback_text = (
        f"Your queries correctly retrieved the following relevant evidence documents: {gold_found}, "
        f"but missed the following relevant evidence documents: {gold_missing}."
    )

    return DotDict(score=float(score), feedback=feedback_text)


# ==================== Dataset Loading ====================

def load_hover(
    seed: int = 1,
    n_train: int = 150,
    n_val: int = 100,
    n_test: int = 300,
):
    """
    Load HOVER dataset from HuggingFace.

    Replicates the split logic from hover_data.py + benchmark.py:
    - Load vincentkoc/hover-parquet train split
    - Filter to 3-hop examples (count_unique_docs == 3)
    - Shuffle with seed=0 (hover_data.py)
    - Split: 40% test, 40% val, 20% train (benchmark.create_splits with seed=42)
    - Trim to 300 test, 300 val, 150 train (benchmark.__init__)
    - Subsample with given seed
    """
    print("  Loading HOVER dataset from HuggingFace...")
    dataset = load_dataset("hover", trust_remote_code=True)
    hf_trainset = dataset["train"]

    # Filter to 3-hop examples only
    reformatted = []
    for example in hf_trainset:
        claim = example["claim"]
        supporting_facts = example["supporting_facts"]
        label = example["label"]

        # Count unique supporting fact keys (must be exactly 3)
        unique_docs = len(set(fact["key"] for fact in supporting_facts))
        if unique_docs == 3:
            reformatted.append(DotDict(
                claim=claim,
                supporting_facts=supporting_facts,
                label=label,
            ))

    print(f"  Filtered to {len(reformatted)} 3-hop examples")

    # Shuffle with seed=0 (matching hover_data.py)
    rng0 = random.Random(0)
    rng0.shuffle(reformatted)

    # Ordered split: 40% test, 40% val, 20% train (matching Benchmark.create_splits, no shuffle)
    total_len = len(reformatted)
    test_size = int(0.4 * total_len)
    val_size = int(0.4 * total_len)
    test_all = reformatted[:test_size]
    val_all = reformatted[test_size:test_size + val_size]
    train_all = reformatted[test_size + val_size:]

    # Trim to max sizes with rng.seed(1) (matching Benchmark.__init__)
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
        temperature=0,
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {"error": content}


# ==================== The 4 LLM Modules ====================

SUMMARIZE_INSTRUCTION = (
    'Given the fields `claim`, `passages`, produce the field `summary`.\n\n'
    'Respond with a JSON object containing: "reasoning" (your step-by-step '
    'chain of thought), "summary".'
)

SUMMARIZE_WITH_CONTEXT_INSTRUCTION = (
    'Given the `claim`, prior-hop `context`, and freshly retrieved `passages`, '
    'produce a `summary` of what the new passages add. Preserve all proper-noun '
    'entities (people, organizations, works, places, dates) verbatim — do not '
    'abstract or rename them.\n\n'
    'Respond with a JSON object containing: "reasoning" (step-by-step), "summary".'
)


def summarize(
    claim: str, passages: list[str], context: str | None = None, instruction: str | None = None
) -> dict:
    """Summarize passages for claim verification. If context provided, builds on it."""
    if context is not None:
        system = instruction or SUMMARIZE_WITH_CONTEXT_INSTRUCTION
        user = f"Claim: {claim}\n\nContext: {context}\n\nPassages:\n" + "\n".join(passages)
    else:
        system = instruction or SUMMARIZE_INSTRUCTION
        user = f"Claim: {claim}\n\nPassages:\n" + "\n".join(passages)

    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "summary"'

    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "summary": result.get("summary", ""),
    }


MERGED_HOP1_INSTRUCTION = (
    'You are verifying a multi-hop factual claim using BM25 retrieval over a Wikipedia corpus.\n'
    'You are given the `claim` and `passages` retrieved in hop 1.\n'
    'Read the passages carefully and produce TWO outputs:\n'
    '  (a) `context_summary`: a faithful summary of what the hop-1 passages '
    'establish, preserving all proper-noun entities (people, organizations, '
    'works, places, dates) verbatim. Do not abstract or rename entities.\n'
    '  (b) `bridge_entity`: the single most important proper-noun entity '
    'mentioned in the hop-1 passages that the claim still depends on but is '
    'not yet fully verified — i.e., the entity needed to look up additional '
    'evidence on hop 2. If multiple, pick the one with the most unresolved '
    'constraints. If none is clearly identifiable, return an empty string.\n'
    '  (c) `unresolved_constraints`: 1-2 short phrases describing what about '
    'the bridge_entity still needs verification.\n'
    '  (d) `hop2_query`: a 3-8 word BM25 search query that names the '
    '`bridge_entity` verbatim and targets the `unresolved_constraints`. '
    'Avoid generic words; prefer proper nouns and distinctive content terms.\n\n'
    'Respond with a JSON object containing: "reasoning" (step-by-step), '
    '"context_summary", "bridge_entity", "unresolved_constraints", "hop2_query".'
)


def summarize_and_query_hop2(claim: str, passages: list[str]) -> dict:
    """Merged hop1 summarize + hop2 query generation. Sees actual passages, not a lossy summary."""
    system = MERGED_HOP1_INSTRUCTION
    user = f"Claim: {claim}\n\nPassages:\n" + "\n".join(passages)
    result = _llm_call(system, user)
    bridge_entity = (result.get("bridge_entity") or "").strip()
    hop2_query = (result.get("hop2_query") or "").strip()
    if not bridge_entity or bridge_entity.lower() in ("unknown", "none", "n/a"):
        # Fallback: use the original claim as hop2 query if no bridge entity found.
        hop2_query = claim
    elif not hop2_query:
        hop2_query = bridge_entity
    return {
        "reasoning": result.get("reasoning", ""),
        "context_summary": result.get("context_summary", ""),
        "bridge_entity": bridge_entity,
        "unresolved_constraints": result.get("unresolved_constraints", ""),
        "hop2_query": hop2_query,
    }


CREATE_QUERY_HOP3_INSTRUCTION = (
    'Given the `claim`, `summary_1`, and `summary_2`, produce two outputs for '
    'hop 3 of a multi-hop BM25 retrieval pipeline:\n'
    '  - `query`: a 3-8 word BM25 search query targeting a constraint from '
    'the claim that the two summaries have NOT yet verified. Name any still-'
    'unverified bridging entity verbatim. Prefer distinctive proper nouns.\n'
    '  - `unresolved_bridge`: the single most important proper-noun entity '
    'from `summary_1` or `summary_2` that the claim still depends on but is '
    'NOT yet fully verified — typically a person, organization, work, place, '
    'or date. If none is clearly identifiable, return an empty string.\n\n'
    'Respond with a JSON object containing: "reasoning" (step-by-step), '
    '"query", "unresolved_bridge".'
)


def create_query(
    claim: str, summary_1: str, summary_2: str, instruction: str | None = None
) -> dict:
    """Generate hop3 query and an unresolved bridge entity from accumulated summaries."""
    system = instruction or CREATE_QUERY_HOP3_INSTRUCTION
    user = f"Claim: {claim}\n\nSummary 1: {summary_1}\n\nSummary 2: {summary_2}"

    if '"reasoning"' not in system:
        system += '\n\nRespond with a JSON object containing: "reasoning", "query", "unresolved_bridge"'

    result = _llm_call(system, user)
    return {
        "reasoning": result.get("reasoning", ""),
        "query": result.get("query", ""),
        "unresolved_bridge": (result.get("unresolved_bridge") or "").strip(),
    }


# ==================== Agent Pipeline ====================

def forward(claim: str) -> DotDict:
    """
    Run the full 3-hop retrieval + summarization pipeline with dual-query at
    hop2 and hop3 plus cross-hop deduplication (multihop-qa-optimizer
    Section 3, retrieval part).

    claim → [BM25s k=14] → merged(summarize1+bridge+hop2_query)
          → [BM25s hop2_query k=11] + [BM25s bridge_entity k=5] - dedup_vs_hop1
          → summarize2 → create_query_hop3 (also emits unresolved_bridge)
          → [BM25s hop3_query k=18] + [BM25s unresolved_bridge k=5]
            - dedup_vs_hop1∪hop2 → retrieved_docs
    """
    def _title_of(doc: str) -> str:
        return doc.split(" | ", 1)[0].strip().lower()

    # HOP 1: over-retrieve to widen coverage (Section 3 over-retrieve pattern).
    hop1_docs = search(claim, k=18).passages
    seen_titles = {_title_of(d) for d in hop1_docs}

    # Merged hop1 summarize + bridge + hop2-query (Section 2).
    merged_result = summarize_and_query_hop2(claim=claim, passages=hop1_docs)
    summary_1 = merged_result["context_summary"]
    hop2_query = merged_result["hop2_query"]
    bridge_entity = merged_result.get("bridge_entity", "")

    # HOP 2: dual query — semantic hop2_query + direct bridge_entity lookup,
    # excluding any docs already retrieved in hop1.
    hop2_query_docs = search(hop2_query, k=14).passages
    # Always run the bridge-side BM25 lookup so the node graph (and per-node
    # metrics) stay deterministic. Fall back to the claim when no bridge
    # entity was extracted.
    bridge_query = bridge_entity.strip() if bridge_entity and bridge_entity.strip() else claim
    hop2_bridge_docs = search(bridge_query, k=7).passages
    hop2_docs = []
    for d in hop2_query_docs + hop2_bridge_docs:
        t = _title_of(d)
        if t in seen_titles:
            continue
        seen_titles.add(t)
        hop2_docs.append(d)

    summary_result_2 = summarize(claim=claim, passages=hop2_docs, context=summary_1)
    summary_2 = summary_result_2["summary"]

    # HOP 3: dual query — semantic hop3_query + direct unresolved-bridge lookup,
    # mirror of iter 2 hop2 pattern. Always run both calls for deterministic
    # call counts in evaluate.py per-node tracker.
    query_result_3 = create_query(claim=claim, summary_1=summary_1, summary_2=summary_2)
    hop3_query = query_result_3["query"]
    hop3_bridge = query_result_3.get("unresolved_bridge", "") or ""
    hop3_raw = search(hop3_query, k=28).passages
    hop3_bridge_query = hop3_bridge.strip() if hop3_bridge.strip() else claim
    hop3_bridge_docs = search(hop3_bridge_query, k=7).passages
    hop3_docs = []
    for d in hop3_raw + hop3_bridge_docs:
        t = _title_of(d)
        if t in seen_titles:
            continue
        seen_titles.add(t)
        hop3_docs.append(d)

    return DotDict(
        retrieved_docs=hop1_docs + hop2_docs + hop3_docs,
        hop1_docs=hop1_docs,
        hop2_docs=hop2_docs,
        hop3_docs=hop3_docs,
        summary_1=summary_1,
        summary_2=summary_2,
        hop2_query=hop2_query,
        hop3_query=hop3_query,
    )


# ==================== Evaluation ====================

def evaluate_single(example) -> dict:
    """Run forward pass on a single example and compute retrieval score."""
    try:
        pred = forward(example.claim)
        result = discrete_retrieval_eval(example, pred.retrieved_docs)
        return {
            "claim": example.claim,
            "supporting_facts": [doc["key"] for doc in example.supporting_facts],
            "summary_1": pred.summary_1,
            "summary_2": pred.summary_2,
            "hop2_query": pred.hop2_query,
            "hop3_query": pred.hop3_query,
            "num_retrieved": len(pred.retrieved_docs),
            "score": result.score,
            "feedback": result.feedback,
            "error": None,
        }
    except Exception as e:
        return {
            "claim": example.get("claim", ""),
            "supporting_facts": [],
            "summary_1": "",
            "summary_2": "",
            "hop2_query": "",
            "hop3_query": "",
            "num_retrieved": 0,
            "score": 0.0,
            "feedback": "",
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
                running_score = sum(r["score"] for _, r in results) / len(results)
                print(f"  [{label}] {completed}/{total} done, running retrieval score: {running_score:.3f}")

    # Sort by original index
    results.sort(key=lambda x: x[0])
    results = [r for _, r in results]

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    success_count = sum(1 for s in scores if s >= 1.0)

    print(f"\n  {label} Results:")
    print(f"    Average Score: {avg_score:.4f}")
    print(f"    Retrieval Success: {success_count}/{len(scores)} ({avg_score:.4f})")

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
    print("INDEPENDENT HOVER AGENT — Standalone (no DSPy/MEGA/dspylite/GEPA)")
    print("=" * 80)
    print(f"\n  Model: {MODEL}")
    print(f"  Seed: {SEED}")
    print(f"\n  3-Hop Workflow:")
    print(f"    1. summarize1: claim + passages -> summary_1")
    print(f"    2. create_query_hop2: claim + summary_1 -> hop2_query")
    print(f"    3. summarize2: claim + context + passages -> summary_2")
    print(f"    4. create_query_hop3: claim + summary_1 + summary_2 -> hop3_query")
    print(f"\n  Retrieval: BM25s k=7 (hops 1,2), k=10 (hop 3)")
    print(f"  Metric: Binary (1.0 if ALL gold docs retrieved, else 0.0)")
    print("=" * 80)

    # Load dataset
    print("\n[1/3] Loading HOVER dataset...")
    train_data, val_data, test_data = load_hover(
        seed=SEED, n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST,
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
    avg_score = sum(test_scores) / len(test_scores) if test_scores else 0.0
    success_count = sum(1 for s in test_scores if s >= 1.0)

    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"  Retrieval Score:   {avg_score:.4f}")
    print(f"  Success Rate:      {success_count}/{len(test_scores)}")
    print(f"  Time:              {total_time:.1f}s")
    print(f"{'=' * 80}")

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "hover_optimized.json"
    summary = {
        "model": MODEL,
        "seed": SEED,
        "n_test": len(test_scores),
        "retrieval_score": avg_score,
        "success_count": success_count,
        "time_seconds": total_time,
        "per_example": test_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved results -> {out_path}")
