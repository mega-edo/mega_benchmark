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


# ==================== The 4 Prompts ====================

EXTRACT_AND_QUERY_INSTRUCTION = (
    'You are a multi-hop question answering assistant. You extract the bridge '
    'entity and known facts from the retrieved passages AND generate a '
    'targeted search query for the next retrieval hop, all in one step.\n\n'
    'Given a question and passages, do TWO things:\n'
    '1. EXTRACT the bridge entity, unresolved constraints, and a concise '
    'context summary from the passages.\n'
    '2. GENERATE a precise search query (3-8 words) to find the ONE missing '
    'fact needed to answer the question.\n\n'
    'Bridge extraction rules:\n'
    '- bridge_entity MUST be a specific proper noun found in the passages, '
    'NOT a vague description. If the passages contain the entity\'s full '
    'formal name, use that full name.\n'
    '- For comparison or yes/no questions, identify BOTH entities and note '
    'what is known about each in context_summary.\n'
    '- context_summary should preserve ALL relevant facts from the passages, '
    'including partial ones — downstream stages depend on it.\n\n'
    'Query generation rules:\n'
    '- Use the resolved bridge_entity name you just extracted, NOT the '
    'original description from the question.\n'
    '- Target the SPECIFIC attribute still needed (birthplace, founding year, '
    'nationality, occupation, etc.).\n'
    '- For comparison questions, search for the entity whose attribute is '
    'still unknown.\n'
    '- 3-8 words, specific enough to find the exact fact in Wikipedia.\n'
    '- If bridge_entity cannot be confidently extracted, set bridge_entity '
    'to "" and use the original question as hop2_query.\n\n'
    'Respond in JSON with exactly: "reasoning" (brief chain of thought), '
    '"bridge_entity", "unresolved_constraints", "context_summary", '
    '"hop2_query".'
)


def extract_and_query(
    question: str, passages: list[str], instruction: str | None = None
) -> dict:
    """Stage 1+2 (merged): Bridge extraction + hop2 query generation in one LLM call.

    Produces summary_1 (= context_summary) and hop2_query directly from raw passages.
    """
    system = instruction or EXTRACT_AND_QUERY_INSTRUCTION
    if '"reasoning"' not in system:
        system += (
            '\n\nRespond with a JSON object containing: "reasoning", '
            '"bridge_entity", "unresolved_constraints", "context_summary", '
            '"hop2_query"'
        )
    passages_str = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))
    user = f"<question>{question}</question>\n\n<passages>\n{passages_str}\n</passages>"
    result = _llm_call(system, user)
    bridge = (result.get("bridge_entity") or "").strip()
    summary = (result.get("context_summary") or "").strip()
    constraints = (result.get("unresolved_constraints") or "").strip()
    hop2_query = (result.get("hop2_query") or "").strip()
    if not hop2_query or not bridge:
        # Fallback per Section 2 step 4
        hop2_query = hop2_query or question
    # summary_1 carries the full context, including bridge + constraints, for downstream stages
    summary_1 = summary
    if bridge:
        summary_1 = f"Bridge entity: {bridge}. {summary_1}".strip()
    if constraints:
        summary_1 = f"{summary_1} Unresolved: {constraints}".strip()
    return {
        "reasoning": result.get("reasoning", ""),
        "summary": summary_1,
        "query": hop2_query,
        "bridge_entity": bridge,
    }


SUMMARIZE2_INSTRUCTION = (
    'You are summarizing the second-hop evidence for a multi-hop question. '
    'The downstream extractor needs to find the SHORT answer span verbatim '
    'in your summary.\n\n'
    'Rules:\n'
    '- Extract every entity, date, number, place, title, nationality, '
    'profession, or comparison fact that could plausibly answer the '
    'question. Use the EXACT surface form from the passages (do not '
    'paraphrase a date as "the late 1880s" if the passage says "1886"; '
    'do not paraphrase a title — keep "That\'s Dancing!" verbatim with '
    'apostrophe and exclamation).\n'
    '- For comparison/choice questions, summarize the relevant attribute '
    'for BOTH candidates side-by-side (e.g. "Marco Martins won Best '
    'Picture at festival X in 2007; Douchan Gersi did not appear in any '
    'Best Picture record"). Never collapse to a yes/no judgment.\n'
    '- Preserve qualifying compounds (e.g. "Irish-born", "58th-ranked", '
    '"head of the Government of Lithuania") verbatim — these compounds '
    'are often the gold answer.\n'
    '- Carry forward the bridge_entity from the previous hop\'s context '
    'verbatim so the answer extractor can resolve cross-hop references.\n'
    '- If passages are off-topic, summarize what you do see and note '
    'which question constraint remains unresolved — never abstain.\n\n'
    'Respond with a JSON object containing: "reasoning" (brief chain of '
    'thought) and "summary" (a fact-dense paragraph with verbatim spans).'
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
    'You are answering a HotpotQA-style multi-hop question using two evidence '
    'summaries. The output is scored by EXACT-MATCH after lowercasing, '
    'article removal (a/an/the), and punctuation stripping. Surface-form '
    'discipline matters more than fluency.\n\n'
    'Output rules for the `answer` field — NO EXCEPTIONS:\n'
    '1. YES/NO QUESTIONS: answer must be exactly "yes" or "no". No explanation, '
    'no qualifiers, no "Yes, because...", no "No, X is not Y". Just the word.\n'
    '2. Otherwise return the SHORTEST span that fully answers the question — '
    'typically a proper noun, a number, a date, or a noun phrase. Never a '
    'full sentence. Do not prefix with "The answer is", "It is", or restate '
    'the question.\n'
    '3. NUMBERS: use the form the question implies. If the question asks '
    '"How many X" prefer the digit form ("3" not "three") UNLESS the gold '
    'evidence uses a word — match the evidence form. Do not append units '
    'unless the question explicitly asks for them.\n'
    '4. DATES: use the most specific form supported by the evidence (year '
    'alone is fine if only the year is asked).\n'
    '5. NAMES / TITLES: use the canonical name as it appears in the '
    'evidence (e.g. "The Internet" not "Internet" if the title contains '
    '"The"; conversely drop honorifics like "Mr.", "Dr." unless the question '
    'requires them).\n'
    '6. COMPARISON / CHOICE questions: "Which is older/taller/first", '
    '"Did X or Y win", "Was X always the Y" — answer with the chosen '
    'ENTITY NAME, NEVER yes/no. The yes/no rule (Rule 1) only applies to '
    'true polar questions like "Is X a Y?" or "Does X have Y?" without an '
    '"or" between candidate entities.\n'
    '7. NEVER abstain. Forbidden outputs: "", "none", "unknown", "cannot '
    'determine", "n/a", "not specified". If the summaries lack the answer, '
    'pick the SINGLE most likely entity/fact/value from the evidence even '
    'if uncertain. An empty answer is always graded wrong; a wrong guess '
    'might still match.\n'
    '8. Strip adjectives, titles, and qualifiers that the question does NOT '
    'request: "Bishop Wulfstan" → "Wulfstan" if the question just asks '
    '"Which bishop"; "2017 Cannes Film Festival" → "Cannes Film Festival" '
    'if the question asks "which festival"; "indo islamic architecture '
    'around 7th century" → "Islamic architecture" if asked "what '
    'architecture style". Match the granularity of the question.\n'
    '9. For "What nationality" questions, prefer the precise form the '
    'evidence uses (e.g. "Irish-born" if that exact compound appears) over '
    'the simpler base ("Irish"). Read the gold-evidence phrasing first.\n\n'
    'Respond with a JSON object containing exactly two fields: '
    '"reasoning" (a brief chain of thought, 1-3 sentences, used for '
    'auditing only) and "answer" (the bare answer span obeying the rules '
    'above).'
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
    # Stage 1+2 (merged): Retrieve hop1 docs, extract bridge entity, generate hop2 query
    hop1_docs = search(question, k=k).passages
    extract_result = extract_and_query(question=question, passages=hop1_docs)
    summary_1 = extract_result["summary"]
    hop2_query = extract_result["query"]

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
        summarize1_reasoning=extract_result["reasoning"],
        query_reasoning="",
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

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "hotpotqa_optimized.json"
    summary = {
        "model": MODEL,
        "seed": SEED,
        "n_test": len(test_scores),
        "test_em": test_em,
        "time_seconds": total_time,
        "per_example": test_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved results -> {out_path}")
