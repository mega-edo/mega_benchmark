#!/usr/bin/env python3
"""Step 2: Discover relevant skills for each task using AgentSkillOS tree search.

Searcher.search() internally uses ThreadPoolExecutor (4 threads) for parallel
LLM calls at each tree level, so external parallelism is unnecessary.

Upstream dependency (used unmodified): https://github.com/ynulihao/AgentSkillOS
Before running, clone the upstream repo at the skillbench-records root:
    git clone https://github.com/ynulihao/AgentSkillOS.git AgentSkillOS-main
"""

import sys
import os
import json
import time
from pathlib import Path

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

SKILLSBENCH_ROOT = Path(__file__).resolve().parents[2]
AGENTSKILLOS_ROOT = SKILLSBENCH_ROOT / "AgentSkillOS-main"

# Load .env BEFORE importing AgentSkillOS (config.py reads env at import time)
from dotenv import load_dotenv
load_dotenv(SKILLSBENCH_ROOT / ".env")

if os.environ.get("GEMINI_API_KEY") and not os.environ.get("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.environ["GEMINI_API_KEY"]

sys.path.insert(0, str(AGENTSKILLOS_ROOT / "src"))
from manager.tree.searcher import Searcher

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "skill_discovery_results.json"
LATENCY_FILE = OUTPUT_DIR / "latency_metrics.json"

MAX_SKILLS = 8  # Match AgentSkillOS default (config.yaml max_skills)
INTERNAL_PARALLEL = 4  # Searcher internal thread count


def get_task_instruction(task_dir: Path) -> str:
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return ""
    text = instruction_path.read_text(encoding="utf-8")
    if "\n---\n" in text:
        text = text.split("\n---\n")[0]
    return text.strip()


def main():
    tree_path = AGENTSKILLOS_ROOT / "data" / "capability_trees" / "tree_4200.yaml"
    tasks_dir = SKILLSBENCH_ROOT / "tasks-no-skills"

    if not tree_path.exists():
        print(f"ERROR: Tree not found: {tree_path}")
        sys.exit(1)

    if not tasks_dir.exists():
        print(f"ERROR: Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    model = os.environ.get("LLM_MODEL", "gemini/gemini-3-flash-preview")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("ERROR: Set GEMINI_API_KEY or LLM_API_KEY in .env or environment")
        sys.exit(1)

    print(f"API key: {api_key[:8]}...{api_key[-4:]}")
    print(f"Model: {model}")
    print(f"Internal parallelism: {INTERNAL_PARALLEL} threads per search\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results for resume
    results = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())
        print(f"Loaded {len(results)} existing results (resume mode)")

    latency = {}
    if LATENCY_FILE.exists():
        latency = json.loads(LATENCY_FILE.read_text())

    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir() and (d / "instruction.md").exists()])
    pending = [d for d in task_dirs if d.name not in results]
    print(f"Tasks: {len(task_dirs)} total, {len(pending)} pending\n")

    if not pending:
        print("All tasks already completed!")
    else:
        searcher = Searcher(
            tree_path=str(tree_path),
            model=model,
            api_key=api_key,
            max_parallel=INTERNAL_PARALLEL,
        )

        for i, task_dir in enumerate(pending):
            task_name = task_dir.name
            instruction = get_task_instruction(task_dir)
            if not instruction:
                continue

            t0 = time.time()
            try:
                result = searcher.search(instruction, verbose=False)
                elapsed = time.time() - t0
                skills = result.selected_skills[:MAX_SKILLS]
                print(f"  [{i+1}/{len(pending)}] {task_name} — {len(skills)} skills in {elapsed:.1f}s: {[s['name'] for s in skills]}")
                results[task_name] = {
                    "skills": skills,
                    "llm_calls": result.llm_calls,
                    "explored_nodes": result.explored_nodes,
                }
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(pending)}] {task_name} — ERROR ({elapsed:.1f}s): {e}")
                results[task_name] = {"skills": [], "error": str(e)}

            latency.setdefault(task_name, {})["discovery_seconds"] = round(elapsed, 3)

            # Checkpoint after each task
            OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            LATENCY_FILE.write_text(json.dumps(latency, indent=2, ensure_ascii=False))

    # Summary
    print(f"\nDone! Results saved to {OUTPUT_FILE}")
    total_skills = sum(len(r.get("skills", [])) for r in results.values())
    print(f"Total: {len(results)} tasks, {total_skills} skills discovered")

    disc_times = [v["discovery_seconds"] for v in latency.values() if "discovery_seconds" in v]
    if disc_times:
        avg = sum(disc_times) / len(disc_times)
        print(f"Latency — avg: {avg:.1f}s, min: {min(disc_times):.1f}s, max: {max(disc_times):.1f}s")


if __name__ == "__main__":
    main()
