#!/usr/bin/env python3
"""Step 3: Generate DAG plans for each task using AgentSkillOS planner prompt.

Upstream dependency (used unmodified): https://github.com/ynulihao/AgentSkillOS
Before running, clone the upstream repo at the skillbench-records root:
    git clone https://github.com/ynulihao/AgentSkillOS.git AgentSkillOS-main
"""

import sys
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SKILLSBENCH_ROOT = Path(__file__).resolve().parents[2]
AGENTSKILLOS_ROOT = SKILLSBENCH_ROOT / "AgentSkillOS-main"
sys.path.insert(0, str(AGENTSKILLOS_ROOT / "src"))

import litellm
from orchestrator.dag.prompts import build_planner_prompt

SKILLS_DIR = SKILLSBENCH_ROOT / "4200-skills" / "SKILLS"
DISCOVERY_FILE = Path(__file__).parent / "outputs" / "skill_discovery_results.json"
OUTPUT_FILE = Path(__file__).parent / "outputs" / "dag_plans.json"
LATENCY_FILE = Path(__file__).parent / "outputs" / "latency_metrics.json"
TASKS_DIR = SKILLSBENCH_ROOT / "tasks-no-skills"

MAX_WORKERS = int(os.environ.get("DAG_WORKERS", "8"))


def read_skill_content(skill_name: str) -> str:
    """Read SKILL.md content for a given skill."""
    skill_path = SKILLS_DIR / skill_name / "SKILL.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")
    return f"Skill '{skill_name}' documentation not found."


def get_task_instruction(task_name: str) -> str:
    """Read base instruction for a task."""
    instruction_path = TASKS_DIR / task_name / "instruction.md"
    if not instruction_path.exists():
        return ""
    text = instruction_path.read_text(encoding="utf-8")
    if "\n---\n" in text:
        text = text.split("\n---\n")[0]
    return text.strip()


def build_skills_info(skills: list[dict]) -> str:
    """Build skills info string for the planner prompt."""
    lines = []
    for skill in skills:
        name = skill.get("name", skill.get("id", "unknown"))
        desc = skill.get("description", "No description")
        content = read_skill_content(name)
        # Truncate very long skill content — 5000 chars matches AgentSkillOS default
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"
        lines.append(f"### {name}\n{desc}\n\n{content}\n")
    return "\n".join(lines)


def generate_plan(task_name: str, instruction: str, skills: list[dict], model: str) -> dict:
    """Generate DAG plans for a single task."""
    skills_info = build_skills_info(skills)
    prompt = build_planner_prompt(task=instruction, skills_info=skills_info)

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    try:
        plans = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            plans = json.loads(match.group(1))
        else:
            plans = {"plans": [], "error": "Failed to parse LLM response", "raw": content[:500]}

    return plans


def process_task(task_name: str, data: dict, model: str) -> tuple[str, dict, float]:
    """Process a single task for DAG generation. Returns (task_name, result, elapsed)."""
    instruction = get_task_instruction(task_name)
    if not instruction:
        return task_name, None, 0

    skills = data["skills"]
    t0 = time.time()
    try:
        plans = generate_plan(task_name, instruction, skills, model)
        elapsed = time.time() - t0
        result = {
            "plans": plans.get("plans", []),
            "skill_names": [s.get("name", s.get("id")) for s in skills],
        }
        return task_name, result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        result = {
            "plans": [],
            "skill_names": [s.get("name", s.get("id")) for s in skills],
            "error": str(e),
        }
        return task_name, result, elapsed


def main():
    from dotenv import load_dotenv
    load_dotenv(SKILLSBENCH_ROOT / ".env")

    model = os.environ.get("LLM_MODEL", "gemini/gemini-3-flash-preview")
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if api_key:
        # Set for litellm
        if "gemini" in model.lower():
            os.environ["GEMINI_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key

    if not DISCOVERY_FILE.exists():
        print(f"ERROR: Skill discovery results not found: {DISCOVERY_FILE}")
        print("Run discover_skills.py first (Step 2)")
        sys.exit(1)

    discovery = json.loads(DISCOVERY_FILE.read_text())
    print(f"Loaded skill discovery for {len(discovery)} tasks")

    # Load existing results for resume
    results = {}
    if OUTPUT_FILE.exists():
        results = json.loads(OUTPUT_FILE.read_text())
        print(f"Loaded {len(results)} existing DAG plans from {OUTPUT_FILE}")

    # Load existing latency metrics (shared with discover_skills.py)
    latency = {}
    if LATENCY_FILE.exists():
        latency = json.loads(LATENCY_FILE.read_text())
        print(f"Loaded {len(latency)} existing latency entries")

    tasks_with_skills = {k: v for k, v in discovery.items() if v.get("skills")}
    print(f"Tasks with discovered skills: {len(tasks_with_skills)}")

    # Filter to pending tasks
    pending = {k: v for k, v in sorted(tasks_with_skills.items()) if k not in results or "plans" not in results[k]}
    print(f"Pending: {len(pending)} tasks (skipping {len(tasks_with_skills) - len(pending)} already done)")

    if not pending:
        print("All tasks already completed!")
    else:
        workers = min(MAX_WORKERS, len(pending))
        print(f"Running with {workers} parallel workers (set DAG_WORKERS to change)")

        save_lock = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_task, task_name, data, model): task_name
                for task_name, data in pending.items()
            }

            for future in as_completed(futures):
                task_name, result, elapsed = future.result()
                completed += 1

                with save_lock:
                    if result is None:
                        print(f"  [{completed}/{len(pending)}] {task_name} — skipped (no instruction)")
                    elif "error" in result:
                        print(f"  [{completed}/{len(pending)}] {task_name} — ERROR ({elapsed:.1f}s): {result['error']}")
                        results[task_name] = result
                    else:
                        plan_count = len(result.get("plans", []))
                        print(f"  [{completed}/{len(pending)}] {task_name} — {plan_count} plans in {elapsed:.1f}s")
                        results[task_name] = result

                    if task_name not in latency:
                        latency[task_name] = {}
                    latency[task_name]["dag_generation_seconds"] = round(elapsed, 3)

                    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                    LATENCY_FILE.write_text(json.dumps(latency, indent=2, ensure_ascii=False))

    print(f"\nDone! DAG plans saved to {OUTPUT_FILE}")

    # Print latency summary for DAG generation
    dag_times = [v["dag_generation_seconds"] for v in latency.values() if "dag_generation_seconds" in v]
    if dag_times:
        avg = sum(dag_times) / len(dag_times)
        print(f"DAG generation latency — avg: {avg:.2f}s, min: {min(dag_times):.2f}s, max: {max(dag_times):.2f}s, n={len(dag_times)}")

    # Print combined summary
    combined = [
        v.get("discovery_seconds", 0) + v.get("dag_generation_seconds", 0)
        for v in latency.values()
        if "discovery_seconds" in v and "dag_generation_seconds" in v
    ]
    if combined:
        avg = sum(combined) / len(combined)
        print(f"Total query-time latency (discovery+DAG) — avg: {avg:.2f}s, min: {min(combined):.2f}s, max: {max(combined):.2f}s, n={len(combined)}")


if __name__ == "__main__":
    main()
