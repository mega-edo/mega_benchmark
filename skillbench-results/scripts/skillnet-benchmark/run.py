#!/usr/bin/env python3
"""
SkillNet benchmark pipeline — faithful reproduction of SkillNet's experiment code.

For each task:
  1. retrieve_relevant_skills() — send ALL 4,110 skills' metadata to LLM, select up to 5
  2. generate_overall_procedure() — compile full skill contents, LLM generates procedure
  3. Assemble task directory with skills + procedure in instruction.md

Both prompts are imported directly from SkillNet-main/experiments/src/prompt_generator.py.
Skill content compilation reproduces SkillModule.generate_overall_procedure() from skill.py.

Upstream dependency (used unmodified): https://github.com/zjunlp/SkillNet
Before running, clone the upstream repo at the skillbench-records root:
    git clone https://github.com/zjunlp/SkillNet.git SkillNet-main

Requires:
  export GEMINI_API_KEY=<your-gemini-api-key>

Usage:
  python scripts/skillnet-benchmark/run.py
  python scripts/skillnet-benchmark/run.py --tasks flood-risk-analysis 3d-scan-calc
  python scripts/skillnet-benchmark/run.py --model gemini/gemini-3-flash-preview
"""

import argparse
import json
import os
import sys
import time
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

try:
    import litellm
except ImportError:
    print("ERROR: litellm is required. Install with: pip install litellm")
    sys.exit(1)

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

SKILLSBENCH_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = SKILLSBENCH_ROOT / "4200-skills" / "SKILLS"
TASKS_DIR = SKILLSBENCH_ROOT / "tasks"
OUTPUT_TASKS_DIR = SKILLSBENCH_ROOT / "tasks-skillnet-benchmark"
OUTPUT_DIR = Path(__file__).parent / "outputs"
RESULTS_FILE = OUTPUT_DIR / "results.json"
METADATA_FILE = OUTPUT_DIR / "all_skills_metadata.json"
LATENCY_FILE = OUTPUT_DIR / "latency_metrics.json"

MAX_WORKERS = int(os.environ.get("SKILLNET_WORKERS", "8"))

# Import SkillNet's exact prompts
SKILLNET_SRC = SKILLSBENCH_ROOT / "SkillNet-main" / "experiments" / "src"
sys.path.insert(0, str(SKILLNET_SRC))
from prompt_generator import retrieve_relevant_skills_prompt, generate_overall_procedure_prompt


def llm_call(messages: list[dict], model: str) -> str:
    """Call LLM with retry logic."""
    for attempt in range(5):
        try:
            response = litellm.completion(model=model, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"    LLM error (attempt {attempt+1}): {e}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def get_task_instruction(task_dir: Path) -> str:
    """Read the base instruction (without workflow section) from a task."""
    instruction_path = task_dir / "instruction.md"
    if not instruction_path.exists():
        return ""
    text = instruction_path.read_text(encoding="utf-8")
    if "\n---\n" in text:
        text = text.split("\n---\n")[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Metadata loading — reproduces SkillModule._load_metadata() (skill.py:39-69)
# ---------------------------------------------------------------------------

def build_metadata() -> dict:
    """Load all skill metadata (name -> {description, skill_dir})."""
    if METADATA_FILE.exists():
        print(f"Loading cached metadata from {METADATA_FILE}")
        return json.loads(METADATA_FILE.read_text())

    print("Building metadata from 4200-skills/SKILLS/...")
    metadata = {}
    skipped = 0

    for skill_dir in sorted(SKILLS_DIR.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            skipped += 1
            continue
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception:
            skipped += 1
            continue
        if not content.strip().startswith("---"):
            skipped += 1
            continue
        parts = content.split("---", 2)
        if len(parts) < 3:
            skipped += 1
            continue
        try:
            header = yaml.safe_load(parts[1])
        except Exception:
            skipped += 1
            continue
        if not isinstance(header, dict):
            skipped += 1
            continue
        name = header.get("name", "")
        description = header.get("description", "")
        if not isinstance(name, str) or not isinstance(description, str):
            skipped += 1
            continue
        if not name or not description:
            skipped += 1
            continue
        metadata[name] = {
            "description": description.strip(),
            "skill_dir": str(skill_dir),
        }

    print(f"  Loaded {len(metadata)} skills (skipped {skipped})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


# ---------------------------------------------------------------------------
# Stage 1: Skill retrieval — reproduces skill.py:71-83
# ---------------------------------------------------------------------------

def retrieve_relevant_skills(task_instruction: str, metadata: dict, model: str) -> list[str]:
    """Send ALL metadata to LLM, return up to 5 skill names.

    Parsing matches original SkillNet skill.py:80-82 (strict, no fallback).
    """
    messages = retrieve_relevant_skills_prompt(metadata, task_instruction)
    response = llm_call(messages, model)

    raw = response.split("<Relevant_Skill_Names>")[1].split("</Relevant_Skill_Names>")[0]
    raw = raw.strip("`json\n").strip("`\n").strip("```\n")
    relevant_skill_names = json.loads(raw)
    return relevant_skill_names


# ---------------------------------------------------------------------------
# Stage 2: Procedure generation — reproduces skill.py:85-137
# ---------------------------------------------------------------------------

def compile_skill_contents(skill_names: list[str], metadata: dict) -> list[tuple]:
    """Compile full skill contents (SKILL.md + auxiliary files).

    Reproduces skill.py lines 98-126.
    """
    skill_contents = []
    for skill_name in skill_names:
        info = metadata.get(skill_name)
        if not info:
            continue
        skill_dir = Path(info["skill_dir"])
        if not skill_dir.is_dir():
            continue

        combined_text = f"=== Skill: {skill_name} ===\n"

        main_file = skill_dir / "SKILL.md"
        if main_file.exists():
            combined_text += f"\n[File: SKILL.md]\n"
            combined_text += main_file.read_text(encoding="utf-8") + "\n"

        for file_path in skill_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "SKILL.md":
                try:
                    relative_path = file_path.relative_to(skill_dir)
                    content = file_path.read_text(encoding="utf-8")
                    combined_text += f"\n[File: {relative_path}]\n"
                    combined_text += content + "\n"
                except (UnicodeDecodeError, Exception):
                    continue

        skill_contents.append((skill_name, combined_text))
    return skill_contents


def generate_overall_procedure(task_instruction: str, skill_names: list[str], metadata: dict, model: str) -> str:
    """Generate procedure using SkillNet's exact prompt."""
    skill_contents = compile_skill_contents(skill_names, metadata)
    if not skill_contents:
        return ""

    messages = generate_overall_procedure_prompt(task_instruction, "", skill_contents)
    response = llm_call(messages, model)

    overall_procedure = response.split("<Overall_Procedure>")[1].split("</Overall_Procedure>")[0].strip()
    return overall_procedure


# ---------------------------------------------------------------------------
# Task assembly
# ---------------------------------------------------------------------------

def assemble_task(task_name: str, skill_names: list[str], procedure: str, metadata: dict) -> bool:
    """Copy base task, replace skills, update instruction.md."""
    base_task = TASKS_DIR / task_name
    output_task = OUTPUT_TASKS_DIR / task_name

    if not base_task.exists():
        print(f"  WARNING: Base task not found: {base_task}")
        return False

    if output_task.exists():
        shutil.rmtree(output_task)
    shutil.copytree(base_task, output_task)

    # Replace environment/skills/
    skills_out = output_task / "environment" / "skills"
    if skills_out.exists():
        shutil.rmtree(skills_out)
    skills_out.mkdir(parents=True, exist_ok=True)

    for skill_name in skill_names:
        info = metadata.get(skill_name)
        if not info:
            continue
        src = Path(info["skill_dir"])
        # Use original directory name (kebab-case) instead of display name
        # to avoid spaces/caps/parentheses in filesystem paths
        dst = skills_out / src.name
        if src.exists():
            shutil.copytree(src, dst)

    # Update instruction.md — same format as generate_procedures.py:97
    if procedure:
        # Normalize agent-specific skill paths to environment/skills/
        import re
        procedure = re.sub(r'/root/\.[a-z]+/skills/', 'environment/skills/', procedure)
        base_instruction = get_task_instruction(base_task)
        new_content = base_instruction + "\n\n---\n\n" + procedure + "\n"
        (output_task / "instruction.md").write_text(new_content, encoding="utf-8")

    return True


# ---------------------------------------------------------------------------
# Per-task worker function
# ---------------------------------------------------------------------------

def process_task(task_name: str, metadata: dict, model: str) -> tuple[str, dict, dict]:
    """Process a single task end-to-end. Returns (task_name, result, task_latency)."""
    task_dir = TASKS_DIR / task_name
    instruction = get_task_instruction(task_dir)
    if not instruction:
        return task_name, {"skills": [], "procedure_len": 0, "error": "no instruction"}, {}

    task_latency = {}
    try:
        # Stage 1: Retrieve skills
        t0 = time.time()
        skills = retrieve_relevant_skills(instruction, metadata, model)
        task_latency["discovery_seconds"] = round(time.time() - t0, 3)

        # Stage 2: Generate procedure
        procedure = ""
        if skills:
            t0 = time.time()
            procedure = generate_overall_procedure(instruction, skills, metadata, model)
            task_latency["procedure_generation_seconds"] = round(time.time() - t0, 3)

        # Stage 3: Assemble task directory
        assemble_task(task_name, skills, procedure, metadata)

        result = {"skills": skills, "procedure_len": len(procedure)}
        return task_name, result, task_latency

    except Exception as e:
        result = {"skills": [], "procedure_len": 0, "error": str(e)}
        assemble_task(task_name, [], "", metadata)
        return task_name, result, task_latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SkillNet benchmark — faithful reproduction")
    parser.add_argument("--model", default="gemini/gemini-3-flash-preview")
    parser.add_argument("--tasks", nargs="*", help="Specific task names (default: all)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: Set GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Load ALL metadata
    metadata = build_metadata()
    print(f"Total skills: {len(metadata)}")

    # Load existing results for resume
    results = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        print(f"Loaded {len(results)} existing results (resume mode)")

    # Load existing latency metrics for resume
    latency = {}
    if LATENCY_FILE.exists():
        latency = json.loads(LATENCY_FILE.read_text())
        print(f"Loaded {len(latency)} existing latency entries")

    OUTPUT_TASKS_DIR.mkdir(parents=True, exist_ok=True)

    # Enumerate tasks
    if args.tasks:
        task_names = args.tasks
    else:
        task_names = sorted([
            d.name for d in TASKS_DIR.iterdir()
            if d.is_dir() and (d / "instruction.md").exists()
        ])

    # Filter to pending tasks
    pending = [
        t for t in task_names
        if t not in results or not (OUTPUT_TASKS_DIR / t).exists()
    ]

    print(f"\nTotal: {len(task_names)} tasks, Pending: {len(pending)} (skipping {len(task_names) - len(pending)} already done)")
    print(f"Sending ALL {len(metadata)} skills to LLM per task (SkillNet's original method)")

    if not pending:
        print("All tasks already completed!")
    else:
        workers = min(MAX_WORKERS, len(pending))
        print(f"Running with {workers} parallel workers (set SKILLNET_WORKERS to change)\n")

        save_lock = threading.Lock()
        assemble_lock = threading.Lock()
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_task, task_name, metadata, args.model): task_name
                for task_name in pending
            }

            for future in as_completed(futures):
                task_name, result, task_latency = future.result()
                completed += 1

                skills = result.get("skills", [])
                error = result.get("error")
                if error:
                    print(f"  [{completed}/{len(pending)}] {task_name} — ERROR: {error}")
                else:
                    disc_t = task_latency.get("discovery_seconds", 0)
                    proc_t = task_latency.get("procedure_generation_seconds", 0)
                    print(f"  [{completed}/{len(pending)}] {task_name} — {len(skills)} skills in {disc_t:.1f}s, procedure in {proc_t:.1f}s: {skills}")

                with save_lock:
                    results[task_name] = result
                    latency[task_name] = task_latency
                    RESULTS_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                    LATENCY_FILE.write_text(json.dumps(latency, indent=2, ensure_ascii=False))

    # Copy remaining tasks not yet assembled
    all_base = {d.name for d in TASKS_DIR.iterdir() if d.is_dir()}
    assembled = {d.name for d in OUTPUT_TASKS_DIR.iterdir() if d.is_dir()}
    missing = all_base - assembled
    if missing:
        print(f"\nCopying {len(missing)} remaining tasks...")
        for name in sorted(missing):
            src = TASKS_DIR / name
            dst = OUTPUT_TASKS_DIR / name
            if not dst.exists():
                shutil.copytree(src, dst)
                sd = dst / "environment" / "skills"
                if sd.exists():
                    shutil.rmtree(sd)
                    sd.mkdir(parents=True, exist_ok=True)

    # Summary
    total = len(list(OUTPUT_TASKS_DIR.iterdir()))
    total_skills = sum(len(r.get("skills", [])) for r in results.values())
    with_skills = sum(1 for r in results.values() if r.get("skills"))
    print(f"\n{'='*60}")
    print(f"Done! {total} tasks in tasks-skillnet-benchmark/")
    print(f"Tasks with skills: {with_skills}, total skills: {total_skills}")
    print(f"Results: {RESULTS_FILE}")

    # Latency summary
    disc_times = [v["discovery_seconds"] for v in latency.values() if "discovery_seconds" in v]
    proc_times = [v["procedure_generation_seconds"] for v in latency.values() if "procedure_generation_seconds" in v]
    if disc_times:
        avg = sum(disc_times) / len(disc_times)
        print(f"Discovery latency — avg: {avg:.2f}s, min: {min(disc_times):.2f}s, max: {max(disc_times):.2f}s, n={len(disc_times)}")
    if proc_times:
        avg = sum(proc_times) / len(proc_times)
        print(f"Procedure generation latency — avg: {avg:.2f}s, min: {min(proc_times):.2f}s, max: {max(proc_times):.2f}s, n={len(proc_times)}")
    combined = [
        v.get("discovery_seconds", 0) + v.get("procedure_generation_seconds", 0)
        for v in latency.values()
        if "discovery_seconds" in v
    ]
    if combined:
        avg = sum(combined) / len(combined)
        print(f"Total query-time latency — avg: {avg:.2f}s, min: {min(combined):.2f}s, max: {max(combined):.2f}s, n={len(combined)}")
    print(f"Latency metrics: {LATENCY_FILE}")


if __name__ == "__main__":
    main()
