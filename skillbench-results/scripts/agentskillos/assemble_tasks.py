#!/usr/bin/env python3
"""Step 4: Assemble tasks-agentskillos/ directory from DAG plans and discovered skills.

Upstream dependency (used unmodified): https://github.com/ynulihao/AgentSkillOS
Before running, clone the upstream repo at the skillbench-records root:
    git clone https://github.com/ynulihao/AgentSkillOS.git AgentSkillOS-main
(skill_seeds/ are read from the upstream tree.)
"""

import json
import shutil
from pathlib import Path

SKILLSBENCH_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = SKILLSBENCH_ROOT / "4200-skills" / "SKILLS"
SEED_SKILLS_DIR = SKILLSBENCH_ROOT / "AgentSkillOS-main" / "data" / "skill_seeds"
BASE_TASKS_DIR = SKILLSBENCH_ROOT / "tasks"  # Use tasks/ (has Dockerfile COPY skills)
OUTPUT_TASKS_DIR = SKILLSBENCH_ROOT / "tasks-agentskillos"
DAG_PLANS_FILE = Path(__file__).parent / "outputs" / "dag_plans.json"
DISCOVERY_FILE = Path(__file__).parent / "outputs" / "skill_discovery_results.json"


def topological_sort(nodes: list[dict]) -> list[dict]:
    """Sort DAG nodes in dependency order."""
    node_map = {n["id"]: n for n in nodes}
    graph = {n["id"]: set(n.get("depends_on", [])) for n in nodes}
    visited = set()
    ordered = []

    while len(ordered) < len(nodes):
        ready = [nid for nid, deps in graph.items()
                 if nid not in visited and deps.issubset(visited)]
        if not ready:
            # Remaining nodes have circular deps; add them anyway
            ready = [nid for nid in graph if nid not in visited]
        for nid in sorted(ready):
            ordered.append(node_map[nid])
            visited.add(nid)

    return ordered



def group_by_depth(nodes: list[dict]) -> list[list[dict]]:
    """Group DAG nodes into depth layers for parallel execution."""
    node_map = {n["id"]: n for n in nodes}
    graph = {n["id"]: set(n.get("depends_on", [])) for n in nodes}

    depth = {}
    visited = set()
    layers: list[list[dict]] = []

    while len(visited) < len(nodes):
        ready = [nid for nid, deps in graph.items()
                 if nid not in visited and deps.issubset(visited)]
        if not ready:
            ready = [nid for nid in graph if nid not in visited]
        layer = [node_map[nid] for nid in sorted(ready)]
        layers.append(layer)
        visited.update(ready)

    return layers


def generate_workflow_section(plan: dict) -> str:
    """Generate workflow section from a DAG plan, preserving parallel structure."""
    nodes = plan.get("nodes", [])
    if not nodes:
        return ""

    layers = group_by_depth(nodes)
    lines = [
        "",
        "---",
        "",
        "# Workflow",
        "",
        "Follow steps in order. Steps within the same group can run in parallel.",
        "",
    ]

    step_num = 0
    for layer in layers:
        if len(layer) > 1:
            step_num += 1
            lines.append(f"## step-{step_num} (parallel)")
            lines.append("")
            for node in layer:
                skill_name = node.get("name", "unknown")
                purpose = node.get("purpose", "Execute task step")
                outputs = node.get("outputs_summary", "")
                lines.append(f"- **{purpose}**")
                lines.append(f"  - Skill: `{skill_name}` — See `environment/skills/{skill_name}/SKILL.md`")
                if outputs:
                    lines.append(f"  - Expected outputs: {outputs}")
                lines.append("")
        else:
            node = layer[0]
            step_num += 1
            skill_name = node.get("name", "unknown")
            purpose = node.get("purpose", "Execute task step")
            outputs = node.get("outputs_summary", "")
            downstream = node.get("downstream_hint", "")

            lines.append(f"## step-{step_num}: {purpose}")
            lines.append("")
            lines.append(f"- **Skill:** `{skill_name}`")
            lines.append(f"  - See `environment/skills/{skill_name}/SKILL.md`")
            lines.append("")

            if outputs:
                lines.append(f"**Expected outputs:** {outputs}")
                lines.append("")

            if downstream:
                lines.append(f"**Downstream:** {downstream}")
                lines.append("")

    return "\n".join(lines)


def build_skill_lookup(discovery_data: dict) -> dict:
    """Build a lookup from skill display name to (actual_dir_name, source_path).

    Discovery results contain a `skill_path` pointing to the original SKILL.md.
    The parent directory of that path is the real directory name on disk, which
    may differ from the display `name` (e.g. "FFmpeg Media Info" vs "ffmpeg-media-info").
    We search both 4200-skills/SKILLS/ and AgentSkillOS seed_skills/.
    """
    lookup: dict[str, tuple[str, Path]] = {}
    for task_data in discovery_data.values():
        for skill in task_data.get("skills", []):
            name = skill.get("name", "")
            if name in lookup:
                continue

            # Extract the actual directory name from skill_path
            skill_path = skill.get("skill_path", "")
            if skill_path:
                actual_dir = Path(skill_path).parent.name
            else:
                actual_dir = name

            # Try to locate the skill in known directories
            src = SKILLS_DIR / actual_dir
            if src.exists():
                lookup[name] = (actual_dir, src)
                continue

            src = SEED_SKILLS_DIR / actual_dir
            if src.exists():
                lookup[name] = (actual_dir, src)
                continue

            # Fallback: try the display name directly (may work for kebab-case names)
            src = SKILLS_DIR / name
            if src.exists():
                lookup[name] = (name, src)
                continue

            src = SEED_SKILLS_DIR / name
            if src.exists():
                lookup[name] = (name, src)
                continue

            lookup[name] = (actual_dir, None)  # Not found anywhere

    return lookup


def assemble_task(task_name: str, plan_data: dict, skill_lookup: dict):
    """Assemble a single task directory."""
    base_task = BASE_TASKS_DIR / task_name
    output_task = OUTPUT_TASKS_DIR / task_name

    if not base_task.exists():
        print(f"  WARNING: Base task not found: {base_task}")
        return False

    # Copy base structure
    if output_task.exists():
        shutil.rmtree(output_task)

    # Copy everything from base task
    shutil.copytree(base_task, output_task)

    # Clear existing skills and replace with AgentSkillOS-discovered ones
    skills_out = output_task / "environment" / "skills"
    if skills_out.exists():
        shutil.rmtree(skills_out)
    skills_out.mkdir(parents=True, exist_ok=True)

    # Copy only skills used in the selected DAG plan (Quality-First = first plan)
    plans = plan_data.get("plans", [])
    dag_skill_names = set()
    if plans:
        for node in plans[0].get("nodes", []):
            dag_skill_names.add(node.get("name", ""))
    dag_skill_names.discard("")

    copied_skills = []
    for skill_name in dag_skill_names:
        actual_dir, src = skill_lookup.get(skill_name, (skill_name, None))
        # Use skill_name as destination dir so workflow references stay consistent
        dst = skills_out / skill_name
        if src and src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied_skills.append(skill_name)
        else:
            print(f"  WARNING: Skill not found anywhere: {skill_name} (tried dir: {actual_dir})")

    # Generate enhanced instruction.md
    plans = plan_data.get("plans", [])
    if plans:
        # Use Quality-First plan (first plan)
        selected_plan = plans[0]

        base_instruction = (base_task / "instruction.md").read_text(encoding="utf-8").strip()
        workflow = generate_workflow_section(selected_plan)
        enhanced_instruction = base_instruction + "\n" + workflow

        (output_task / "instruction.md").write_text(enhanced_instruction, encoding="utf-8")
    else:
        print(f"  WARNING: No plans available for {task_name}, using base instruction")

    return True


def main():
    if not DAG_PLANS_FILE.exists():
        print(f"ERROR: DAG plans not found: {DAG_PLANS_FILE}")
        print("Run generate_dag_plans.py first (Step 3)")
        return

    dag_plans = json.loads(DAG_PLANS_FILE.read_text())
    print(f"Loaded DAG plans for {len(dag_plans)} tasks")

    # Load discovery results to build skill name -> actual path lookup
    if DISCOVERY_FILE.exists():
        discovery_data = json.loads(DISCOVERY_FILE.read_text())
        print(f"Loaded discovery results for {len(discovery_data)} tasks")
    else:
        print(f"WARNING: Discovery file not found: {DISCOVERY_FILE}")
        discovery_data = {}

    skill_lookup = build_skill_lookup(discovery_data)
    not_found = [name for name, (_, src) in skill_lookup.items() if src is None]
    if not_found:
        print(f"WARNING: {len(not_found)} skills not found anywhere: {not_found}")

    # Create output directory
    OUTPUT_TASKS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0

    for i, (task_name, plan_data) in enumerate(sorted(dag_plans.items())):
        plans = plan_data.get("plans", [])
        skill_count = len(plan_data.get("skill_names", []))

        if not plans:
            print(f"[{i+1}/{len(dag_plans)}] {task_name} — skipped (no plans)")
            skip_count += 1
            continue

        node_count = len(plans[0].get("nodes", []))
        print(f"[{i+1}/{len(dag_plans)}] {task_name} — {skill_count} skills, {node_count} DAG nodes")

        if assemble_task(task_name, plan_data, skill_lookup):
            success_count += 1

    print(f"\nDone! Assembled {success_count} tasks in {OUTPUT_TASKS_DIR}")
    print(f"Skipped: {skip_count}")

    # Also assemble tasks that had no skills discovered (use base instruction only)
    # so we have a complete task set
    all_base_tasks = {d.name for d in BASE_TASKS_DIR.iterdir() if d.is_dir()}
    assembled_tasks = {d.name for d in OUTPUT_TASKS_DIR.iterdir() if d.is_dir()}
    missing = all_base_tasks - assembled_tasks

    if missing:
        print(f"\nCopying {len(missing)} remaining tasks without DAG plans...")
        for task_name in sorted(missing):
            src = BASE_TASKS_DIR / task_name
            dst = OUTPUT_TASKS_DIR / task_name
            if not dst.exists():
                shutil.copytree(src, dst)
        print(f"Total tasks in output: {len(list(OUTPUT_TASKS_DIR.iterdir()))}")


if __name__ == "__main__":
    main()
