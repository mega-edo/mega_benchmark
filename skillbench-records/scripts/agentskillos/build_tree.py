#!/usr/bin/env python3
"""Step 1: Build capability tree from 4200 skills using AgentSkillOS TreeBuilder.

Upstream dependency (used unmodified): https://github.com/ynulihao/AgentSkillOS
Before running, clone the upstream repo at the skillbench-records root:
    git clone https://github.com/ynulihao/AgentSkillOS.git AgentSkillOS-main
"""

import sys
import os
from pathlib import Path

# AgentSkillOS-main/ must be a clone of https://github.com/ynulihao/AgentSkillOS
SKILLSBENCH_ROOT = Path(__file__).resolve().parents[2]
AGENTSKILLOS_ROOT = SKILLSBENCH_ROOT / "AgentSkillOS-main"
sys.path.insert(0, str(AGENTSKILLOS_ROOT / "src"))

from manager.tree.builder import TreeBuilder
from manager.tree.models import DynamicTreeConfig


def main():
    skills_dir = SKILLSBENCH_ROOT / "4200-skills" / "SKILLS"
    output_path = AGENTSKILLOS_ROOT / "data" / "capability_trees" / "tree_4200.yaml"

    if not skills_dir.exists():
        print(f"ERROR: Skills directory not found: {skills_dir}")
        sys.exit(1)

    skill_count = sum(1 for d in skills_dir.iterdir() if d.is_dir())
    print(f"Found {skill_count} skill directories in {skills_dir}")

    # branching_factor=7 matches AgentSkillOS default (config.yaml)
    config = DynamicTreeConfig(branching_factor=7, max_depth=6)

    builder = TreeBuilder(
        skills_dir=str(skills_dir),
        output_path=str(output_path),
        config=config,
        model=os.environ.get("LLM_MODEL", "gemini/gemini-3-flash-preview"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY"),
    )

    print(f"Building tree with branching_factor={config.branching_factor}...")
    print(f"Output: {output_path}")
    print(f"Model: {os.environ.get('LLM_MODEL', 'gemini/gemini-3-flash-preview')}")

    tree_dict = builder.build(verbose=True, show_tree=True, generate_html=True)
    print(f"\nTree built successfully! Output saved to {output_path}")


if __name__ == "__main__":
    main()
