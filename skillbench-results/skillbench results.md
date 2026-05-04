# SkillsBench Curation Records

Experimental records comparing **AgentSkillOS**, **SkillNet**, and **Wisdom Curation (MEGA / Wisdomgraph)** on skill discovery and curation quality, built on top of [benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench).

> Note: the Wisdom Curation condition was run under internal codenames (`router-medium-best-v2`, `harbor-medium-best-v2`) during experimentation. All published artifacts in this repo use the unified name **`wisdomgraph`** ("Wisdom Curation").

## 1. Purpose

This repository holds the raw trial outputs, configs, and retrieval/curation scripts used to measure **end-to-end pipeline performance** (retrieval + curation injected into the agent) across four conditions on the 84-task SkillsBench suite.

All conditions share:

- **VM:** GCP `n2d-standard-128`
- **Agent:** Gemini CLI + Gemini 3 Flash
- **LLM for every curation step:** Gemini 3 Flash (no condition benefits from a stronger model at any stage)
- **Attempts per task:** 5
- **Concurrency:** 43 concurrent Docker trials
- **Skill pool:** 4,207 assets (includes SkillsBench's golden skills for all curation systems)
- **Tasks, verifiers, Docker env:** identical

The only variable is the **skill discovery and orchestration method**:

- **No Skills** — agent receives only the task instruction.
- **SkillNet** — LLM metadata matching + LLM procedural guidance (original `retrieve_relevant_skills_prompt`, `generate_overall_procedure_prompt`).
- **AgentSkillOS** — original `TreeBuilder` + `Searcher` hierarchical tree search, then Quality-First DAG via the original `build_planner_prompt`; DAG is topologically linearized for sequential execution.
- **MEGA (WG)** — PCST-based compositional retrieval from WG-DB + role-differentiated plan assembly.

## 2. Results

Wisdom curation quality on SkillsBench (84 tasks, Gemini 3 Flash). Pass rate is the primary metric; efficiency = pass rate per megatoken consumed (score/Mtok).

| Condition       | Pass Rate (%) | Avg Tokens/Task (k) | Curation Latency (sec/task) | Efficiency (score/Mtok) |
|-----------------|:-------------:|:-------------------:|:---------------------------:|:-----------------------:|
| No Skills       | 31.5          | 894                 | —                           | 0.353                   |
| AgentSkillOS    | 41.1          | 1189                | 403.4                       | 0.345                   |
| SkillNet        | 41.7          | 983                 | 37.8                        | 0.424                   |
| **MEGA (WG)**   | **46.5**      | **822**             | **11.8**                    | **0.566**               |

## 3. Verifying the results

Per-condition trial bundles are published as GitHub Release assets (each bundle
is 100–150 MB, exceeding GitHub's per-file repo limit). Download them from the
[`v0.1.0-jobs` release](https://github.com/mega-edo/mega_benchmark/releases/tag/v0.1.0-jobs):

- [no-skills.zip](https://github.com/mega-edo/mega_benchmark/releases/download/v0.1.0-jobs/no-skills.zip)
- [skillnet.zip](https://github.com/mega-edo/mega_benchmark/releases/download/v0.1.0-jobs/skillnet.zip)
- [agentskillos.zip](https://github.com/mega-edo/mega_benchmark/releases/download/v0.1.0-jobs/agentskillos.zip)
- [wisdomgraph.zip](https://github.com/mega-edo/mega_benchmark/releases/download/v0.1.0-jobs/wisdomgraph.zip)

Or fetch all four with the GitHub CLI:

```bash
mkdir -p experiments/jobs
gh release download v0.1.0-jobs \
  --repo mega-edo/mega_benchmark \
  --dir experiments/jobs
```

To inspect results:

1. Unzip the condition you want to audit, e.g. `unzip experiments/jobs/wisdomgraph.zip -d experiments/jobs/`.
2. The top-level `<condition>/result.json` contains the aggregate pass rate, error counts, and per-task reward stats for the whole job.
3. Each trial has its own directory `<condition>/<task-name>__<trial-id>/` with its own `result.json` holding the per-trial config, agent info, verifier output, and reward.

The aggregate mean under `stats.evals.<eval-key>.metrics[0].mean` in the top-level `result.json` is the pass rate reported in the table above.

Run configs used to launch each job are in [experiments/configs/](experiments/configs/) (`no-skills.yaml`, `skillnet.yaml`, `agentskillos.yaml`, `wisdomgraph.yaml`).

The `datasets[].path` fields in those configs reference `${TASKS_DIR}/tasks-<condition>`. Set `TASKS_DIR` to the directory holding the unzipped task bundles (e.g. `export TASKS_DIR=$PWD`) before launching a job.

## 4. Curation pipeline scripts

The two baseline pipelines below import code directly from upstream repos. Those repos are **not vendored** in this record set — clone them at the project root before running:

```bash
git clone https://github.com/ynulihao/AgentSkillOS.git AgentSkillOS-main
git clone https://github.com/zjunlp/SkillNet.git    SkillNet-main
```

Each script's docstring repeats this instruction. Both upstreams are used **unmodified** (no patches applied).

### AgentSkillOS — [scripts/agentskillos/](scripts/agentskillos/)

Wraps the upstream [AgentSkillOS](https://github.com/ynulihao/AgentSkillOS) modules (`TreeBuilder`, `Searcher`, planner prompt) unmodified, driven by four stages:

1. [build_tree.py](scripts/agentskillos/build_tree.py) — builds the hierarchical skill tree over the 4,207-asset pool.
2. [discover_skills.py](scripts/agentskillos/discover_skills.py) — runs `Searcher.search()` per task (ThreadPoolExecutor, 4 threads internally) and emits `outputs/skill_discovery_results.json`.
3. [generate_dag_plans.py](scripts/agentskillos/generate_dag_plans.py) — applies the original Quality-First `build_planner_prompt` to the retrieved skills, producing `outputs/dag_plans.json`.
4. [assemble_tasks.py](scripts/agentskillos/assemble_tasks.py) — topologically linearizes each DAG and injects the plan into the SkillsBench task context for Gemini CLI execution.

Latency per task is logged to `scripts/agentskillos/outputs/latency_metrics.json`.

### SkillNet — [scripts/skillnet-benchmark/](scripts/skillnet-benchmark/)

Wraps the upstream [SkillNet](https://github.com/zjunlp/SkillNet) prompts (`retrieve_relevant_skills_prompt`, `generate_overall_procedure_prompt`) unmodified through a single driver:

- [run.py](scripts/skillnet-benchmark/run.py) — for each task: (a) matches skills from `outputs/all_skills_metadata.json` against the task via the retrieval prompt, (b) generates the overall procedure, (c) writes the merged context used at execution time.

Outputs:

- `outputs/all_skills_metadata.json` — metadata view of the 4,207-asset pool.
- `outputs/domain_index.json` — domain-level index used during matching.
- `outputs/results.json` — per-task retrieval + procedure records.
- `outputs/latency_metrics.json` — per-task wall-clock curation latency.
