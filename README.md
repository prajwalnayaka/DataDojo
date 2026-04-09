---
title: DataDojo
emoji: 🥷
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DataDojo: The Autonomous Data Cleaning Benchmark

DataDojo is a containerized reinforcement learning environment designed to evaluate the reasoning and data-wrangling capabilities of AI agents. It provides a standardized "gym" where LLMs interact with corrupted datasets to reach a clean "reference" state through autonomous decision-making.
___

## The Architecture

The system operates on a dual-component architecture, or simply "The Twins," ensuring complete separation between data generation and corruption logic:

- **Genesis Engine** is the source of truth. It procedurally generates a randomized "master dataset" from a library of domain skeletons. The datasets have varying number of columns, row count, and data types on every episode — producing a perfectly clean reference state that the agent must restore.

- **Ruiner Engine** systematically injects "dirt" into the master dataset by introducing missing values, duplicate rows, regex-defying string corruptions, and inconsistent categorical casing.

**The agent's success is measured by its ability to reverse the Ruiner's chaos and restore the dataset to the Genesis standard.**
___

## Task Levels

DataDojo evaluates agents across three increasing levels of corruption intensity:

**1. Easy:** Duplicate removal, dropping a fully empty column, and filling missing values (NaNs).

**2. Medium:** All Easy challenges, plus cleaning numerical columns corrupted with currency-style string formatting (dtype: object → float/int via STRIP_CHAR then TYPE_CAST).

**3. Hard:** All Medium challenges, plus detecting and standardizing inconsistent string casing across categorical columns.
___

## Environment Logic

**Actions & Observations:**

- **Observations:** Each step, the agent receives the current dataset state — column schema, NaN counts per column, a sample of rows, the result of any EDA tool call, and a detailed breakdown of the reward from the previous action. This gives the agent everything it needs to reason about its next move.

- **Actions:** Tool calls for DataFrame manipulation. The agent must specify the tool name and, for most actions, the exact column to operate on. Available tools span the full cleaning pipeline: `DROP_DUPLICATES`, `DROP_COLUMN`, `FILL_NA`, `STRIP_CHAR`, `TYPE_CAST`, `LOWERCASE`, `GET_VALUE_COUNTS`, and `MAP_VALUES`.

___

## Grading & Subtle Logic

The reward per step is a `tanh`-normalized sum of component rewards and penalties, producing a score in `[-1, 1]` which is then remapped to the range `[0, 1]`.

### Reward shaping:

- **Error reduction reward:** The primary signal. Reward is proportional to how many errors (NaNs, duplicates, and value mismatches against the master) the action eliminates relative to the episode's starting error count.

- **The "One Free Drop":** The first `DROP_COLUMN` call is unpunished, but every subsequent drop carries an action penalty. Dropping a non-empty column carries an additional harsh penalty. This prevents reward hacking — without it, an agent could learn to drop columns indiscriminately to reduce error count and collect rewards without actually cleaning anything.

- **DROP_DUPLICATES spam penalty:** Using `DROP_DUPLICATES` more than once per episode incurs a penalty, discouraging it as a zero-cost fallback action.

- **Invalid column penalty:** Referencing a column name that doesn't exist in the current dataset is penalized, pushing the agent to ground its actions in the observed schema rather than hallucinate column names.

- **Datatype mismatch penalty:** If the agent operates on a column but leaves it in the wrong dtype relative to the master, a penalty is applied nudging the agent toward the `STRIP_CHAR` → `TYPE_CAST` sequence rather than stopping halfway.

- **Action dependencies:** Agents must learn the correct order of operations. A `TYPE_CAST` to float will fail if `STRIP_CHAR` hasn't first removed non-numeric symbols like `$` or `.`. The environment does not hand-hold failed actions are reported back and the agent must recover.

___

## A Note on Difficulty

DataDojo is intentionally hard. Each episode generates a **new, randomized dataset** from a randomized schema — column names, row counts, and data types vary every time, so the agent cannot memorize a fixed solution. It must genuinely read and reason about the data it is given.

During development, we ran `Qwen2.5-72B-Instruct`, a state-of-the-art 72 billion parameter open-source model as the baseline agent. Even at the Easy level, the model reliably identifies and drops the empty column but frequently struggles to chain together the remaining steps (fill NaNs, remove duplicates) within the 10-step budget. At Medium and Hard levels, the multi-step reasoning is required to identify the corrupted column, strip the offending characters, then type-cast to the correct dtype, all while tracking what has already been done; this consistently challenges the model.

This is a feature, not a limitation. An environment that a frontier-scale model solves trivially on the first try offers no training signal. DataDojo is designed to sit at the frontier of what current LLMs can do with tool-use and multi-step data reasoning, leaving meaningful headroom for agents that are fine-tuned or trained via RL to demonstrate measurable improvement over the zero-shot baseline.
___