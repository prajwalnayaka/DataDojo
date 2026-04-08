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

## The Architecture:

The system operates on a dual-component architecture, or simply "The Twins," ensuring complete separation between data hosting and agent logic:

- This is the source of truth. It generates or serves the "master dataset", which is the perfectly cleaned, formatted, and validated reference data that represents the ideal state.

- The Ruiner systematically injects "dirt" into the master dataset by introducing missing values, duplicate rows, regex-defying string corruptions, and inconsistent categorical casing.

**The Agent's success is measured by its ability to reverse the Ruiner's chaos and restore the dataset to the Genesis' standard.**
___

## Task Levels

DataDojo evaluates agents across three increasing levels of corruption intensity:

**1. Easy: Duplicate removal, dropping an useless column and handling missing values (NaNs).**

**2. Medium: All the challenges from Easy level plus cleaning up numerical (dtype = object) columns by removing introduced chars and type casting (dtype=object to int/float).**

**3. Hard: All the challenges from easy and medium plus correcting the disorganized casings of strings in a column.**
___

## Environment Logic

 **Actions & Observations:**
 -  Observations: he agent observes a "corrupted" state of the dataset injected with missing values, duplicates, and regex-defying string corruptions.


 - Actions: Tool calls for DataFrame manipulation. The LLM agent must provide the tool call name and the column of its choice.

___

## Grading & Subtle Logic


The final score is a tanh-normalized average of rewards and penalties, which is again mapped to [0,1] range.

### Subtle logics:

- **The "One Free Drop"**: To encourage surgical precision in the agent, the very first column drop tool is free but the subsequent second call has an action penalty, this combined with the check and harsh penalty for dropping the wrong column helps the agent be very cautious about the drop column tool call. This prevent **reward hacking**, where the agent could learn to drop the columns to reduce error count and get rewards for it.

- Action Dependencies: Agents must learn the order of operations—for example, a TYPE_CAST will fail if a STRIP_CHAR hasn't first removed non-numeric symbols like '$' or '.' .

___