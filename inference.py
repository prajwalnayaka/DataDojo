import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import DataDojoEnv
from models import ActionModel, ActionType

from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://prajwalnayaka-datadojo.hf.space"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
DIFFICULTIES = ["Easy", "Medium", "Hard"]
BENCHMARK = "DataDojo"
MAX_STEPS = 10
TEMPERATURE = 0.2  # Low temp for precise, structured JSON actions
MAX_TOKENS = 512
SUCCESS_SCORE_THRESHOLD = 0.7  # Reward is tanh-normalized in [-1, 1]; 0.7 is a strong clean

# ==========================================
# 2. PROMPTS
# ==========================================

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a data cleaning agent. You are given a dirty pandas DataFrame and must clean it
    using the available tools. Your goal is to make the dataset match a clean reference version
    by removing duplicates, filling or dropping missing values, fixing corrupted numeric columns,
    and standardizing categorical text.

    AVAILABLE ACTIONS:
    - DROP_DUPLICATES: Remove duplicate rows. No column_name needed.
    - DROP_COLUMN: Must provide a value for 'column_name only from the data schema'. Look at 'NaN counts'—if a column has 100% NaNs, drop it by name. NEVER send null.
    - FILL_NA: Fill NaN values in a column. fill_value can be "mean", "median", "mode", or a literal value.
    - STRIP_CHAR: Remove characters from a column using a regex pattern. Use to fix currency strings like "$1,250.00".
    - TYPE_CAST: Cast a column to a target type. target_type can be "float", "int", "str".
    - LOWERCASE: Lowercase and strip all text values in a column.
    - GET_VALUE_COUNTS: Inspect unique values and their counts in a column. Does not modify the dataset.
    - MAP_VALUES: Replace specific values in a column using a dictionary mapping.

    STRATEGY BY DIFFICULTY:
    - Easy: Look for duplicate rows, a fully empty column (drop it), and a column with ~30% NaNs (fill it).
    - Medium: Same as Easy, plus a numeric column corrupted with currency formatting (STRIP_CHAR then TYPE_CAST).
    - Hard: Same as Medium, plus categorical columns with inconsistent casing (LOWERCASE or MAP_VALUES).
    
    CRITICAL CONSTRAINTS:
     - You MUST provide a valid string for 'column_name' for every action EXCEPT DROP_DUPLICATES.
     - NEVER set 'column_name' to null for DROP_COLUMN, FILL_NA, or STRIP_CHAR. 
     - If you fail to provide a valid column name from the schema, you'll be penalized.

    RESPONSE FORMAT:
    You must respond with a single valid JSON object and nothing else. No markdown, no explanation.
    The JSON must match this schema exactly:
    {
    "action": "ACTION_TYPE",
    "column_name": "actual_column_name_to_be_altered", 
    "fill_value": "value",
    "target_type": "type",
    "regex_pattern": "pattern",
    "mapping_dict": {"old": "new"}
    }
    
    IMPORTANT: DROP_DUPLICATES should only be used ONCE per episode. 
    Using it more than once wastes a step and incurs a penalty. If you have already used DROP_DUPLICATES, do NOT use it again.

    Examples:
    {"action": "STRIP_CHAR", "column_name": "Charges", "fill_value": null, "target_type": null, "regex_pattern": "[\\£,]", "mapping_dict": null}
    {"action": "DROP_DUPLICATES", "column_name": null, "fill_value": null, "target_type": null, "regex_pattern": null, "mapping_dict": null}
    """
).strip()


def build_user_prompt(
    step: int,
    difficulty: str,
    data_schema: dict,
    nans: dict,
    sample: list,
    last_info: str,
    last_reward: float,
    last_breakdown: list,
    eda_result: Optional[dict],
    history: List[str],
) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    eda_block = json.dumps(eda_result, indent=2) if eda_result else "None"
    if last_breakdown:
        breakdown_lines = "\n".join(
            f"  • {list(d.keys())[0]}: {list(d.values())[0]:+.3f}"
            for d in last_breakdown
        )
        breakdown_block = f"\n{breakdown_lines}"
    else:
        breakdown_block = " None"

    error_block = ""
    if "Error" in last_info:
        error_block = (f"\n⚠️ LAST ACTION FAILED: {last_info}\nDo NOT repeat this action. column_name is REQUIRED for DROP_COLUMN, FILL_NA, STRIP_CHAR, TYPE_CAST, LOWERCASE, GET_VALUE_COUNTS, MAP_VALUES. Use exact column names from the schema.\n "
                       f"Valid columns RIGHT NOW are: {list(data_schema.keys())}. You MUST pick from this exact list.")

    return textwrap.dedent(
        f"""
        
         AVAILABLE COLUMNS (use ONLY these exact names):
        {json.dumps(list(data_schema.keys()), indent=2)}
    
        Step: {step} / {MAX_STEPS}
        Difficulty: {difficulty}

        --- DATASET STATE ---
        Schema (column: dtype):
        {json.dumps(data_schema, indent=2)}

        NaN counts per column:
        {json.dumps(nans, indent=2)}

        Sample rows (first 10):
        {json.dumps(sample, indent=2)}

        --- LAST ACTION RESULT ---
        Info: {last_info}
        Reward: {last_reward:.3f}
        Breakdown of Reward: {breakdown_block}

        --- EDA RESULT (GET_VALUE_COUNTS output) ---
        {eda_block}

        --- ACTION HISTORY ---
        {history_block}
        
        --- ERROR LOGS ---
        {error_block}

        Based on the dataset state above, decide your next cleaning action.
        Respond with a single JSON object only.
        """
    ).strip()


# ==========================================
# 3. LOGGING
# ==========================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ==========================================
# 4. LLM CALL
# ==========================================

def get_model_action(
    client: OpenAI,
    step: int,
    difficulty: str,
    data_schema: dict,
    nans: dict,
    sample: list,
    last_info: str,
    last_reward: float,
    last_breakdown: list,
    eda_result: Optional[dict],
    history: List[str],
) -> ActionModel:
    """Calls the LLM and parses its JSON response into an ActionModel."""
    user_prompt = build_user_prompt(
        step, difficulty, data_schema, nans, sample,
        last_info, last_reward, last_breakdown, eda_result, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if the model wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        for key in ["column_name", "fill_value", "target_type", "regex_pattern"]:
            if parsed.get(key) in ["None", "null", "column_name", "<column_name or null>"]:
                parsed[key] = None
        return ActionModel(**parsed)

    except Exception as exc:
        print(f"[DEBUG] Model action parse failed: {exc}", flush=True)
        # Safe fallback — always a valid no-op that won't crash the env
        return ActionModel(action=ActionType.DROP_DUPLICATES)


# ==========================================
# 5. EPISODE RUNNER & MAIN LOOP
# ==========================================

async def run_episode(env: DataDojoEnv, difficulty: str, client: OpenAI) -> None:
    """Runs one full episode at a given difficulty level."""
    task_name = f"data-cleaning-{difficulty.lower()}"
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = await env.reset(difficulty=difficulty)
        obs = reset_result.observation
        done = reset_result.done

        last_info = obs.info
        last_reward = 0.0
        last_breakdown = obs.metadata.get("breakdown", [])
        eda_result = obs.EDA

        for step in range(1, MAX_STEPS + 1):


            action = get_model_action(
                client=client,
                step=step,
                difficulty=difficulty,
                data_schema=obs.data_schema,
                nans=obs.NaNs,
                sample=obs.sample,
                last_info=last_info,
                last_reward=last_reward,
                last_breakdown=last_breakdown,
                eda_result=eda_result,
                history=history,
            )

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None if "Error" not in obs.info else obs.info

            rewards.append(reward)
            steps_taken = step
            last_info = obs.info
            last_reward = reward
            last_breakdown = obs.metadata.get("breakdown", [])
            eda_result = obs.EDA

            action_str = action.model_dump_json(exclude_none=False)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action.model_dump_json(exclude_none=False)} -> info='{obs.info}' reward={reward:+.3f}")

            if done: # If the tasks in done after this current step, break out of the for i in range(1, MAX_STEPS+1)
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0  # average, not sum
        score = max(0.001, min(score, 0.999))  # clamp strictly within (0,1)
        success = done and score > SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error ({difficulty}): {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    print(f"[DEBUG] ENV_BASE_URL: {os.getenv('ENV_BASE_URL')}")
    print(f"[DEBUG] LOCAL_IMAGE_NAME: {os.getenv('LOCAL_IMAGE_NAME')}")
    print(f"[DEBUG] HF_TOKEN length: {len(os.getenv('HF_TOKEN', ''))}")
    env_url = os.getenv("ENV_BASE_URL")
    image_name = os.getenv("LOCAL_IMAGE_NAME")
    hf_token = os.getenv("HF_TOKEN")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    if not hf_token:
        print("[ERROR] HF_TOKEN is missing!")
        return
    try:
        if env_url:
            print(f"[INFO] Connecting to provided environment at {env_url}", flush=True)
            env = DataDojoEnv(base_url=env_url)
            await env.__aenter__()
        elif image_name:
            # Local Docker fallback
            print(f"[INFO] Starting Docker image: {image_name}", flush=True)
            env = await DataDojoEnv.from_docker_image(image_name, port=8000)
            await env.__aenter__()
        else:
            print(f"[INFO] Starting Docker image: {image_name}")
            env = DataDojoEnv(base_url="http://localhost:8000")
            await env.__aenter__()
    except Exception as e:
        print(f"[FATAL] Connection to environment failed: {e}",flush=True)
        return

    try:
        for difficulty in DIFFICULTIES:
            await run_episode(env, difficulty, client)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())