import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from typing import Tuple
from models import ActionModel, ObservationModel, RewardModel, ActionType
from genesis_engine_mk3 import generate_mk3_dataframe
from ruiner_engine_mk3 import run_ruiner


class DataCleaningEnv:
    """
    The main Arena. This class manages the state of the dataset and
    evaluates the LLM agent's cleaning attempts.
    """

    def __init__(self, difficulty: str = "Easy", max_turns: int = 10):
        self.difficulty = difficulty
        self.max_turns = max_turns
        self.skeletons_dir = Path(r"D:\DataDojo\Skeletons")
        self.episode_id=str(uuid.uuid4())
        self.turn_count = 0
        self.master_df = None
        self.current_df = None
        self.last_eda_result = None

    def _get_observation(self) -> ObservationModel:
        return ObservationModel(
            schema=self.current_df.dtypes.apply(lambda x: x.name).to_dict(),
            NaNs=self.current_df.isnull().sum().to_dict(),
            sample=self.current_df.head(3).replace({np.nan: None}).to_dict(orient="records"),
            EDA=self.last_eda_result
        )

    def state(self) -> dict:
        """Returns the current state of the environment."""
        return {
            "episode_id": self.episode_id,
            "step_count": self.turn_count,
            "difficulty": self.difficulty,
            "max_turns": self.max_turns
        }

    def reset(self) -> ObservationModel:
        """Starts a new episode with a fresh, ruined dataset."""
        self.master_df = generate_mk3_dataframe(self.skeletons_dir)
        self.current_df = run_ruiner(self.master_df.copy(), self.difficulty)
        self.turn_count = 0
        self.last_eda_result = None

        return self._get_observation()

    def step(self, action_input: ActionModel) -> Tuple[ObservationModel, RewardModel]:
        """Executes one cleaning action and returns the result."""
        self.turn_count += 1
        self.last_eda_result = None  # Clear old tool outputs
        info_msg = ""

        # --- THE SWITCHBOARD (Execution) ---
        try:
            act = action_input.action
            col = action_input.column_name

            if act == ActionType.DROP_COLUMN:
                self.current_df.drop(columns=[col], inplace=True)
                info_msg = f"Successfully dropped column: {col}"

            elif act == ActionType.DROP_DUPLICATES:
                self.current_df.drop_duplicates(inplace=True)
                info_msg = "Successfully removed duplicate rows."

            elif act == ActionType.FILL_NA:
                val = action_input.fill_value
                self.current_df[col] = self.current_df[col].fillna(val)
                info_msg = f"Filled NaNs in {col} with {val}"

            elif act == ActionType.STRIP_CHAR:
                pattern = action_input.regex_pattern
                self.current_df[col] = self.current_df[col].astype(str).str.replace(pattern, "", regex=True)
                info_msg = f"Stripped characters from {col} using pattern: {pattern}"

            elif act == ActionType.TYPE_CAST:
                target = action_input.target_type
                self.current_df[col] = self.current_df[col].astype(target)
                info_msg = f"Cast {col} to {target}"

            elif act == ActionType.GET_VALUE_COUNTS:
                # This doesn't change the DF, just provides info
                self.last_eda_result = self.current_df[col].value_counts().to_dict()
                info_msg = f"Retrieved value counts for {col}"

            elif act == ActionType.MAP_VALUES:
                mapping = action_input.mapping_dict
                self.current_df[col] = self.current_df[col].replace(mapping)
                info_msg = f"Mapped values in {col} using provided dictionary."

        except Exception as e:
            info_msg = f"Error executing {action_input.action}: {str(e)}"

        # --- THE GRADER (Reward & Victory) ---
        # Binary check for now: Is it perfect?
        # (You can make this more complex later by checking null counts)
        is_perfect = self.current_df.equals(self.master_df)

        reward = 1.0 if is_perfect else 0.0
        done = is_perfect or self.turn_count >= self.max_turns

        return self._get_observation(), RewardModel(score=reward, done=done, info={"message": info_msg})