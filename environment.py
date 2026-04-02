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
        self.initial_error_count=None
        self.prev_error_count=None
        self.master_df = None
        self.current_df = None
        self.last_eda_result = None

    def _get_observation(self) -> ObservationModel:
        return ObservationModel(
            schema=self.current_df.dtypes.apply(lambda x: x.name).to_dict(),
            NaNs=self.current_df.isnull().sum().to_dict(),
            sample=self.current_df.head(10).replace({np.nan: None}).to_dict(orient="records"),
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
        self.initial_error_count = self._calculate_total_errors(self.current_df.copy())
        self.prev_error_count = self._calculate_total_errors(self.current_df.copy())
        self.last_eda_result = None

        return self._get_observation()

    def _calculate_total_errors(self, current_df: pd.DataFrame) -> int:
        nans=current_df.isnull().sum().sum()
        dupes=current_df.duplicated().sum()
        common_cols = current_df.columns.intersection(self.master_df.columns)
        mismatches= (current_df[common_cols] != self.master_df[common_cols]).sum().sum()
        return nans+dupes+mismatches

    def _is_delete_column_abuse(self,current_df: pd.DataFrame,dropped_column_name) -> int:
        master_column_count=len(self.master_df.columns)
        current_column_count=len(current_df.columns)
        if (master_column_count-current_column_count) > 1 and current_df[dropped_column_name].isna().sum()<len(current_df[dropped_column_name]):
            return True
        else:
            return False

    def step(self, action_input: ActionModel) -> Tuple[ObservationModel, RewardModel]:
        """Executes one cleaning action and returns the result."""
        self.turn_count += 1
        current_error_count = self._calculate_total_errors(self.current_df.copy())
        self.prev_error_count = current_error_count
        self.last_eda_result = None  # Clear old tool outputs
        info_msg = ""

        # --- THE SWITCHBOARD (Execution) ---
        try:
            act = action_input.action
            col = action_input.column_name

            current_df_copy=self.current_df.copy()

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
        is_perfect = self.current_df.equals(self.master_df)

        new_error_count = self._calculate_total_errors(self.current_df.copy())
        error_count_reward = (self.prev_error_count - new_error_count)/self.initial_error_count
        is_delete_column_abuse = self._is_delete_column_abuse(current_df_copy, action_input.column_name)
        delete_column_abuse=-0.2 if is_delete_column_abuse else 0
        done = is_perfect or self.turn_count >= self.max_turns

        return self._get_observation(), RewardModel(score=reward, done=done, info={"message": info_msg})