import pandas as pd
import numpy as np
import os
import uuid
from openenv.core.env_server.interfaces import Environment
from pathlib import Path
from typing import Tuple
from models import ActionModel, ObservationModel, RewardModel, StateModel, ActionType
from .genesis_engine_mk3 import generate_mk3_dataframe
from .ruiner_engine_mk3 import run_ruiner


class DataCleaningEnv(Environment):
    """
    The main Arena. This class manages the state of the dataset and
    evaluates the LLM agent's cleaning attempts.
    """
    SUPPORTS_CONCURRENT_SESSIONS = True
    ENABLE_WEB_INTERFACE = True
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SKELETONS = os.path.join(BASE_DIR, "Skeletons")
    def __init__(self, difficulty: str = "Easy", max_steps: int = 10, skeletons_dir=SKELETONS):
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.skeletons_dir = Path(skeletons_dir)
        self.episode_id=str(uuid.uuid4())
        self.step_count = 0
        self.drop_dupes_counter = 0
        self.reward=0.0
        self.done = False
        self.breakdown = []
        self.info=""
        self.initial_error_count=None
        self.prev_error_count=None
        self.master_df = None
        self.current_df = None
        self.last_eda_result = None

    def _get_observation(self)-> ObservationModel:
        clean_sample = (self.current_df.head(5).astype(object).where(self.current_df.head(5).notnull(), None))
        # Timestamp/Date objects to strings
        processed_records = []
        for record in clean_sample.to_dict(orient="records"):
            new_record = {
                k: (v.isoformat() if hasattr(v, 'isoformat') else v)
                for k, v in record.items()
            }
            processed_records.append(new_record)
        return ObservationModel(
            done=self.done,
            reward=self.reward,
            metadata={"breakdown":self.breakdown},
            data_schema=self.current_df.dtypes.apply(lambda x: x.name).to_dict(),
            NaNs={k: int(v) for k, v in self.current_df.isnull().sum().to_dict().items()},
            sample=processed_records,
            info=self.info,
            EDA=self.last_eda_result
        )

    @property
    def state(self) -> StateModel:
        """Returns the current state of the environment."""
        return StateModel(
        episode_id= self.episode_id,
        step_count= self.step_count,
        difficulty= self.difficulty,
        max_steps= self.max_steps
        )

    def reset(self,difficulty: str = None) -> ObservationModel:
        """Starts a new episode with a fresh, ruined dataset."""
        if difficulty:
            self.difficulty = difficulty
        self.master_df = generate_mk3_dataframe(self.skeletons_dir)
        self.current_df = run_ruiner(self.master_df.copy(), self.difficulty)
        self.step_count = 0
        self.drop_dupes_counter = 0
        self.initial_error_count,_ = self._calculate_total_errors(self.current_df.copy())
        self.prev_error_count,_ = self._calculate_total_errors(self.current_df.copy())
        self.last_eda_result = None
        print(f"DEBUG: Resetting Env ID: {self.episode_id}")

        return self._get_observation()


    def _calculate_total_errors(self, current_df: pd.DataFrame):
        nans=current_df.isnull().sum().sum()
        dupes=current_df.duplicated().sum()
        min_rows = min(len(self.master_df), len(current_df))
        current_df_trim=current_df.iloc[:min_rows].reset_index(drop=True)
        master_df_trim=self.master_df.iloc[:min_rows].reset_index(drop=True)
        common_cols_trim=current_df_trim.columns.intersection(master_df_trim.columns)
        current_df_trim=current_df_trim[common_cols_trim]
        master_df_trim=master_df_trim[common_cols_trim]
        mismatches=((current_df_trim.astype(str)!=master_df_trim.astype(str)) & ~(current_df_trim.isna() & master_df_trim.isna())).sum().sum()
        # ^ Compare as string because STRIP_CHAR will leave a column as str despite the fact that it now (after a perfect STRIP_CHAR ction) may contain only ints/floats. It has to be type casted.
        breakdown="Reward for reducing the total number of errors in the dataset."
        return nans+dupes+mismatches, breakdown

    def _get_deletion_penalty(self, current_df: pd.DataFrame, col: str):
        penalty = 0.0
        master_cols = len(self.master_df.columns)
        current_cols = len(current_df.columns)
        if current_df[col].notna().any():
            penalty -= 0.5
        if current_cols < master_cols: # Yes this applies an action penalty i.e, penalty for everytime the agent uses drop_col EXCEPT for its first use [current_df_copy before first drop_col will have the same number of cols as master]
            penalty -= 0.2             # This encourages the agent to use that free drop_na on the correct empty column in its very first try.
        breakdown="Penalty for deleting a column. Harder penalties for deleting the wrong column."
        return penalty, breakdown

    def step(self, action_input: ActionModel) -> RewardModel:
        """Executes one cleaning action and returns the result."""
        print(f"DEBUG: Stepping Env ID: {self.episode_id}")
        self.step_count += 1
        current_df_copy = self.current_df.copy()
        current_error_count,_ = self._calculate_total_errors(current_df_copy)
        self.prev_error_count = current_error_count
        self.last_eda_result = None  # Clear old tool outputs
        self.breakdown = [] # Clear previous breakdowns
        self.info = "" # Clear previous info messages
        datatype_mismatch = 0
        drop_dupes_spam=0
        drop_col_flag=False
        eda_flag=False

        # --- If-elif switchboard ---
        try:
            act = action_input.action
            col = action_input.column_name


            if act == ActionType.DROP_COLUMN:
                self.current_df.drop(columns=[col], inplace=True)
                drop_col_flag = True
                self.info = f"Successfully dropped column: {col}"

            elif act == ActionType.DROP_DUPLICATES:
                self.current_df.drop_duplicates(inplace=True)
                self.drop_dupes_counter += 1
                self.info = "Successfully removed duplicate rows."

            elif act == ActionType.FILL_NA:
                val = action_input.fill_value
                if val == "mean":
                    fill = self.current_df[col].mean()
                elif val == "median":
                    fill = self.current_df[col].median()
                elif val == "mode":
                    mode_result = self.current_df[col].mode()
                    fill = mode_result.iloc[0] if not mode_result.empty else np.nan
                else:
                    fill = val
                self.current_df[col] = self.current_df[col].fillna(fill)
                self.info = f"Filled NaNs in {col} with {val} ({fill})"

            elif act == ActionType.STRIP_CHAR:
                pattern = action_input.regex_pattern
                self.current_df[col] = self.current_df[col].astype(str).str.replace(pattern, "", regex=True)
                self.info = f"Stripped characters from {col} using pattern: {pattern}"

            elif act == ActionType.TYPE_CAST:
                target = action_input.target_type
                self.current_df[col] = self.current_df[col].astype(target)
                self.info = f"Cast {col} to {target}"

            elif act == ActionType.GET_VALUE_COUNTS:
                # This doesn't change the DF, just provides info
                self.last_eda_result = self.current_df[col].value_counts().to_dict()
                eda_flag = True
                self.info = f"Retrieved value counts for {col}"

            elif act == ActionType.MAP_VALUES:
                mapping = action_input.mapping_dict
                self.current_df[col] = self.current_df[col].replace(mapping)
                self.info = f"Mapped values in {col} using provided dictionary."

            elif act == ActionType.LOWERCASE:
                self.current_df[col] = self.current_df[col].astype(str).str.lower().str.strip()
                self.info = f"Lowercased all values in column: {col}"

        except Exception as e:
            self.info = f"Error executing {action_input.action}: {str(e)}"

        col_diff=abs(len(self.master_df.columns) - len(self.current_df.columns))

        if eda_flag and self.last_eda_result>7:
            self.last_eda_result = dict(list(self.last_eda_result.items())[:7])

        # --- Rewards & Penalties ---

        invalid_col_penalty = 0
        if action_input.column_name:
            col = action_input.column_name
            if col not in self.current_df.columns and action_input.action != ActionType.DROP_DUPLICATES: # Penalty if the agent send a column name that isn't in the dataset at all.
                invalid_col_penalty = -0.1
                self.breakdown.append({"Invalid column name — must use exact column name from the schema.": invalid_col_penalty})
            elif col in self.current_df.columns:
                if self.current_df[col].dtype != self.master_df[col].dtype: # Penalty if the datatypes of the column isn't the same as in master_df. Breakdown encourages the agent to type cast into the correct datatype
                    datatype_mismatch = -0.2                                # master_df because current_df_copy because it's already wrong, courtesy of Ruiner
                    self.breakdown.append({"The datatypes of the selected column is not accurate, needs to be changed.":datatype_mismatch})
        else:
            datatype_mismatch = 0

        new_error_count, error_count_reward_breakdown = self._calculate_total_errors(self.current_df.copy()) # Not current_df_copy as it doesn't reflect the changes that have been just done
        REWARD_MULTIPLIER = 5
        error_count_reward = ((self.prev_error_count - new_error_count)/self.initial_error_count)*REWARD_MULTIPLIER
        self.breakdown.append({error_count_reward_breakdown:round(error_count_reward,3)})

        if drop_col_flag:
            delete_column_abuse, delete_column_abuse_breakdown = self._get_deletion_penalty(current_df_copy, action_input.column_name) # current_copy_df coz we need to inspect the deleted column,
            self.breakdown.append({delete_column_abuse_breakdown: delete_column_abuse})                                                # which is no longer available in current_df_copy
        else:
            delete_column_abuse=0

        if self.drop_dupes_counter>1:
            drop_dupes_spam=-0.2
            self.breakdown.append({"Used DROP_DUPLICATES tool call more than once.":drop_dupes_spam})

        self.reward = round(float(np.tanh(error_count_reward + datatype_mismatch + delete_column_abuse + drop_dupes_spam + invalid_col_penalty)), 4)
        self.reward = round((self.reward + 1.0) / 2.0, 4)  # remap [-1,1] -> (0,1) as float
        self.reward = max(0.001, min(self.reward, 0.999))  # clamp strictly within (0,1)

        self.done = new_error_count < 0.10 * self.initial_error_count or self.step_count >= self.max_steps or col_diff > 1
        # If current dataset's error is less than 10% of what the model started off with OR if current step is greater than max allowed steeps OR difference in columns of current dataset and master dataset is more than 1

        obs = self._get_observation()
        rew = RewardModel(
            observation=obs,
            reward=float(self.reward),
            done=bool(self.done),
            info=self.info,
        )

        return rew