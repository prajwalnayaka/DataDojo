from enum import Enum
from pydantic import BaseModel, Field
from openenv.core.env_server.interfaces import Action, Observation, State
from typing import Optional, Dict, Any, List


class ActionType(str, Enum):
    # Level 1: Easy
    DROP_DUPLICATES = "DROP_DUPLICATES"
    DROP_COLUMN = "DROP_COLUMN"
    FILL_NA = "FILL_NA"

    # Level 2: Medium
    STRIP_CHAR = "STRIP_CHAR"
    TYPE_CAST = "TYPE_CAST"
    LOWERCASE = "LOWERCASE"

    # Level 3: Hard (The Categorical Detective MVP)
    GET_VALUE_COUNTS = "GET_VALUE_COUNTS"
    MAP_VALUES = "MAP_VALUES"


class ActionModel(Action):
    """The strict schema the agent must output to interact with the dataset."""

    action: ActionType = Field(..., description="The specific cleaning or EDA tool to execute.")
    column_name: Optional[str] = Field(default=None, description="The target column. Required for almost all actions.")

    # --- Tool-Specific Parameters ---
    fill_value: Optional[str] = Field( default=None, description="Required for 'FILL_NA'. Can be a specific value, 'mean', 'median' or 'mode'.")

    target_type: Optional[str] = Field(default=None, description="Required for 'TYPE_CAST'. E.g., 'float', 'int', 'string'.")

    regex_pattern: Optional[str] = Field(default=None, description="Required for 'STRIP_CHAR'. The regex pattern to remove.")

    mapping_dict: Optional[Dict[str, str]] = Field(default=None, description="Required for 'MAP_VALUES'. A dictionary of {old_value: new_value} to fix typos. e.g., {'new york': 'New York', 'NY': 'New York'} or {'option 1' : 'Option '}")


class ObservationModel(Observation):
    """The strict schema the environment must output, allowing the agent to understand the dataset."""
    done: bool = Field(...)

    reward: float = Field(...)

    metadata: Dict[Any, Any] = Field(...)

    data_schema: Dict[str,str] = Field(...,description="The datatypes of the dataset's columns. e.g., {'Price': 'object', 'Age': 'float64'}")

    NaNs:Dict[str,int] = Field(...,description="The NaN values in the dataset's columns. e.g., {'Transport': 38, 'Charges': 76}")

    sample:List[Dict[str,Any]] = Field(...,description="Sample of the dataset, each dictionary in the list is one row in the dataset. e.g., [{'Price': '$1,250', 'Age': 25}, {'Price': 400, 'Age': null}]")

    info: str = Field(...,description="'Success: Dropped column 'Unnamed: 0' OR 'Error: Cannot cast '$1,250' to float directly.'")

    EDA:Optional[Dict[str,Any]] = Field(default=None ,description="The results of GET_VALUE_COUNTS tool call. e.g., {'Option A':300, 'option A':230}")


class StateModel(State):
    """The schema for the environment's state."""
    episode_id:str = Field(...,description="Episode ID.")

    step_count:int = Field(...,description="Number of steps taken.")

    difficulty:str = Field(...,description="Difficulty of the dataset and subsequently the environment.")

    max_steps:int = Field(...,description="Max number of turns the agent has.")


class RewardModel(BaseModel):
    """The strict schema of the reward provided by the environment to the agent."""

    observation: ObservationModel = Field(...)

    reward:float = Field(...)

    done:bool = Field(...)

    info:str = Field(...,description="'Success: Dropped column 'Unnamed: 0' OR 'Error: Cannot cast '$1,250' to float directly.'")