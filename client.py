from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import ActionModel, ObservationModel, StateModel

class DataDojoEnv(EnvClient[ActionModel, ObservationModel, StateModel]):

    def _step_payload(self, action: ActionModel)->dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=ObservationModel(
                done=obs_data.get("done", False),
                reward=obs_data.get("reward", 0.0),
                metadata=obs_data.get("metadata", {}),
                data_schema=obs_data.get("data_schema", {}),
                NaNs=obs_data.get("NaNs", {}),
                sample=obs_data.get("sample", []),
                info=obs_data.get("info", ""),
                EDA=obs_data.get("EDA", None)
            ),
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )
    def _parse_state(self, payload: dict) -> StateModel:
        return StateModel(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count"),
            difficulty=payload.get("difficulty"),
            max_steps=payload.get("max_steps"),
        )