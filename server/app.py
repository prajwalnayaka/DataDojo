import uvicorn
import argparse
from openenv.core.env_server.http_server import create_app

try:
    from models import ActionModel, ObservationModel
    from environment import DataCleaningEnv
except ModuleNotFoundError:
    from .models import ActionModel, ObservationModel
    from .environment import DataCleaningEnv

app = create_app(
    DataCleaningEnv,
    ActionModel,
    ObservationModel,
    env_name="DataDojo",
    max_concurrent_envs=5
)

def main(host:str="0.0.0.0", port:int=8080):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--port", type=int, default=8080)
     args = parser.parse_args()
     main(port=args.port)