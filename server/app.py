import uvicorn
import argparse
import sys
from pathlib import Path
from openenv.core.env_server import create_web_interface_app

path = Path(__file__).resolve().parent.parent
sys.path.append(str(path))

try:
    from models import ActionModel, ObservationModel
    from environment import DataCleaningEnv
except ModuleNotFoundError:
    from ..models import ActionModel, ObservationModel
    from .environment import DataCleaningEnv


app = create_web_interface_app(
    DataCleaningEnv,
    ActionModel,
    ObservationModel,
)



def main(host:str="0.0.0.0", port:int=7860):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--port", type=int, default=7860)
     args = parser.parse_args()
     main(port=args.port)