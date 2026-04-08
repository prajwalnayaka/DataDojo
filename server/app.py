import uvicorn
import argparse
import sys
from pathlib import Path
from openenv.core.env_server import create_web_interface_app

path = Path(__file__).resolve().parent.parent
sys.path.append(str(path))


from models import ActionModel, ObservationModel
from .environment import DataCleaningEnv



app = create_web_interface_app(
    DataCleaningEnv,
    ActionModel,
    ObservationModel,
)



def main(host:str="0.0.0.0", port:int=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()