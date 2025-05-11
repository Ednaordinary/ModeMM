import argparse

from fastapi import FastAPI, Request

from .config import ModemmConfigDynamic, ModemmConfigStatic
from .errors import ModelNotFound


def main():
    """
    Runs the Modemm server
    """
    parser = argparse.ArgumentParser(description='Run time config for Modemm')
    parser.add_argument("-c", "--config_file", type=str, help="Server config file path", default="./config.json")
    parser.add_argument("-d", "--dynamic-config", type=bool, help="Use the dynamic config loader", default=True)

    args = parser.parse_args()

    config = ModemmConfigDynamic(args.config_file) if args.dynamic_config else ModemmConfigStatic(args.config_file)
    app = FastAPI(title="Modemm Server")

    @app.get("/")
    def root():
        # By default, lets return the sub paths for different features
        return {"config": "/server/config", "models": "/server/models"}  # placeholder

    @app.get("/server/config")
    def get_config():
        return config.get()

    @app.get("/server/models")
    def get_models():
        return config.get()["models"]

    @app.get("/server/request/{model_id}")
    def make_request(model_id: str, request: Request):
        models = get_models()
        if model_id not in models:
            return ModelNotFound(model_id).get_error()


if __name__ == "__main__":
    main()
