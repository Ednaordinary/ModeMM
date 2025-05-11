import argparse

from fastapi import FastAPI, Request

from .config import ModemmConfigDynamic, ModemmConfigStatic
from .errors import ModelNotFound


def build(args: argparse.Namespace) -> FastAPI:
    """
    Builds the Modemm API
    :param args: A namespace built by argparse
    :return: The Modemm API
    """
    print(args.dynamic_config)
    config = ModemmConfigDynamic(args.config_file) if args.dynamic_config else ModemmConfigStatic(args.config_file)
    if args.no_advertise:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, swagger_ui_oauth2_redirect_url=None)
    else:
        app = FastAPI(title="Modemm Server", version="0.0.1")

    @app.get("/")
    def root():
        # By default, lets return the sub paths for different features
        if args.no_advertise:
            return {}
        else:
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
            return {"error": ModelNotFound(model_id).get_error()}

    return app
