import argparse
from typing import Union, Dict, Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .model_handler import ModelHandlerBase, ModelExecutor
from .config import ModemmConfigDynamic, ModemmConfigStatic
from .errors import ModelNotFound, ModelNotLoaded


def build(args: argparse.Namespace) -> FastAPI:
    """
    Builds the Modemm API
    :param args: A namespace built by argparse
    :return: The Modemm API
    """
    config = ModemmConfigDynamic(args.config_file) if args.dynamic_config else ModemmConfigStatic(args.config_file)
    executor = ModelExecutor()
    handler = ModelHandlerBase(config, executor)
    if args.no_advertise:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, swagger_ui_oauth2_redirect_url=None)
    else:
        app = FastAPI(title="Modemm Server", version="0.0.1")

    @app.get("/")
    def root():
        # By default, lets return the sub paths for different features unless no-advertise is specified
        if args.no_advertise:
            return {}
        else:
            return {"config": "/modemm/config", "models": "/modemm/models"}

    @app.get("/modemm/config")
    def get_config():
        return config.get()

    @app.get("/modemm/models")
    def get_models():
        return [x["name"] for x in config.get()["models"]]

    def run_handler(handler, model_id, stream, kwargs):
        # definition for StreamingResponse
        return handler.run(model_id, stream=stream, kwargs=kwargs)

    @app.get("/modemm/request/{model_id}")
    def make_request(model_id: str, request: Request, kwargs: Union[Dict[str, Any], None] = None, stream: bool = True):
        if kwargs is None:
            kwargs = {}
        models = get_models()
        if model_id not in models:
            return {"error": ModelNotFound(model_id).get_error()}
        loaded = handler.allocate(model_id)
        if not loaded:
            return {"error": ModelNotLoaded(model_id).get_error()}
        if stream:
            return StreamingResponse(run_handler(handler, model_id, stream=stream, kwargs=kwargs))
        else:
            return run_handler(handler, model_id, stream=stream, kwargs=kwargs)

    return app
