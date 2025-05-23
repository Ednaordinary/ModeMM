import argparse
from typing import Union, Dict, Any

from fastapi import FastAPI, Request

from .middleware import ContentSizeMiddleware
from .models.base import validate_kwargs
from .model_handler import ModelHandlerBase, ModelExecutor
from .config import ModemmConfigDynamic, ModemmConfigStatic
from .errors import ModelNotFound, ModelNotLoaded
from .util import kwarg_types_name

def build(args: argparse.Namespace) -> FastAPI:
    """
    Builds the Modemm API
    :param args: A namespace built by argparse
    :return: The Modemm API
    """
    config = ModemmConfigDynamic(args.config_file) if args.dynamic_config else ModemmConfigStatic(args.config_file)
    config.register()
    executor = ModelExecutor()
    handler = ModelHandlerBase(config, executor)
    if args.no_advertise:
        app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None, swagger_ui_oauth2_redirect_url=None)
    else:
        app = FastAPI(title="Modemm Server", version="0.0.1")


    # This limits the maximum incoming content size to make sure the server isn't overwhelmed.
    app.add_middleware(ContentSizeMiddleware, max_content=args.max_income)

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
        return list(config.registered["models"].keys())

    @app.get("/modemm/models/{model_id}/kwargs")
    def get_kwargs(model_id: str):
        models = get_models()
        if model_id not in models:
            return {"state": "error", "error": ModelNotFound(model_id).get_error()}
        return kwarg_types_name(config.registered["models"][model_id].accept_kwargs)

    @app.get("/modemm/request/{model_id}")
    def make_request(model_id: str, request: Request, arguments: dict = None, stream: bool = True):
        kwargs = arguments
        if kwargs is None:
            kwargs = {}
        models = get_models()
        if model_id not in models:
            return {"state": "error", "error": ModelNotFound(model_id).get_error()}
        errors = validate_kwargs(config.registered["models"][model_id], kwargs)
        if errors:
            return {"state": "error", "error": errors.get_error()}
        loaded = handler.allocate(model_id)
        if not loaded:
            handler.deallocate(model_id)
            return {"state": "error", "error": ModelNotLoaded(model_id).get_error()}
        run = handler.run(model_id, stream=stream, kwargs=kwargs)
        handler.deallocate(model_id)
        return run

    return app