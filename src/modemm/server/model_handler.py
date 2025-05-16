from typing import Dict, Any
import threading
import asyncio

from fastapi.responses import StreamingResponse

from .config import ModemmConfigBase
from .models.base import ModemmModel
from .response import QueuedResponse

class ModelExecutor:
    """
    A loop to execute model requests in
    """

    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def run_loop(self):
        self.loop.run_forever()


class ModelHandlerBase:
    """
    The ModelHandler classes define how a model is loaded when a request is made. For instance, a model loader could
    choose to only load at request time or could stay loaded all the time. This base class ignores everything else
    and loads the model when requested and unloads it when there are no users, which may not be a good idea.
    """

    def __init__(self, config: ModemmConfigBase, executor: ModelExecutor):
        self.config = config
        self.configured_models: Dict[str, ModemmModel] = {}
        self.loaded_models: Dict[str, int] = {}
        self.executor = executor
        self.runner_thread = threading.Thread(target=self.executor.run_loop)
        self.runner_thread.start()

    def allocate(self, model: str) -> bool:
        """
        Waits for a requests model to be loaded
        :param model: The model to allocate
        :return: Whether the model loaded successfully
        """
        if model in self.loaded_models.keys():
            self.loaded_models[model] += 1
            return True
        else:
            if "models" not in self.config.registered.keys():
                self.config.register()
            self.configured_models[model] = self.config.registered["models"][model]
            self.loaded_models[model] = 1
            loaded = self.configured_models[model].load()
            return loaded

    def deallocate(self, model: str) -> bool:
        """
        Waits for a requests model to be unloaded
        :param model: The model to deallocate
        :return: Whether the model unloaded successfully
        """
        if model in self.loaded_models.keys():
            self.loaded_models[model] -= 1
            if self.loaded_models[model] <= 0:
                unloaded = self.configured_models[model].unload()
                del self.loaded_models[model]
                return unloaded
            else:
                return True
        else:
            return True

    def _run_stream(self, model, kwargs):
        streamer = QueuedResponse()
        asyncio.run_coroutine_threadsafe(model(**kwargs, streamer=streamer), loop=self.executor.loop)
        for i in streamer.wait():
            yield i

    def run(self, model: str, stream, kwargs) -> Any:
        model = self.configured_models[model]
        if stream and model.streamable:
            return StreamingResponse(self._run_stream(model, kwargs))
        else:
            return asyncio.run_coroutine_threadsafe(model(**kwargs), loop=self.executor.loop).result()
