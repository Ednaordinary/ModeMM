from typing import Dict, Any
import traceback
import threading
import asyncio

from fastapi.responses import StreamingResponse

from .config import ModemmConfigBase
from .models.base import ModemmModel
from .response import QueuedResponse, EOS
from .errors import ModemmError, ModelNotLoaded


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
            loaded = asyncio.run_coroutine_threadsafe(self.configured_models[model].load(),
                                                      loop=self.executor.loop).result()
            if loaded:
                self.loaded_models[model] = 1
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
                asyncio.run_coroutine_threadsafe(self.configured_models[model].unload(),
                                                 loop=self.executor.loop)
                try:
                    del self.loaded_models[model]
                except:
                    pass
                return True
            else:
                return True
        else:
            return True

    async def call_wrapper(self, model, kwargs, streamer=None):
        """
        It is a bad idea to break if something in the call breaks, so this function wraps call to provide debug and
        keep the engine running
        """
        try:
            out = await model(kwargs, streamer=streamer)
        except:
            print(traceback.format_exc())
            error = ModemmError("Failed during call")
            if streamer:
                streamer.queue.put(error)
                streamer.queue.put(EOS)
            else:
                return error
        else:
            if not streamer:
                return out

    def _run_stream(self, model, model_id, kwargs):
        streamer = QueuedResponse()
        asyncio.run_coroutine_threadsafe(self.call_wrapper(model, kwargs, streamer=streamer), loop=self.executor.loop)
        for i in streamer.wait():
            yield i
        # Eventually, allocation and deallocation should happen in two separate threads and/or event loops
        # For now, this blocks the requests completion until the model is done unloading
        self.deallocate(model_id)
        return

    def run(self, model: str, stream, kwargs) -> Any:
        model_id = model
        loaded = self.allocate(model_id)
        if not loaded:
            self.deallocate(model_id)
            return {"state": "error", "error": ModelNotLoaded(model_id).get_error()}
        model = self.configured_models[model_id]
        if stream and model.streamable:
            return StreamingResponse(self._run_stream(model, model_id, kwargs))
        else:
            result = asyncio.run_coroutine_threadsafe(self.call_wrapper(model, kwargs), loop=self.executor.loop).result()
            if hasattr(result, "to_json"):
                result = result.to_json()
            if isinstance(result, ModemmError):
                result = {"state": "error", "error": result.get_error()}
            self.deallocate(model_id)
            return result
