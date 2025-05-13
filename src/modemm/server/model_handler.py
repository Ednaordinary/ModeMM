from typing import Dict, Tuple, List, Any
from queue import Queue

from .config import ModemmConfigBase
from .models.base import ModemmModel


class QueuedResponse:
    def __init__(self):
        self.queue = Queue()

    def wait(self) -> List[Any]:
        response = []
        obj = self.queue.get()
        while obj is not None:
            response.append(obj)
            obj = self.queue.get()
        return response

    def wait_stream(self) -> Any:
        obj = self.queue.get()
        while obj is not None:
            yield obj
            obj = self.queue.get()


class ModelHandlerBase:
    """
    The ModelHandler classes define how a model is loaded when a request is made. For instance, a model loader could
    choose to only load at request time or could stay loaded all the time. This base class ignores everything else
    and loads the model when requested and unloads it when there are no users, which may not be a good idea.
    """

    def __init__(self, config: ModemmConfigBase):
        self.config = config
        self.configured_models: Dict[str, ModemmModel] = {}
        self.loaded_models: Dict[str, int] = {}

    def allocate(self, model: str) -> Tuple[ModemmModel, bool]:
        """
        Waits for a requests model to be loaded
        :param model: The model to allocate
        :return: Whether the model loaded successfully
        """
        if model in self.loaded_models.keys():
            self.loaded_models[model] += 1
            return self.configured_models[model], True
        else:
            if "models" not in self.config.registered.keys():
                self.config.register()
            self.configured_models[model] = self.config.registered["models"][model]
            self.loaded_models[model] = 1
            loaded = self.configured_models[model].load()
            return self.configured_models[model], loaded

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

    # current issue: how to run this in as few threads as possible
    def run(self, model: str) -> QueuedResponse:
        pass
