from typing import Dict, Any, Union, List
from PIL import Image
import time

from ..response import QueuedResponse, EOS
from ..errors import ModemmError, StackedErrors, ArgumentError, ArgValueError, ArgRequiredError


class FakeModel:
    """
    A fake model that return meow
    """

    def __init__(self):
        pass

    @staticmethod
    def __call__(**kwargs):
        """
        Call the fake model in order to return meow
        :param kwargs: kwargs passed to the fake model (dropped)
        :return: meow
        """
        return "meow"

    @staticmethod
    def load():
        """
        Pretend to load the fake model
        :return: if the model loaded correctly
        """
        return True

    @staticmethod
    def unload():
        """
        Pretend to unload the fake model
        :return: if the model unloaded correctly
        """
        return True

    @staticmethod
    def stream(**kwargs) -> str:
        """
        Stream 10 meows
        :param kwargs: kwargs passed to the fake model (dropped)
        :return: 10 meows
        """
        for i in range(10):
            time.sleep(0.1)
            yield "meow\n"


class ModemmModel:
    """
    A ModemmModel that can process a request. The model accepts specific kwargs and hardcodes others. Hardcoded
    kwargs may be added by specifying a value in default_kwargs but not in accept_kwargs.
    """
    accept_kwargs: Dict[str, Any] = {}
    default_kwargs: Dict[str, Any] = {}
    requires: List[str] = []
    streamable: bool = True

    def __init__(self):
        self._model = FakeModel()  # Underlying model from a different library

    def load(self, device=None) -> bool:
        """
        Defines how a model is loaded. This method cannot have kwargs since loading logic should be handled by the
        ModelHandler :return: A bool stating whether the model was successfully loaded
        """
        return self._model.load()

    def unload(self) -> bool:
        """
        Defines how a model is unloaded. This method cannot have kwargs since unloading logic should be handled by the
        ModelHandler :return: A bool stating whether the model was successfully unloaded
        """
        return self._model.unload()

    def _return(self, obj: Any, streamer):
        if streamer is not None:
            streamer.queue.put(obj)
            streamer.queue.put(EOS)
            return
        else:
            return obj

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        errors = validate_kwargs(self, kwargs)
        if errors:
            return self._return(errors, streamer)
        kwargs = write_default_kwargs(self, kwargs)
        if self.streamable and streamer is not None:
            for i in self._model.stream(**kwargs):
                streamer.queue.put(i)
            streamer.queue.put(EOS)
        else:
            return self._return(self._model(**kwargs), streamer)


def write_default_kwargs(model: ModemmModel, kwargs: dict) -> dict:
    """
    Processes default kwargs and introduces hardcoded kwargs
    :param model: The ModemmModel that the request has specified
    :param kwargs: The kwargs being sent by the request
    :return: The kwargs to run through the model
    """
    new_kwargs = model.default_kwargs.copy()
    new_kwargs.update(kwargs)
    return new_kwargs


def validate_kwargs(model: ModemmModel, kwargs: dict) -> Union[ModemmError, List[ModemmError], None]:
    """
    Validates kwargs sent to the model before running it
    :param model: The ModemmModel that the request has specified
    :param kwargs: The kwargs being sent by the request
    :return: None, an error, or a list of errors
    """
    bad_kwargs = []
    bad_values = []
    req_values = []
    for kwarg in kwargs.keys():
        if kwarg not in model.accept_kwargs.keys():
            bad_kwargs.append(kwarg)
    for kwarg, value in kwargs.items():
        if kwarg not in bad_kwargs:
            if not isinstance(value, model.accept_kwargs[kwarg]):
                bad_values.append(tuple((kwarg, type(value))))
    for kwarg in model.accept_kwargs.keys():
        if kwarg not in model.default_kwargs.keys() and kwarg not in kwargs.keys():
            req_values.append(kwarg)
    errors = []
    if bad_kwargs:
        errors.append(ArgumentError(bad_kwargs))
    if bad_values:
        errors.append(ArgValueError(bad_values))
    if req_values:
        errors.append(ArgRequiredError(req_values))
    return StackedErrors(errors) if errors else None
