from typing import Dict, Any, Union, List
from PIL import Image

from ..errors import ModemmError, ArgumentError, ArgValueError


class FakeModel:
    def __init__(self):
        pass

    @staticmethod
    def __call__(self, **kwargs):
        return "meow"

    @staticmethod
    def load(self):
        return True

    @staticmethod
    def unload(self):
        return True


class ModemmModel:
    """
    A ModemmModel that can process a request. The model accepts specific kwargs and hardcodes others. Hardcoded
    kwargs may be added by specifying a value in default_kwargs but not in accept_kwargs.
    """

    def __init__(self):
        self._model = FakeModel()  # Underlying model from a different library
        self.accept_kwargs: Dict[str, Any] = {}
        self.default_kwargs: Dict[str, Any] = {}

    def load(self) -> bool:
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

    def __call__(self, **kwargs) -> Union[str, Image, ModemmError]:
        errors = validate_kwargs(self, kwargs)
        if errors:
            return errors
        kwargs = write_default_kwargs(self, kwargs)
        return self._model(**kwargs)


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
    for kwarg in kwargs.keys():
        if kwarg not in model.accept_kwargs.keys():
            bad_kwargs.append(kwarg)
    for kwarg, value in kwargs.items():
        if kwarg not in bad_kwargs:
            if not isinstance(value, model.accept_kwargs[kwarg]):
                bad_values.append(tuple((kwarg, type(value))))
    errors = []
    if bad_kwargs:
        errors.append(ArgumentError(bad_kwargs))
    if bad_values:
        errors.append(ArgValueError(bad_values))
    return errors if errors else None
