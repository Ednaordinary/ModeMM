from typing import Dict, Any, Union, List
from PIL import Image

from ..errors import ModemmError, ArgumentError, ArgValueError


def fake_model(**kwargs):
    return "meow"


class ModemmModel:
    def __init__(self):
        self._model = fake_model
        self.accept_kwargs: Dict[str, Any] = {}

    def __call__(self, **kwargs) -> Union[str, Image, ModemmError]:
        return self._model(**kwargs)


def validate_kwargs(model: ModemmModel, **kwargs) -> Union[ModemmError, List[ModemmError], None]:
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
