from typing import Any

from .models.base import ModemmModel

class ModemmRequest:
    """
    A request made by a client.
    """

    def __init__(self, model, **kwargs):
        self.model: ModemmModel = model
        self.kwargs: dict = kwargs

    def handle(self) -> Any:
        """
        Runs the request using the original kwargs
        :return: The output of the model
        """
        return self.model(**self.kwargs)
