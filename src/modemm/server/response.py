import io
from queue import Queue
from typing import Any

import torch
import numpy as np

from fastapi.responses import Response

from .errors import ModemmError
from .util import np_save

class QueuedResponse:
    def __init__(self):
        self.queue = Queue()

    def wait(self) -> Any:
        obj = self.queue.get()
        while obj is not None:
            if hasattr(obj, "to_json"):
                yield obj.to_json()
            if isinstance(obj, ModemmError):
                obj = {"state": "error", "error": obj.get_error()}
            else:
                yield obj
            obj = self.queue.get()


class EOS:
    pass


class Progress:
    def __init__(self, current, total):
        self.current = current
        self.total = total

    def to_json(self) -> dict:
        return {"state": "progress", "progress": [self.current, self.total]}

class NPYTensor:
    """
    A response with a torch tensor.
    """
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def to_json(self):
        """
        Converts the tensor to bytes
        :return: A fastapi response with bytes representing a npy file
        """
        tensor = self.tensor.float().numpy(force=True).astype(np.float16)
        tensor_io = io.BytesIO()
        np_save(tensor_io, tensor)
        tensor_io.seek(0)
        tensor = tensor_io.read()
        return Response(tensor)
