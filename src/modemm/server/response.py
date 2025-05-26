import io
from queue import Queue
from typing import Any, List
import base64
import ujson

import torch
import numpy as np
import imageio as iio
from PIL import Image

from .errors import ModemmError
from .util import np_save


class QueuedResponse:
    def __init__(self):
        self.queue = Queue()

    def wait(self) -> Any:
        obj = self.queue.get()
        while obj is not EOS:
            if hasattr(obj, "to_json"):
                yield ujson.dumps(obj.to_json())
            elif isinstance(obj, ModemmError):
                obj = ujson.dumps({"state": "error", "error": obj.get_error()})
                yield obj
            else:
                yield str(obj)
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
        response = {"tensor": base64.b64encode(tensor).decode('UTF-8')}
        return response


class NestedJSON:
    """
    Nests multiple responses together
    """

    def __init__(self, responses: dict):
        self.responses = responses

    def to_json(self):
        response = {}
        for i, v in self.responses.items():
            if hasattr(v, "to_json"):
                response[i] = v.to_json()
            else:
                response[i] = v
        return response

class PILVideo:
    """
    Encodes a list of PIL images into a h.264 video, then returns the bytes representing it
    """
    def __init__(self, video: List[Image], fps):
        self.video = video
        self.fps = fps

    def to_json(self):
        request = iio.core.Request("<bytes>", mode="w", extension=".mp4")
        pyavobj = iio.plugin.pyav.PyAVPlugin(request)
        pyavobj.init_video_stream("libx264", fps=self.fps, pixel_format=None)
        pyavobj._video_stream.options = {'crf': '20'}
        frames = np.array(self.video)
        vid_bytes = pyavobj.write(frames, codev="libx264", fps=self.fps)
        vid_bytes = base64.b64encode(vid_bytes).decode('UTF-8')
        return vid_bytes