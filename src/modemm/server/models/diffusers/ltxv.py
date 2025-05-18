from typing import Dict, Any, List, Union
import gc

from PIL import Image

from ..base import ModemmModel, validate_kwargs, write_default_kwargs
from ...response import QueuedResponse
from ...errors import ModemmError

class LTX096DVideoModel(ModemmModel):
    accept_kwargs: Dict[str, Any] = {"prompt": str}
    default_kwargs: Dict[str, Any] = {}
    requires: List[str] = ["torch", "diffusers"]
    streamable: bool = True

    def __init__(self, path: str, steps: int):
        self.path = path
        self.steps = steps
        self.model = None

    def load(self) -> bool:
        try:
            import torch
            from diffusers import LTXConditionPipeline
            self.model = LTXConditionPipeline.from_pretrained(self.path, torch_dtype=torch.bfloat16)
            self.model.to("cuda")
            self.model.vae.enable_tiling()
        except Exception as e:
            return False
        else:
            return True

    def unload(self) -> bool:
        try:
            del self.model
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            return False

    def _stream(self, streamer, **kwargs):
        def callback(pipe, i, t, pipe_kwargs):
            streamer.put()
            return pipe_kwargs

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
