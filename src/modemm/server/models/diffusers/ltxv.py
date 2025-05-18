from typing import Dict, Any, List, Union
import gc

from PIL import Image

from ..base import ModemmModel, validate_kwargs, write_default_kwargs
from ...response import QueuedResponse, EOS, Progress
from ...errors import ModemmError

class LTX096DVideoModel(ModemmModel):
    """
    A model wrapper for LTX Video 0.9.6 distilled
    """
    accept_kwargs: Dict[str, Any] = {"prompt": str}
    default_kwargs: Dict[str, Any] = {}
    requires: List[str] = ["torch", "diffusers"]
    streamable: bool = True

    def __init__(self, path: str, steps: int):
        self.path = path
        self.steps = steps
        self._model = None

    def load(self, device="cuda") -> bool:
        try:
            import torch
            from diffusers import LTXConditionPipeline
            if not isinstance(self._model, LTXConditionPipeline):
                self._model = LTXConditionPipeline.from_single_file(self.path, torch_dtype=torch.bfloat16)
            self._model.to(device)
            self._model.vae.enable_tiling()
        except Exception as e:
            return False
        else:
            return True

    def unload(self) -> bool:
        try:
            if hasattr(self, "_model"):
                del self._model
                self._model = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            return False

    def _stream(self, streamer, **kwargs):
        def callback(pipe, i, t, pipe_kwargs):
            streamer.put(Progress(i, self.steps))
            return pipe_kwargs

        self._model()

        streamer.queue.put(EOS)


    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        errors = validate_kwargs(self, kwargs)
        if errors:
            return self._return(errors, streamer)
        kwargs = write_default_kwargs(self, kwargs)
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            return self._return(self._model(**kwargs), streamer)
