from typing import Dict, Any, List, Union
import gc

from PIL import Image

from ..base import ModemmModel, write_default_kwargs
from ...response import QueuedResponse, EOS, Progress
from ...errors import ModemmError

class LTX096DVideoModel(ModemmModel):
    """
    A model wrapper for LTX Video 0.9.6 distilled
    """
    accept_kwargs: Dict[str, Any] = {"prompt_embeds": bytes}
    default_kwargs: Dict[str, Any] = {
        "width": 1216,
        "height": 704,
        "num_frames": 141,
        "num_inference_steps": 8,
        "max_sequence_length": 512,
        "decode_timestep": 0.05,
        "guidance_scale": 1,
    }
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
                self._model = LTXConditionPipeline.from_single_file(self.path, text_encoder=None, tokenizer=None, vae=None, torch_dtype=torch.bfloat16)
            self._model.to(device)
            self._model.scheduler._shift = 150.0
        except Exception as e:
            return False
        else:
            return True

    async def unload(self) -> bool:
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

        self._model(**kwargs, callback=callback)

        streamer.queue.put(EOS)


    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        kwargs["prompt_embeds"] = bytes(kwargs["prompt_embeds"])
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            return self._return(self._model(**kwargs), streamer)

class LTXVaeModel(ModemmModel):
    """
    A model wrapper for an LTX Vae
    """
    accept_kwargs: Dict[str, Any] = {"prompt_embeds": bytes}
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
            from diffusers import AutoencoderKLLTXVideo
            if not isinstance(self._model, AutoencoderKLLTXVideo):
                self._model = AutoencoderKLLTXVideo.from_single_file(self.path, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
            self._model.to(device)
            self._model.vae.enable_tiling()
            self._model.vae.use_framewise_decoding = True
            self._model.vae.tile_sample_stride_height = 704
            pipe.vae.tile_sample_stride_width = 704
            self._model.scheduler._shift = 150.0
        except Exception as e:
            return False
        else:
            return True

    async def unload(self) -> bool:
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

        self._model(**kwargs, callback=callback)

        streamer.queue.put(EOS)


    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        kwargs["prompt_embeds"] = bytes(kwargs["prompt_embeds"])
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            return self._return(self._model(**kwargs), streamer)