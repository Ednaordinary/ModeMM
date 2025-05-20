import io
from typing import Dict, Any, List, Union
import gc

import torch
from PIL import Image

from ..base import ModemmModel, write_default_kwargs
from ...response import QueuedResponse, EOS, Progress, NPYTensor
from ...errors import ModemmError, BadLatentShapeError, T5MaxLengthError, BadTensor
from ...util import np_load

class LTXEmptyLatent(ModemmModel):
    """
    Generate LTX latents
    """
    accept_kwargs: Dict[str, Any] = {"height": int, "width": int, "frames": int, "seed": int}
    default_kwargs: Dict[str, Any] = {
        "width": 704,
        "height": 512,
        "frames": 121,
        "seed": None,
    }
    requires: List[str] = ["torch"]
    streamable: bool = False

    def __init__(self):
        pass

    async def load(self, device=None) -> bool:
        return True

    async def unload(self) -> bool:
        return True

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        try:
            height = kwargs["height"] // 32
            width = kwargs["width"] // 32
            print(height, width)
            frames = (kwargs["frames"] - 1) // 8 + 1
            if (height * width) > 880 or (height * width) < 1 or frames > 161:
                error = BadLatentShapeError()
                return self._return(error, streamer)
            shape = (1, 128, frames, height, width)
            generator = torch.Generator(device='cpu')
            if kwargs["seed"]:
                generator.manual_seed(kwargs["seed"])
            latents = torch.randn(shape, generator=generator, dtype=torch.float16)
            result = NPYTensor(latents)
            return self._return(result, streamer)
        except:
            error = ModemmError("Failed during call")
            return self._return(error, streamer)


class LTX096DVideoModel(ModemmModel):
    """
    A model wrapper for LTX Video 0.9.6 distilled
    """
    accept_kwargs: Dict[str, Any] = {"prompt_embeds": bytes, "latents": bytes}
    default_kwargs: Dict[str, Any] = {
        "width": 1216,
        "height": 704,
        "num_frames": 141,
        "num_inference_steps": 8,
        "max_sequence_length": 512,
        "decode_timestep": 0.05,
        "guidance_scale": 1,
        "output_type": "latent",
    }
    requires: List[str] = ["torch", "diffusers"]
    streamable: bool = True

    def __init__(self, path: str, steps: int):
        self.path = path
        self.steps = steps
        self._model = None

    def load(self, device="cuda") -> bool:
        try:
            from diffusers import LTXConditionPipeline
            if not isinstance(self._model, LTXConditionPipeline):
                self._model = LTXConditionPipeline.from_single_file(self.path, text_encoder=None, tokenizer=None,
                                                                    vae=None, torch_dtype=torch.bfloat16)
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
        if len(kwargs["prompt_embeds"]) > 4194432:
            error = T5MaxLengthError()
            return self._return(error, streamer)
        try:
            tensor_io = io.BytesIO(kwargs["prompt_embeds"])
            tensor_io.seek(0)
            tensor = np_load(tensor_io, (1, 512, 4096))
        except:
            error = BadTensor()
            return self._return(error, streamer)
        if not tensor:
            error = BadTensor()
            return self._return(error, streamer)
        if tensor.shape[2] != 4096:
            error = BadTensor()
            return self._return(error, streamer)
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
    streamable: bool = False

    def __init__(self, path: str, steps: int):
        self.path = path
        self.steps = steps
        self._model = None

    def load(self, device="cuda") -> bool:
        try:
            from diffusers import AutoencoderKLLTXVideo
            if not isinstance(self._model, AutoencoderKLLTXVideo):
                self._model = AutoencoderKLLTXVideo.from_single_file(self.path, torch_dtype=torch.bfloat16)
            self._model.to(device)
            self._model.enable_tiling()
            self._model.use_framewise_decoding = True
            self._model.tile_sample_stride_height = 704
            self._model.tile_sample_stride_width = 704
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

        self._model(**kwargs)

        streamer.queue.put(EOS)

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        kwargs["prompt_embeds"] = bytes(kwargs["prompt_embeds"])
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            return self._return(self._model(**kwargs), streamer)
