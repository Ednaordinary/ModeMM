import traceback
from typing import Dict, Any, List, Union
import base64
import gc
import io

from PIL import Image
import torch

from ..base import ModemmModel, write_default_kwargs
from ...response import QueuedResponse, EOS, Progress, NPYTensor
from .diff_errors import BadLatentShapeError, T5MaxLengthError, BadTensor, BadAttnMask
from ...errors import ModemmError, ArgRequiredError
from ...util import np_load


# {"name": "LTXLatent", "module": "modemm.server.models.diffusers.ltxv", "class":  "LTXEmptyLatent", "init_kwargs":  {}},
# Empty latents for LTX are not supported right now.

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
            print(latents.shape)
            result = NPYTensor(latents)
            return self._return(result, streamer)
        except:
            error = ModemmError("Failed during call")
            return self._return(error, streamer)


class FakeLTXVae:
    """
    This class exists to fix an error in the pipeline
    """

    def __init__(self):
        self.dtype = None


class LTX096DVideoModel(ModemmModel):
    """
    A model wrapper for LTX Video 0.9.6 distilled
    """
    accept_kwargs: Dict[str, Any] = {"prompt_embeds": str, "height": int, "width": int, "frames": int,
                                     "attn_mask": list}
    default_kwargs: Dict[str, Any] = {
        "width": 1216,
        "height": 704,
        "frames": 141,
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

    async def load(self, device="cuda") -> bool:
        try:
            from diffusers import LTXConditionPipeline
            if not isinstance(self._model, LTXConditionPipeline):
                self._model = LTXConditionPipeline.from_single_file(self.path, text_encoder=None, tokenizer=None,
                                                                    vae=None, torch_dtype=torch.bfloat16)
                self._model.vae = FakeLTXVae()
            self._model.to(device)
            self._model.scheduler._shift = 150.0
        except Exception as e:
            print(traceback.format_exc())
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

        output = self._model(**kwargs, callback=callback).frames

        streamer.queue.put(NPYTensor(output))

        streamer.queue.put(EOS)

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        kwargs["prompt_embeds"] = base64.b64decode(kwargs["prompt_embeds"].encode('UTF-8'))
        kwargs["num_frames"] = kwargs.pop("frames", 141)
        kwargs["prompt_attention_mask"] = kwargs.pop("attn_mask", None)
        if kwargs["prompt_attention_mask"] is None:
            error = ArgRequiredError("attn_mask")
            return self._return(error, streamer)
        if not all((x == 1 or x == 0) for x in kwargs["prompt_attention_mask"]):
            error = BadAttnMask()
            return self._return(error, streamer)
        if len(kwargs["prompt_embeds"]) > 4194432:
            error = T5MaxLengthError()
            return self._return(error, streamer)
        try:
            tensor_io = io.BytesIO(kwargs["prompt_embeds"])
            tensor_io.seek(0)
            prompt_embeds = np_load(tensor_io, (1, 512, 4096))
            del tensor_io
        except:
            error = BadTensor()
            print(traceback.format_exc())
            return self._return(error, streamer)
        if prompt_embeds is None:
            error = BadTensor()
            return self._return(error, streamer)
        if prompt_embeds.shape[2] != 4096:
            error = BadTensor()
            return self._return(error, streamer)
        kwargs["prompt_embeds"] = torch.tensor(prompt_embeds).to("cuda", dtype=torch.bfloat16)
        kwargs["prompt_attention_mask"] = torch.tensor(kwargs["prompt_attention_mask"]).unsqueeze(0).to("cuda",
                                                                                                        dtype=torch.bfloat16)
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            output = self._model(**kwargs).frames  # this is actually a latent! it is silly
            output = NPYTensor(output)
            return self._return(output, streamer)


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

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[str, Image, ModemmError]:
        kwargs = write_default_kwargs(self, kwargs)
        kwargs["prompt_embeds"] = bytes(kwargs["prompt_embeds"])
        if self.streamable and streamer is not None:
            self._stream(streamer, **kwargs)
        else:
            return self._return(self._model(**kwargs), streamer)
