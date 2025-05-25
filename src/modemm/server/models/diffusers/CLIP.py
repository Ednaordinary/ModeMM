from typing import Dict, Any, List, Union
import gc
import traceback

import torch

from ..base import ModemmModel, validate_kwargs, write_default_kwargs
from ...errors import ModemmError
from ...response import NPYTensor, QueuedResponse

class CLIPModel(ModemmModel):
    """
    A model wrapper for CLIP
    """
    accept_kwargs: Dict[str, Any] = {"prompt": str}
    default_kwargs: Dict[str, Any] = {}
    requires: List[str] = ["torch", "transformers", "numpy", "accelerate"]
    streamable: bool = False

    def __init__(self, path: str):
        self.path = path
        self._model = None
        self._tokenizer = None

    async def load(self, device="cuda") -> bool:
        try:
            import torch
            from transformers import CLIPTextModel, CLIPTokenizerFast
            if not hasattr(self, "_model") or not isinstance(self._model, CLIPTextModel):
                self._model = CLIPTextModel.from_pretrained(self.path, torch_dtype=torch.float16)
            self._model.to(device)
            if not hasattr(self, "_tokenizer") or not isinstance(self._tokenizer, CLIPTokenizerFast):
                self._tokenizer = CLIPTokenizerFast.from_pretrained(self.path)
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
            if hasattr(self, "_tokenizer"):
                del self._tokenizer
                self._tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            return False

    async def __call__(self, streamer: Union[QueuedResponse, None] = None, **kwargs) -> Union[ModemmError, NPYTensor]:
        errors = validate_kwargs(self, kwargs)
        if errors:
            return self._return(errors, streamer)
        kwargs = write_default_kwargs(self, kwargs)
        try:
            text_input_ids = self._tokenizer(
                kwargs["prompt"],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            ).input_ids
            prompt_embeds = self._model(text_input_ids.to("cuda"), output_hidden_states=False)[0]
            result = NPYTensor(prompt_embeds)
            return self._return(result, streamer)
        except Exception as e:
            print(traceback.format_exc())
            error = ModemmError("Failed during call")
            return self._return(error, streamer)
