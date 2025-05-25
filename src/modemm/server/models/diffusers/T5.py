from typing import Dict, Any, List, Union
import gc
import traceback

import torch

from ..base import ModemmModel, write_default_kwargs
from ...errors import ModemmError
from .diff_errors import T5MaxLengthError
from ...response import NPYTensor, QueuedResponse, NestedJSON

class T5Model(ModemmModel):
    """
    A model wrapper for T5
    """
    accept_kwargs: Dict[str, Any] = {"prompt": str, "max_length": int, "pad": bool}
    default_kwargs: Dict[str, Any] = {"max_length": 512, "pad": True}
    requires: List[str] = ["torch", "transformers", "sentencepiece", "numpy", "accelerate", "google.protobuf"]
    streamable: bool = False

    def __init__(self, path: str):
        self.path = path
        self._model = None
        self._tokenizer = None

    async def load(self, device="cuda") -> bool:
        try:
            import torch
            from transformers import T5EncoderModel, T5TokenizerFast
            if not hasattr(self, "_model") or not isinstance(self._model, T5EncoderModel):
                self._model = T5EncoderModel.from_pretrained(self.path, torch_dtype=torch.float16)
            self._model.to(device)
            if not hasattr(self, "_tokenizer") or not isinstance(self._tokenizer, T5TokenizerFast):
                self._tokenizer = T5TokenizerFast.from_pretrained(self.path)
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
        kwargs = write_default_kwargs(self, kwargs)
        if kwargs["max_length"] > 512:
            return T5MaxLengthError()
        try:
            text_input_ids = self._tokenizer(
                kwargs["prompt"],
                padding="max_length" if kwargs["pad"] else "do_not_pad",
                max_length=kwargs["max_length"],
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            input_ids = text_input_ids.input_ids
            attn_mask = [int(x) for x in text_input_ids.attention_mask[0]]
            prompt_embeds = self._model(input_ids.to("cuda"), output_hidden_states=False)[0]
            result = {"prompt_embeds": NPYTensor(prompt_embeds), "attn_mask": attn_mask}
            result = NestedJSON(result)
            return self._return(result, streamer)
        except Exception as e:
            print(traceback.format_exc())
            error = ModemmError("Failed during call")
            return self._return(error, streamer)
