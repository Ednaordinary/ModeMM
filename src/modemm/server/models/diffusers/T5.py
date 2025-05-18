from typing import Dict, Any, List, Union, Tuple
import gc

from ..base import ModemmModel, validate_kwargs, write_default_kwargs
from ...errors import ModemmError
from ...response import PromptEmbeds

class T5Model(ModemmModel):
    """
    A model wrapper for T5
    """
    accept_kwargs: Dict[str, Any] = {"prompt": str, "max_length": int}
    default_kwargs: Dict[str, Any] = {"max_length": 512}
    requires: List[str] = ["torch", "transformers", "sentencepiece", "numpy"]
    streamable: bool = False

    def __init__(self, path: str, device: str = "cuda"):
        self.path = path
        self.device = device
        self._model = None

    def load(self) -> bool:
        try:
            import torch
            from transformers import T5EncoderModel, T5TokenizerFast
            self._model = T5EncoderModel.from_pretrained(self.path, torch_dtype=torch.bfloat16)
            self._model.to(self.device)
            self._model.vae.enable_tiling()
            self._tokenizer = T5TokenizerFast.from_pretrained(self.path)
        except Exception as e:
            return False
        else:
            return True

    def unload(self) -> bool:
        try:
            del self._model, self._tokenizer
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        except Exception as e:
            return False

    async def __call__(self, **kwargs) -> Union[ModemmError, PromptEmbeds]:
        errors = validate_kwargs(self, kwargs)
        if errors:
            return errors
        kwargs = write_default_kwargs(self, kwargs)
        text_input_ids = self._tokenizer(
            kwargs["prompt"],
            padding="max_length",
            max_length=kwargs["max_length"],
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        ).input_ids
        import torch
        prompt_embeds: torch.FloatTensor = self._model(text_input_ids.to(self.device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.numpy(force=True).tolist()
        return PromptEmbeds(prompt_embeds)
