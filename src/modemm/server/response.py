from queue import Queue
from typing import Any

from fastapi.responses import Response

from .errors import ModemmError

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

class PromptEmbeds:
    def __init__(self, embeds):
        self.embeds = embeds

    def to_json(self):
        return Response(self.embeds)
