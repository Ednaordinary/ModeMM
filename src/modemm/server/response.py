from queue import Queue
from typing import Any


class QueuedResponse:
    def __init__(self):
        self.queue = Queue()

    def wait(self) -> Any:
        obj = self.queue.get()
        while obj is not None:
            yield obj
            obj = self.queue.get()


class EOS:
    pass
