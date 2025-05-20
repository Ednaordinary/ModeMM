from typing import Optional


# vaguely copied from
# https://github.com/steinnes/content-size-limit-asgi/blob/master/content_size_limit_asgi/middleware.py

class ContentSizeMiddleware:
    """
    Limit content size of an incoming request
    """

    def __init__(self, app, max_content: Optional[int] = None):
        self.app = app
        self.max_content = max_content

    def receive_wrapper(self, receive):
        received = 0

        async def inner():
            nonlocal received
            message = await receive()
            if message["type"] != "http.request" or self.max_content is None:
                return message
            body_len = len(message.get("body", b""))
            received += body_len
            if received > self.max_content:
                raise IOError("Max content size exceeded")
            return message

        return inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        wrapper = self.receive_wrapper(receive)
        await self.app(scope, wrapper, send)
