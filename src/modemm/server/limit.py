import time
import logging
from typing import Dict


from fastapi import Request

logger = logging.getLogger(__name__)


class LimiterBase:
    """
    Basis for limiting requests. By default, does not limit anything.
    """

    def __init__(self):
        pass

    @staticmethod
    def limit(self, **kwargs) -> bool:
        """
        Called on every request. If true, the request should be dropped.
        :param self:
        :param kwargs: Kwargs passed to the limiter function
        :return: bool stating whether to limit the function
        """
        return False


class IPRequests:
    """
    Represents a number of requests made by an IP
    """

    def __init__(self, request_times: list = None):
        if request_times is None:
            request_times = []
        self.request_times = request_times

    def add(self, request_time: float):
        """
        Add a time at which a request was successfully made
        :param request_time:
        :return:
        """
        self.request_times.append(request_time)

    def scroll_limits(self, scroll_rate: int):
        for request in self.request_times:
            if request < time.perf_counter() - (scroll_rate * 60):
                try:
                    self.request_times.remove(request)
                except ValueError:
                    logger.warning("Failed to remove a request from the scrolling window (IPLimiter)")


class IPLimiter(LimiterBase):
    """
    Limits requests based on an amount allowed per a scrolling window.
    """

    def __init__(self, max_rate: int = 3, scroll_rate: int = 24 * 60):
        """
        Constructs an IP based rate limiter
        :param max_rate: The maximum amount of requests that can happen within the scrolling window per IP
        :param scroll_rate: The length of the scrolling window in minutes
        """
        super().__init__()
        self.max_rate = max_rate
        self.scroll_rate = scroll_rate
        self.ips: Dict[str, IPRequests] = {}

    def limit(self, **kwargs) -> bool:
        client: Request.client = kwargs["request"]
        if str(client) in self.ips.keys():
            self.ips[str(client)].scroll_limits(self.scroll_rate)
            if self.ips[str(client)] > self.max_rate:
                return True
            else:
                self.ips[str(client)].add(time.time())
        else:
            self.ips[str(client)] = IPRequests()
            self.ips[str(client)].add(time.time())
        return False


class StackedLimiter(LimiterBase):
    """
    Stacks multiple limiters on top of each-other
    """

    def __init__(self, limiters):
        super().__init__()
        self.limiters = limiters

    def limit(self, **kwargs):
        for limiter in self.limiters:
            if limiter.limit(**kwargs):
                return True
        return False
