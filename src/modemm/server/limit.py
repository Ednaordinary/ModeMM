class LimiterBase:
    """
    Basis for limiting requests. By default, does not limit anything.
    """

    def __init__(self):
        pass

    @staticmethod
    def limit(self, *args, **kwargs) -> bool:
        """
        Called on every request. If true, the request should be dropped.
        :param self:
        :param args: Args passed to the limiter function
        :param kwargs: Kwargs passed to the limiter function
        :return: bool
        """
        return False


class IPLimiter(LimiterBase):
    """
    Limits requests based on an amount allowed per a scrolling 24-hour window.
    """

    def __init__(self, max_rate):
        super().__init__()
        self.max_rate = max_rate
        self.limits = []

    def limit(self, *args, **kwargs) -> bool:
        return False
