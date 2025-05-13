from .request import ModemmRequest


class ModelHandlerBase:
    """
    The ModelHandler classes define how a model is loaded when a request is made. For instance, a model loader could
    choose to only load at request time or could stay loaded all the time. This base class ignores everything else
    and loads the model when requested, which may not be a good idea.
    """

    def __init__(self):
        pass

    def allocate(self, request: ModemmRequest) -> bool:
        """
        Waits for a requests model to be loaded
        :param request: The request given by the client
        :return: Whether the model loaded successfully
        """
        request.model.load()
