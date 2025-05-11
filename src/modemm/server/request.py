class ModemmRequest:
    """
    A request made by a client.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def handle(self):
        return self.model(**self.kwargs)
