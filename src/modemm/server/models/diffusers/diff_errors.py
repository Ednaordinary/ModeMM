from ...errors import ModemmError


class T5MaxLengthError(ModemmError):
    """
    The max length of T5 was set too high
    """
    error_type = "T5 Max Length"

    def __init__(self):
        super().__init__("The maximum length of T5 is 512")

class BadAttnMask(ModemmError):
    """
    Something's wrong with the attention mask
    """
    error_type = "Bad Attention Mask"

    def __init__(self):
        super().__init__("Something's wrong with the attention mask")

class BadLatentShapeError(ModemmError):
    """
    Attempted to make a latent with a bad shape
    """
    error_type = "Bad Latent Shape"

    def __init__(self):
        super().__init__("The latent shape is too large or too small")


class BadTensor(ModemmError):
    """
    Failed to load a tensor over the network
    """
    error_type = "Bad Tensor"

    def __init__(self):
        super().__init__("Failed to load a tensor")
