from typing import Union, List, Tuple, Any


class ModemmError:
    """
    An error from the Modemm server
    """
    error_type = "General Error"

    def __init__(self, info):
        """
        Construct a general Modemm error
        :param info:
        """
        self.info = info

    def get_error(self) -> str:
        """
        Gets info about the error
        :return: str
        """
        return self.error_type + "\n" + str(self.info)

class StackedErrors(ModemmError):
    """
    Multiple errors from the Modemm server
    """
    error_type = "Stacked Error"
    def __init__(self, errors: List[ModemmError]):
        self.errors = errors

    def get_error(self):
        if len(self.errors) == 0:
            return "No errors occured"
        elif len(self.errors) == 1:
            return self.errors[0].get_error()
        else:
            return "The following errors occured:\n" + "\n".join([x.get_error() for x in self.errors])

class ArgumentError(ModemmError):
    """
    An error from the Modemm server specifying a bad argument.
    """
    error_type = "Argument Error"

    def __init__(self, arg: Union[str, List[str]]):
        """
        Construct an argument Modemm Error
        :param arg: The argument or arguments that the model could not accept
        """
        super().__init__(arg)

    def get_error(self):
        arg = self.info
        if isinstance(arg, list):
            if len(arg) == 1:
                self.info = "Argument is not accepted by the model: " + arg[0]
            else:
                self.info = "Arguments are not accepted by the model: " + ", ".join(arg)
        else:
            self.info = "Argument is not accepted by the model: " + arg


def format_arg_value_error(arg) -> str:
    info = "Values are not accepted for arguments by the model: "
    info += ", ".join(["(" + str(x[0]) + ": " + str(x[1].__name__) + ")" for x in arg])
    return info


class ArgValueError(ModemmError):
    """
    An error from the Modemm server specifying a bad value.
    """
    error_type = "Value Error"

    def __init__(self, arg: Union[Tuple[str, Any], List[Tuple[str, Any]]]):
        """
        Construct a value Modemm Error
        :param arg: The argument and value that are not accepted by the model, or a list of them
        """
        super().__init__(arg)

    def get_error(self) -> str:
        arg = self.info
        if isinstance(arg, list):
            if len(arg) == 1:
                arg = arg[0]
                self.info = arg[1].__name__ + " is not accepted for " + arg[0] + " by the model"

            else:
                self.info = format_arg_value_error(arg)
        else:
            self.info = arg[1].__name__ + " is not accepted for " + arg[0] + " by the model"
        return self.info

class ArgRequiredError(ModemmError):
    """
    An error from the Modemm server specifying a required option.
    """
    error_type = "Required Option"

    def __init__(self, arg: Union[str, List[str]]):
        """
        Construct a value Modemm Error
        :param arg: The argument that was not specified
        """
        super().__init__(arg)

    def get_error(self) -> str:
        arg = self.info
        if isinstance(arg, list):
            if len(arg) == 1:
                arg = arg[0]
                self.info = arg + " was required but not found"

            else:
                self.info = "The following args were not specified: " + ", ".join(arg)
        else:
            self.info = arg + " was required but not found."
        return self.info

class ModelNotFound(ModemmError):
    """
    An error from the Modemm server specifying a bad model.
    """
    error_type = "Model Not Found"

    def __init__(self, model: str):
        """
        Construct a Model Not Found Error
        :param model: The model that could not be found
        """
        super().__init__(model)


class ModelNotLoaded(ModemmError):
    """
    An error from the Modemm server specifying the model failed to load
    """
    error_type = "Model Didn't Load"

    def __init__(self, model: str):
        """
        Construct a Model Loading Error
        :param model: The model that could not be loaded
        """
        super().__init__(model)
