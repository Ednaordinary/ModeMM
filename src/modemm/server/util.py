import importlib.util
from typing import List


def check_requires(modules: List[str]) -> List[str]:
    """
    Checks if a list of required modules is installed
    :param modules: A list of modules to check
    :return: A list of modules that are not installed
    """
    errors = []
    for i in modules:
        if not package_available(i):
            errors.append(i)
    return errors


def init_import(modules: List[str]):
    """
    Imports modules on config.register() so they are cached for later
    :param modules: A list of modules to cache
    :return:
    """
    for i in modules:
        _ = importlib.import_module(i)


def package_available(module: str) -> bool:
    """
    Checks if a single module is available
    :param module: The module to check
    :return: Whether the module is installed
    """
    return importlib.util.find_spec(module) is not None


def kwarg_types_name(kwargs: dict):
    """
    Transforms an accepted kwargs dict into one with string instead of direct types
    :param kwargs: The input kwargs list
    :return: The kwargs list with strings
    """
    return {k: v.__name__ for k, v in kwargs.items()}
