import importlib.util
from typing import List
import numpy as np

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


def np_save(file, array):
    magic_string = b"\x93NUMPY\x01\x00v\x00"
    header = bytes(("{'descr': '" + array.dtype.descr[0][1] + "', 'fortran_order': False, 'shape': " + str(
        array.shape) + ", }").ljust(127 - len(magic_string)) + "\n", 'utf-8')
    if type(file) is str:
        file = open(file, "wb")
    file.write(magic_string)
    file.write(header)
    file.write(array.data)

def np_load(file):
    if type(file) is str:
        file=open(file,"rb")
    header = file.read(128)
    if not header:
        return None
    descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
    shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(',)', ')').replace(', }', '').replace('(', '').replace(')', '').split(','))
    datasize = np.lib.format.descr_to_dtype(descr).itemsize
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=descr, buffer=file.read(datasize))
