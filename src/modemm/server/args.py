import os
from typing import Dict
from pydantic_settings import BaseSettings

#args_list = {"MODEMM_D_CONFIG": "dynamic_config", "MODEMM_CONFIG_FILE": "config_file"}

class ModemmSettings(BaseSettings):
    dynamic_config: str = "true"
    config_file: str = "config.json"

def get_args(args_list: Dict[str, str]) -> Dict[str, str]:
    """
    Get arguments for the server from enviroment variables
    :param args_list: A mapping between env variables and dictionary values
    :return: A dictionary with the arguments specified and found
    """
    args = {}
    for i in args_list:
        if i[0] in os.environ.keys():
            args[i[1]] = os.environ[i[0]]

    return args

def parse_bool(arg: str) -> bool:
    """
    Parses a string argument which correlates to a bool
    :param arg:
    :return:
    """
    if arg.strip().lower() == "true":
        return True
    else:
        return False