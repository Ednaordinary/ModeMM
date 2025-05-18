import importlib
import ujson
import logging

from typing import Dict, Any, Union

from .models.base import ModemmModel
from .util import check_requires

logger = logging.getLogger(__name__)


class ModemmConfigBase:
    """
    A Modemm base class. Doesn't do much by itself.
    """
    path: str
    """The path to the Modemm config file."""

    def __init__(self, path: str):
        self.path = path
        self.registered = {}

    def _load(self) -> Dict:
        with open(self.path, "r") as config_file:
            return ujson.load(config_file)

    def get(self, arg: str = None) -> Union[Dict, Any]:
        """
        Get the config or variable from the config.
        :param arg: The specific item to return from the config. If None, returns the config. (defaults to None)
        :return: ModemmConfig, Config item
        """
        return {}

    def save(self, config: dict):
        """Saves the config to initialized path"""
        with open(self.path, "w") as config_file:
            ujson.dump(config, config_file, indent=4)

    def register(self):
        config = self.get()
        registrable = ["models"]
        config_keys = config.keys()
        for i in registrable:
            if i in config_keys:
                self.registered[i] = {}
        if "models" in self.registered.keys():
            for i in self.get()["models"]:
                model = str(i["module"])
                model_name = str(i["name"])
                module: ModemmModel = importlib.import_module(model).__dict__[i["class"]](**i["init_kwargs"])
                failed = check_requires(module.requires)
                if failed:
                    logger.warning("The following packages were not found: " + ", ".join(failed))
                    logger.warning(model_name + " will not be registered")
                else:
                    self.registered["models"][model_name] = module


class ModemmConfigDynamic(ModemmConfigBase):
    """
    A Modemm config that dynamically reloads on every request. Good for debugging models.
    """

    def get(self, arg: str = None):
        if arg:
            return self._load()[arg]
        else:
            return self._load()


class ModemmConfigStatic(ModemmConfigBase):
    """
    A Modemm config that loads once.
    """
    _config: dict

    def __init__(self, path: str):
        super().__init__(path=path)
        self._config = self._load()

    def get(self, arg: str = None):
        if arg:
            return self._config[arg]
        else:
            return self._config
