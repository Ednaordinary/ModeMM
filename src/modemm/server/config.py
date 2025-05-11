import json

from typing import Dict

class ModemmConfigBase:
    """
    A Modemm base class. Doesn't do much by itself.
    """
    path: str
    """The path to the Modemm config file."""

    def __init__(self, path: str):
        self.path = path

    def _load(self) -> Dict:
        with open(self.path, "r") as config_file:
            return json.load(config_file)

    def get(self, arg: str = None):
        """
        Get the config or variable from the config.
        :param arg: The specific item to return from the config. If None, returns the config. (defaults to None)
        :return: ModemmConfig, Config item
        """
        return None

    def save(self, config: dict):
        """Saves the config to initialized path"""
        with open(self.path, "w") as config_file:
            json.dump(config, config_file, indent=4)


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
