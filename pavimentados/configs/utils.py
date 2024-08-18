import json
from pathlib import Path


def update(d, u):
    for k, v in u.items():
        if isinstance(d, dict):
            if isinstance(v, dict):
                r = update(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}
    return d


class Config_Basic:
    def __init__(self):
        self.config = None

    def load_config(self, config_file: Path, config_file_update: Path = None) -> None:
        """Loads a configuration from the specified file.

        Args:
            config_file (Path): The path to the configuration file.
            config_file_update (Path: The path to the updated configuration file.

        Returns:
            None
        """
        with open(str(config_file), "r") as f:
            self.config = json.loads(f.read())

        if config_file_update:
            with open(str(config_file_update), "r") as f:
                self.config = update(self.config, json.loads(f.read()))
