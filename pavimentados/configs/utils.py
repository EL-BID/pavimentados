import json


class Config_Basic:
    def __init__(self):
        pass

    def load_config(self, config_file) -> None:
        """Loads a configuration from the specified file.

        Args:
            config_file (str): The path to the configuration file.

        Returns:
            None
        """
        with open(config_file, "r") as f:
            self.config = json.loads(f.read())
