import json


class BaseModel:
    def __init__(self, device=""):
        self.model = None
        self.device = device

    def predict(self, data):
        pass

    def load_model(self):
        pass

    def load_config(self, config_file: str) -> dict:
        """Loads a configuration from the specified file.

        Args:
            config_file (str): The path to the configuration file.

        Returns:
            The loaded configuration.
        """
        with open(config_file, "r") as f:
            return json.loads(f.read())
