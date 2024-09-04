import json


class Config_Basic:
    def __init__(self):
        pass

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            self.config = json.loads(f.read())
