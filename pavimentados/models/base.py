import json


class BaseModel:
    def __init__(self, device=""):
        self.model = None
        self.device = None

    def predict(self, data):
        pass

    def load_model(self):
        pass

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            return json.loads(f.read())
