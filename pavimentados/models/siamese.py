import pickle  # nosec B403
from pathlib import Path

import numpy as np
import onnxruntime

from pavimentados.models.base import BaseModel

pavimentados_path = Path(__file__).parent.parent


class Siamese_Model(BaseModel):
    def __init__(
        self,
        device="0",
        config_file=pavimentados_path / "configs" / "models_general.json",
        model_config_key: str = "",
        artifacts_path: str = None,
    ):
        super().__init__()
        self.device = device
        self.config = self.load_config(config_file)

        if artifacts_path:
            self.general_path = Path(artifacts_path)
        else:
            self.general_path = Path(self.config["general_path"])

        self.siamese_path = self.general_path / self.config[model_config_key]["path"]
        self.image_size = tuple(self.config[model_config_key]["image_size"])
        self.embeddings_filename = self.config[model_config_key]["embeddings_filename"]
        self.model_filename = self.config[model_config_key]["model_filename"]

        with open(str(self.siamese_path / self.embeddings_filename), "rb") as f:
            self.embeddings_references = pickle.load(f)  # nosec B301

        self.load_model()

    def load_model(self):
        self.model = onnxruntime.InferenceSession(str(self.siamese_path / self.model_filename))

    def predict(self, data):
        img = np.float32(data)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        pred = self.model.run([output_name], {input_name: img})[0]

        score = np.max(pred, axis=1).tolist()
        prediction = [
            sorted([(np.dot(p, self.embeddings_references[k].T).mean(), k) for k in self.embeddings_references.keys()])[-1][1] for p in pred
        ]

        return score, prediction, prediction
