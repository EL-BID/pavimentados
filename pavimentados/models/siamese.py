import pickle
from pathlib import Path

import numpy as np
import onnxruntime
import tensorflow as tf

from pavimentados.models.base import BaseModel

pavimentados_path = Path(__file__).parent.parent


# def preprocess_image(image_path, image_size):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, image_size)
#     image = image / 255.0
#     return image


class Siamese_Model(BaseModel):
    def __init__(
            self,
            device=None,
            config_file=pavimentados_path / "configs" / "siamese_config.json",
            general_config_file=pavimentados_path / "configs" / "models_general.json",
            artifacts_path=None,
    ):
        super().__init__()
        self.device = device
        self.config = self.load_config(config_file)
        self.general_config = self.load_config(general_config_file)

        if artifacts_path:
            self.siamese_path = Path(artifacts_path) / self.config["SIAMESE_PATH"]
        else:
            self.siamese_path = pavimentados_path / self.general_path / self.config["SIAMESE_PATH"]

        self.image_size = tuple(self.config["SIAMESE_IMAGE_SIZE"])

        with open(str(self.siamese_path / 'embeddings_references.pickle'), 'rb') as f:
            self.embeddings_references = pickle.load(f)

        self.load_model()

    def load_model(self):
        self.model = onnxruntime.InferenceSession(str(self.siamese_path / 'onnx_siamese_model.onnx'))

    def predict(self, data):
        img = np.float32(data)

        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        pred = self.model.run([output_name], {input_name: img})[0]

        score = np.max(pred, axis=1).tolist()
        prediction = [sorted([(np.dot(p, self.embeddings_references[k].T).mean(), k) for k in self.embeddings_references.keys()])[-1][1] for p in pred]

        return score, prediction, prediction
