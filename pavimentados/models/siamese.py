import pickle  # nosec B403
from pathlib import Path

import numpy as np
import onnxruntime

from pavimentados.models.base import BaseModel

pavimentados_path = Path(__file__).parent.parent


class Siamese_Model(BaseModel):
    def __init__(
        self,
        config: dict = None,
        device="0",
        model_config_key: str = "",
        artifacts_path: str = None,
    ):
        """Initializes a new instance of the Siamese_Model class.

        Args:
            config (dict): config dictionary.
            device (str, optional): The device to use for computations. Defaults to "0".
            model_config_key (str, optional): The key for the model configuration in the configuration file.
            Defaults to "".
            artifacts_path (str, optional): The path to the artifacts directory. Defaults to None.

        Initializes the following attributes:
            - device (str): The device to use for computations.
            - config (dict): The loaded configuration.
            - general_path (Path): The path to the general directory.
            - siamese_path (Path): The path to the Siamese directory.
            - image_size (tuple): The size of the image.
            - embeddings_filename (str): The filename of the embeddings file.
            - model_filename (str): The filename of the model file.
            - embeddings_references (dict): The loaded embeddings references.

        Loads the model using the `load_model` method.
        """
        super().__init__()
        self.device = device
        self.config = config

        if artifacts_path:
            self.general_path = Path(artifacts_path)
        else:
            self.general_path = Path(self.config["general_path"])

        self.enabled = self.config[model_config_key].get("enabled", False)
        self.siamese_path = self.general_path / self.config[model_config_key]["path"]
        self.image_size = tuple(self.config[model_config_key]["image_size"])
        self.embeddings_filename = self.config[model_config_key]["embeddings_filename"]
        self.model_filename = self.config[model_config_key]["model_filename"]

        self.load_model()

    def load_model(self) -> None:
        """Loads the model.

        Returns:
            None
        """
        if not self.enabled:
            self.model = None
            return

        with open(str(self.siamese_path / self.embeddings_filename), "rb") as f:
            self.embeddings_references = pickle.load(f)  # nosec B301

        self.model = onnxruntime.InferenceSession(str(self.siamese_path / self.model_filename))

    def predict(self, data):
        """Predicts the class labels for the given input data.

        Args:
            data (numpy.ndarray): The input data to be predicted.

        Returns:
            tuple: A tuple containing three elements:
                - score (list): The maximum score for each prediction.
                - prediction (list): The predicted class labels.
                - prediction (list): The predicted class labels.

        This function takes in a numpy array `data` as input and performs the following steps:
        1. Converts the input data to float32 format.
        2. Retrieves the input and output names from the model.
        3. Runs the model on the input data and obtains the predictions.
        4. Calculates the maximum score for each prediction along the second axis.
        5. Sorts the predictions based on the dot product with the embeddings references and retrieves the class labels.
        6. Returns the maximum score, predicted class labels, and the predicted class labels.
        """
        if not self.enabled:
            empty = []
            return empty, empty, empty, empty

        img = np.float32(data)
        input_name = self.model.get_inputs()[0].name
        output_name = self.model.get_outputs()[0].name
        pred = self.model.run([output_name], {input_name: img})[0]
        score = np.max(pred, axis=1).tolist()
        prediction = [
            sorted([(np.dot(p, self.embeddings_references[k].T).max(), k) for k in self.embeddings_references.keys()])[-1][1] for p in pred
        ]
        return score, prediction, prediction, pred
