from pathlib import Path

import numpy as np
from ultralytics import YOLO

from pavimentados.models.base import BaseModel

pavimentados_path = Path(__file__).parent.parent


class YoloV8Model(BaseModel):
    def __init__(
        self,
        device: str = "0",
        config_file: str = pavimentados_path / "configs" / "models_general.json",
        model_config_key: str = "",
        artifacts_path: str = None,
    ):
        """
        Initializes the YoloV8Model with the specified device, configuration file, model configuration key, and artifacts path.

        Args:
            device (str): The device to use for the model, default is "0".
            config_file (str): The path to the configuration file, default is "configs/models_general.json".
            model_config_key (str): The key in the model's configuration file.
            artifacts_path (str): The path to the artifacts directory.

        Returns:
            None
        """
        super().__init__()
        self.device = device
        self.config = self.load_config(config_file)

        if artifacts_path:
            self.general_path = Path(artifacts_path)
        else:
            self.general_path = Path(self.config["general_path"])

        self.yolo_signal_path = self.general_path / self.config[model_config_key]["path"]
        self.model_filename = self.config[model_config_key]["model_filename"]
        self.classes_filename = self.config[model_config_key]["classes_filename"]

        self.yolo_threshold = self.config[model_config_key]["yolo_threshold"]
        self.yolo_iou = self.config[model_config_key]["yolo_iou"]
        self.yolo_max_detections = self.config[model_config_key]["yolo_max_detections"]

        self.classes_count = None
        self.classes_names = None
        self.classes_idx_names = None
        self.classes_names_idx = None

        self.load_model()

    def load_model(self) -> None:
        """
        Load the YOLOv8 model and initialize the necessary attributes.

        This function loads the classes names and their corresponding indices from the classes file. It then creates a dictionary
        mapping the class names to their indices and vice versa. The class names are extracted from the dictionary keys and
        stored in the `classes_names` attribute. The count of classes is calculated and stored in the `classes_count` attribute.

        The model file path is constructed using the `yolo_signal_path` and `model_filename` attributes. The YOLOv8 model is then
        loaded using the `YOLO` class from the YOLOv8 library, with the task set to "detect". The loaded model is stored in the
        `model` attribute.

        Parameters:
            self (YoloV8Model): The instance of the YoloV8Model class.

        Returns:
            None
        """
        self.classes_names_idx = {
            name: idx for idx, name in enumerate(open(self.yolo_signal_path / self.classes_filename).read().splitlines())
        }
        self.classes_idx_names = {idx: name for name, idx in self.classes_names_idx.items()}
        self.classes_names = list(self.classes_names_idx.keys())
        self.classes_count = len(self.classes_names)

        model_path = Path(self.yolo_signal_path) / self.model_filename
        self.model = YOLO(model_path, task="detect")

    def predict(self, data: np.ndarray) -> tuple[list, list, list]:
        """
        Predict boxes, scores, and classes for the given data.

        Args:
            data (np.ndarray): The images to predict.

        Returns:
            tuple: A tuple containing the predicted boxes, scores, and classes.
        """
        results = self.model(list(data), conf=self.yolo_threshold, iou=self.yolo_iou, max_det=self.yolo_max_detections, verbose=False)
        boxes = [r.boxes.xyxyn.cpu().numpy().tolist() for r in results]
        classes = [r.boxes.cls.cpu().int().tolist() for r in results]
        scores = [r.boxes.conf.cpu().numpy().tolist() for r in results]
        return boxes, scores, classes
