import json
from pathlib import Path

from ultralytics import YOLO

pavimentados_path = Path(__file__).parent.parent


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


class YoloV8Model(BaseModel):
    def __init__(
            self,
            device="0",
            config_file=pavimentados_path / "configs" / "yolo_config.json",
            general_config_file=pavimentados_path / "configs" / "models_general.json",
            artifacts_path=None,
            model_path_config_key="yolov8_signal_path"
    ):
        super().__init__()
        self.device = device
        self.config = self.load_config(config_file)
        self.general_config = self.load_config(general_config_file)
        self.general_path = pavimentados_path / Path(self.general_config["general_path"])

        if artifacts_path:
            self.yolo_signal_path = Path(artifacts_path) / self.config[model_path_config_key]
        else:
            self.yolo_signal_path = self.general_path / self.config[model_path_config_key]

        self.classes_count = None
        self.classes_names = None

        self.load_model()

    def load_model(self):
        self.classes_names_idx = {name: idx for idx, name in enumerate(open(self.yolo_signal_path / "classes.names").read().splitlines())}
        self.classes_idx_names = {idx: name for name, idx in self.classes_names_idx.items()}
        self.classes_names = list(self.classes_names_idx.keys())

        self.classes_count = len(self.classes_names)

        model_path = Path(self.yolo_signal_path) / "model.pt"
        self.model = YOLO(model_path, task='detect')

    def predict(self, data, conf=0.1, iou=0.2):
        results = self.model(list(data), conf=conf, iou=iou, verbose=False)
        boxes = [r.boxes.xyxyn.cpu().numpy().tolist() for r in results]
        classes = [r.boxes.cls.cpu().int().tolist() for r in results]
        scores = [r.boxes.conf.cpu().numpy().tolist() for r in results]
        return boxes, scores, classes
