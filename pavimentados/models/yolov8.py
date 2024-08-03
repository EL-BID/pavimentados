from pathlib import Path

from ultralytics import YOLO

from pavimentados.models.base import BaseModel

pavimentados_path = Path(__file__).parent.parent


class YoloV8Model(BaseModel):
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

    def load_model(self):
        self.classes_names_idx = {
            name: idx for idx, name in enumerate(open(self.yolo_signal_path / self.classes_filename).read().splitlines())
        }
        self.classes_idx_names = {idx: name for name, idx in self.classes_names_idx.items()}
        self.classes_names = list(self.classes_names_idx.keys())
        self.classes_count = len(self.classes_names)

        model_path = Path(self.yolo_signal_path) / self.model_filename
        self.model = YOLO(model_path, task="detect")

    def predict(self, data):
        results = self.model(list(data), conf=self.yolo_threshold, iou=self.yolo_iou, max_det=self.yolo_max_detections, verbose=False)
        boxes = [r.boxes.xyxyn.cpu().numpy().tolist() for r in results]
        classes = [r.boxes.cls.cpu().int().tolist() for r in results]
        scores = [r.boxes.conf.cpu().numpy().tolist() for r in results]
        return boxes, scores, classes
