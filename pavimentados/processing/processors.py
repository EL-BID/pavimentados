import logging
import pickle  # nosec
from datetime import datetime
from pathlib import Path

import cv2
from tqdm.autonotebook import tqdm

from pavimentados.configs.utils import Config_Basic
from pavimentados.models.yolov8 import YoloV8Model
from pavimentados.processing.utils import draw_outputs

logger = logging.getLogger(__name__)
pavimentados_path = Path(__file__).parent.parent


class Image_Processor:
    """Predicts the signals over frames."""

    def __init__(self, yolo_device: str = "0", artifacts_path: str = None, config: dict = None):
        self.artifacts_path = artifacts_path
        self.yolo_device = yolo_device
        self.config = config
        self.load_models()

    def load_models(self) -> None:
        """Load the models required for the object detection and tracking
        tasks.

        This function initializes and loads the following models:
        - `yolov8_signal_model`: A YoloV8Model object for detecting signals in the input images.
        - `yolov8_paviment_model`: A YoloV8Model object for detecting pavements in the input images.

        Args:
            None

        Returns:
            None
        """
        logger.info("Loading models...")
        self.yolov8_signal_model = YoloV8Model(
            device=self.yolo_device, model_config_key="signal_model", artifacts_path=self.artifacts_path, config=self.config
        )
        self.yolov8_paviment_model = YoloV8Model(
            device=self.yolo_device, model_config_key="paviment_model", artifacts_path=self.artifacts_path, config=self.config
        )

    def save_images_and_scores(self, image, crop_images, scores, embeddings, predictions, boxes):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pickle_filename = f"prediction_{timestamp}.pkl"

        data = {
            "image": image,
            "crop_images": crop_images,
            "scores": scores,
            "embeddings": embeddings,
            "predictions": predictions,
            "boxes": boxes,
        }
        with open(pickle_filename, "wb") as file:
            pickle.dump(data, file)


class MultiImage_Processor(Config_Basic):
    def __init__(self, config_file: Path = None, artifacts_path=None):
        super().__init__()
        self.yolo_device = "0"

        logger.info("Loading model config...")
        config_file_default = pavimentados_path / "configs" / "models_general.json"
        self.load_config(config_file_default, config_file)

        self.signal_model_enabled = self.config["signal_model_enabled"] = self.config["signal_model"].get("enabled", True)
        self.paviment_model_enabled = self.config["paviment_model_enabled"] = self.config["paviment_model"].get("enabled", True)

        self.processor = Image_Processor(yolo_device=self.yolo_device, artifacts_path=artifacts_path, config=self.config)

    def _process_batch(self, offset, img_batch, video_output=None, image_folder_output=None):
        boxes_pav, scores_pav, classes_pav = self.processor.yolov8_paviment_model.predict(img_batch)
        boxes_signal, scores_signal, classes_signal = self.processor.yolov8_signal_model.predict(img_batch)

        if video_output or image_folder_output:
            j = 0
            for img in img_batch:
                img = img.astype("uint8")
                img = draw_outputs(
                    img, ([boxes_pav[j]], [scores_pav[j]], [classes_pav[j]]), self.processor.yolov8_paviment_model.classes_names
                )
                img = draw_outputs(
                    img,
                    ([boxes_signal[j]], [scores_signal[j]], [classes_signal[j]]),
                    self.processor.yolov8_signal_model.classes_names,
                    classes_signal[j],
                )

                if video_output:
                    video_output.write(img)
                if image_folder_output:
                    frame_file = str(Path(image_folder_output) / f"frame_{offset + j:0>6}.png")
                    cv2.imwrite(frame_file, img)
                j += 1
        return (list(boxes_pav), list(boxes_signal), list(scores_pav), list(scores_signal), list(classes_pav), list(classes_signal))

    def process_images_group(self, img_obj, batch_size=8, video_output_file=None, image_folder_output=None):
        len_imgs = img_obj.get_len()
        if video_output_file:
            altura, base = img_obj.get_altura_base()
            fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
            video_output = cv2.VideoWriter(video_output_file, fourcc, 20.0, (base, altura))
        else:
            video_output = None

        results = list(
            tqdm(
                map(
                    lambda x: self._process_batch(
                        x, img_obj.get_batch(x, batch_size), video_output=video_output, image_folder_output=image_folder_output
                    ),
                    [offset for offset in range(0, len_imgs, batch_size)],
                ),
                total=int(len_imgs // batch_size) + int((len_imgs % batch_size) > 0),
                desc="Processing batches",
            )
        )
        results = list(zip(*results))
        return {
            "boxes_pav": sum(results[0], []),
            "boxes_signal": sum(results[1], []),
            "scores_pav": sum(results[2], []),
            "scores_signal": sum(results[3], []),
            "classes_pav": sum(results[4], []),
            "classes_signal": sum(results[5], []),
            "final_pav_clases": [
                [self.processor.yolov8_paviment_model.classes_idx_names.get(elem, "<UNK>") for elem in item] for item in sum(results[4], [])
            ],
        }
