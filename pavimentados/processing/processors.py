import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm.autonotebook import tqdm

from pavimentados.configs.utils import Config_Basic
from pavimentados.models.siamese import Siamese_Model
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
        - `siamese_model`: A Siamese_Model object for tracking objects across frames.

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
        self.siamese_model = Siamese_Model(
            device=self.yolo_device, model_config_key="siamese_model", artifacts_path=self.artifacts_path, config=self.config
        )

    def crop_img(self, box: list[float], img: np.ndarray) -> np.ndarray:
        """Crop the image based on the provided box.

        Args:
            box (list[float]): The box to crop the image.
            img (np.ndarray): The image to crop.

        Returns:
            np.ndarray: The cropped image.
        """
        img_crop = img[int(box[1] * img.shape[0]) : int(box[3] * img.shape[0]), int(box[0] * img.shape[1]) : int(box[2] * img.shape[1])]
        img_crop = cv2.resize(img_crop, tuple(self.siamese_model.image_size)[:2], interpolation=cv2.INTER_AREA).astype(float) / 255
        return img_crop

    def predict_signal_state_single(self, image: np.ndarray, box: list[float]):
        """Predict the signal state of the object in the provided image and
        box.

        Args:
            image (np.ndarray): The image to predict the signal state.
            box (list[float]): The box to crop the image.

        Returns:
            tuple: A tuple containing the predicted signal state, the predicted signal base, and the predicted signal.
        """
        if len(box) > 0:
            crop_images = list(map(lambda x: self.crop_img(x, image), box))
            signal_pred_scores, pred_signal_base, pred_signal, embeddings = self.siamese_model.predict(np.array(crop_images))
            return pred_signal, pred_signal_base, pred_signal
        else:
            return [], [], []

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

    def predict_signal_state(self, images: np.ndarray, boxes: list[list[float]]) -> tuple[list, list, list]:
        """
        Predict the signal state of the objects in the provided images and boxes.
        Args:
            images: Images to predict the signal state.
            boxes: List of boxes to crop the images.

        Returns:
            tuple: A tuple containing the predicted signal state, the predicted signal base, and the predicted signal.
        """
        mixed_results = map(lambda img, box: self.predict_signal_state_single(img, box), images, boxes)

        signal_predictions, signal_base_predictions, state_predictions = list(zip(*mixed_results))
        return list(signal_predictions), list(signal_base_predictions), list(state_predictions)


class MultiImage_Processor(Config_Basic):
    def __init__(self, config_file: Path = None, artifacts_path=None):
        super().__init__()
        self.yolo_device = "0"

        logger.info("Loading model config...")
        config_file_default = pavimentados_path / "configs" / "models_general.json"
        self.load_config(config_file_default, config_file)

        self.signal_model_enabled = self.config["signal_model_enabled"] = self.config["signal_model"].get("enabled", True)
        self.paviment_model_enabled = self.config["paviment_model_enabled"] = self.config["paviment_model"].get("enabled", True)

        self.processor = Image_Processor(
            yolo_device=self.yolo_device, artifacts_path=artifacts_path, config=self.config
        )

    def _process_batch(self, offset, img_batch, video_output=None, image_folder_output=None):
        boxes_pav, scores_pav, classes_pav = self.processor.yolov8_paviment_model.predict(img_batch)
        boxes_signal, scores_signal, classes_signal = self.processor.yolov8_signal_model.predict(img_batch)
        final_signal_classes, signal_base_predictions, state_predictions = self.processor.predict_signal_state(img_batch, boxes_signal)

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
                    final_signal_classes[j],
                )

                if video_output:
                    video_output.write(img)
                if image_folder_output:
                    frame_file = str(Path(image_folder_output) / f"frame_{offset+j:0>6}.png")
                    cv2.imwrite(frame_file, img)
                j += 1
        return (
            list(boxes_pav),
            list(boxes_signal),
            list(scores_pav),
            list(scores_signal),
            list(classes_pav),
            list(classes_signal),
            final_signal_classes,
            signal_base_predictions,
            state_predictions,
        )

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
                    lambda x: self._process_batch(x,
                        img_obj.get_batch(x, batch_size), video_output=video_output, image_folder_output=image_folder_output
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
            "final_signal_classes": sum(results[6], []),
            "signal_base_predictions": sum(results[7], []),
            "state_predictions": sum(results[8], []),
        }

    def process_folder(self, folder, batch_size=8):
        folder = Path(folder)
        image_list = list(
            filter(lambda x: str(x).lower().split(".")[-1] in self.config["images_allowed"], map(lambda x: folder / x, os.listdir(folder)))
        )
        return self.process_images_group(image_list)
