import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from pavimentados.processing.sources import (
    ListImages,
    ListRoutesImages,
    VideoCaptureImages,
)

logger = logging.getLogger(__name__)
pavimentados_path = Path(__file__).parent.parent


def draw_outputs(img, outputs, classes_labels, final_classes=None):
    """Draws bounding boxes and labels on an image based on the outputs from a
    model.

    Args:
        img (np.ndarray): The input image.
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray]): The outputs from the model.
        classes_labels (List[str]): The list of class labels.
        final_classes (Optional[List[str]]): The final class labels.

    Returns:
        np.ndarray: The image with bounding boxes and labels drawn on it.
    """

    color = (255, 0, 0)
    boxes, objectness, classes = outputs
    boxes, objectness, classes = boxes[0], objectness[0], classes[0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, color, 2)

        # Create the label text with class name and score
        label = f"{classes_labels[int(classes[i])]}"
        label = f"{final_classes[i]}" if final_classes is not None else label
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        x1, y1 = x1y1
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + 10), color, cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img


def get_fotogram(fps: int, frame_count: int, idx: int) -> np.ndarray:
    """Get the index of a frame in a video given the frame rate and total frame
    count.

    Args:
        fps (int): The frame rate of the video.
        frame_count (int): The total number of frames in the video.
        idx (int): The index of the frame to retrieve.

    Returns:
        np.ndarray: The index of the frame in the video.
    """
    fotograms = (np.arange(0, frame_count, fps).reshape(-1, 1) + np.arange(0, fps, fps // 2)[:2]).reshape(-1).astype(int)
    return fotograms[idx]


def put_text(frame, text, position, color=(255, 255, 255)):
    """Put text on an image at the given position.

    Args:
        color:
        frame (np.ndarray): The image to draw text on.
        text (str): The text to draw. Supports multiple lines by using the newline character.
        position (Tuple[int, int]): The (x, y) coordinates of the top-left of the text.

    Returns:
        None
    """
    font_scale = 0.4
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    line_height = text_size[1] + 5
    x, y = position
    for i, line in enumerate(text.split("\n")):
        y = y + i * line_height
        cv2.putText(frame, line, (x, y), font, font_scale, color, thickness, line_type)


def create_output_from_results(
    processor: VideoCaptureImages | ListImages | ListRoutesImages,
    signals_detections: list[dict] | None = None,
    fails_detections: list[dict] | None = None,
    video_output_file: str | None = None,
    image_folder_output: str | None = None,
):
    is_video_output = isinstance(processor, VideoCaptureImages)

    # Signals detections
    df_signals = pd.DataFrame(signals_detections if signals_detections is not None else [])
    if len(df_signals) > 0:
        df_signals["type"] = "S"  # Type S=Signal
        df_signals = df_signals[df_signals.position_boxes != 0]  # Remove position_boxes = 0 (left signals)
        df_signals.rename(
            columns={
                "signal_class_siames": "signal_classes_siames",
                "signal_class_base": "signal_classes_base",
                "signal_class_names": "classes_signal_names",
            },
            inplace=True,
        )  # Rename columns

    # Fails dataset
    df_fails = pd.DataFrame(fails_detections if fails_detections is not None else [])
    if len(df_fails) > 0:
        df_fails["type"] = "F"  # Type F=Fail
        df_fails.rename(
            columns={
                "scores": "score",
                "classes": "final_classes",
                "fotograma": "fotogram",
            },
            inplace=True,
        )  # Rename columns

    # Concatenate results
    df_list = []
    if len(df_signals) > 0:
        df_list.append(df_signals)
    if len(df_fails) > 0:
        df_list.append(df_fails)

    if len(df_list) == 0:
        logger.info("No detections to process")
        return

    df = pd.concat(df_list)
    df = df.sort_values(["fotogram"])

    # Create output
    altura, base = processor.get_altura_base()

    if is_video_output:
        fourcc = cv2.VideoWriter.fourcc("m", "p", "4", "v")
        video_output = cv2.VideoWriter(video_output_file, fourcc, 20.0, (base, altura))

    for fotogram_idx in tqdm(sorted(list(df.fotogram.unique())), desc="Processing frames: "):

        frame_text = "id# "
        if is_video_output:
            fotogram = processor.selected_frames[fotogram_idx]
            processor.vidcap.set(cv2.CAP_PROP_POS_FRAMES, fotogram)
            ret, frame = processor.vidcap.read()
            frame_text += f"{fotogram_idx} - frame# {fotogram}"
        else:
            frame = processor.get_batch(fotogram_idx, 1)[0]
            frame_text += f"{fotogram_idx}"

        put_text(frame, frame_text, (20, 30), color=(255, 0, 0))

        for idx, row in df[df.fotogram == fotogram_idx].iterrows():
            boxes = [float(b) for b in row.boxes]
            wh = np.flip(frame.shape[0:2])
            x1y1 = tuple((np.array(boxes[0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[2:4]) * wh).astype(np.int32))

            color = (255, 0, 0)
            if row["type"] == "S":
                color = (137, 243, 54)
                position = x1y1[0], x2y2[1] + 10
            else:
                position = x1y1[0] + 10, x1y1[1] + 20
            text = f"{row.final_classes}\n{round(row.score, 4)}"
            put_text(frame, text, position)

            frame = cv2.rectangle(frame, x1y1, x2y2, color, 2)

        if is_video_output:
            video_output.write(frame)
        else:
            frame_file = str(Path(image_folder_output).resolve() / f"frame_{fotogram_idx:0>6}.png")
            cv2.imwrite(frame_file, frame)

    if is_video_output:
        video_output.release()
