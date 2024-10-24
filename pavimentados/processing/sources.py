import logging
import os
import secrets
from pathlib import Path
from time import sleep

import cv2
import numpy as np

logger = logging.getLogger(__name__)
pavimentados_path = Path(__file__).parent.parent


def load_video(video_path):
    """Loads a video from the specified path."""
    logger.info("Opening video: %s", video_path)
    vidcap = cv2.VideoCapture(video_path)
    retries = 10
    while not vidcap.isOpened():
        logger.debug("Waiting for video to open...")
        sleep(0.1)
        if retries == 0:
            raise ValueError(f"Could not open video: {video_path}")
        retries -= 1

    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    number_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("fps: %s, number_of_frames: %s", fps, number_of_frames)
    return vidcap, fps, number_of_frames


class ListImages:
    def __init__(self, config, images):
        self.config = config
        self.images = images

    def get_altura_base(self):
        return tuple(self.image[0].shape[1:3])

    def get_len(self):
        return self.images.shape[0]

    def get_section(self, idx_inicial, idx_final):
        return self.images[idx_inicial:idx_final]

    def get_batch(self, idx_inicial, batch_size=8):
        return self.get_section(idx_inicial, idx_inicial + batch_size)


class ListRoutesImages:
    def __init__(self, config, routes):
        self.config = config
        self.routes = routes

    def get_altura_base(self):
        return tuple(cv2.imread(str(self.routes[0])).shape[:2])

    def get_len(self):
        return len(self.routes)

    def get_section(self, idx_inicial, idx_final):
        return self.routes[idx_inicial:idx_final]

    def get_batch(self, idx_inicial, batch_size=8):
        return np.array(
            [cv2.imread(str(img_path)) for img_path in self.get_section(idx_inicial, idx_inicial + batch_size)])


class FolderRoutesImages(ListRoutesImages):
    def __init__(self, config, route):
        self.config = config
        folder = Path(route)
        self.routes = list(
            filter(lambda x: str(x).lower().split(".")[-1] in self.config["images_allowed"],
                   map(lambda x: folder / x, os.listdir(folder)))
        )
        self.routes = sorted(self.routes)


class VideoCaptureImages:
    def __init__(self, config, route, images_per_second=2):
        self.config = config
        self.route = str(route)
        self.images_per_second = images_per_second
        self.load_video_capture()

    def load_video_capture(self):
        vidcap, self.fps, self.number_of_frames = load_video(self.route)
        self.images_dict = {
            item: True
            for item in filter(
                lambda x: x < self.number_of_frames,
                (
                        np.arange(0, self.number_of_frames, self.fps).reshape(-1, 1)
                        + np.arange(0, self.fps, self.fps // self.images_per_second)[: self.images_per_second]
                ).reshape(-1),
            )
        }
        self.lenght = len(self.images_dict.keys())
        self.selected_frames = list(map(int, list(self.images_dict.keys())))
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, secrets.randbelow(self.number_of_frames))
        state, img = vidcap.read()
        self.img_shape = img.shape[:2]
        self.vidcap, fps, number_of_frames = load_video(self.route)
        self.actual_vidcap_count = -1

    def get_altura_base(self):
        return self.img_shape

    def get_len(self):
        return self.lenght

    def get_batch(self, idx_inicial, batch_size=8):
        images = []
        i = 0
        past_img = np.full((*self.img_shape, 3), 255)
        while i < batch_size:
            state, img = self.vidcap.read()
            self.actual_vidcap_count += 1
            if state | (self.actual_vidcap_count < self.lenght):
                if self.images_dict.get(self.actual_vidcap_count, False):
                    img = img if not (img is None) else past_img
                    if not (img is None):
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img)
                    past_img = np.full((*self.img_shape, 3), 255)
                    i += 1
                elif not (img is None):
                    past_img = img.copy()
            else:
                break
        return np.array(images)


source_options_dict = {
    "image_routes": ListRoutesImages,
    "image_folder": FolderRoutesImages,
    "images": ListImages,
    "video": VideoCaptureImages,
}


def Image_Source_Loader(source_type, *args):
    if source_type not in source_options_dict:
        raise NameError(f"{source_type} not implemented on the method")
    return source_options_dict[source_type](*args)
