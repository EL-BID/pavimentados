# import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pavimentados.configs.utils import Config_Basic
from pavimentados.models.yolo import YoloV3

pavimentados_path = Path(__file__).parent.parent


def image_encoder(FILTERS, KERNEL, STRIDE, POOL, USE_BATCH_NORM, USE_DROPOUT, SIAMESE_IMAGE_SIZE):
    inputs = tf.keras.layers.Input(SIAMESE_IMAGE_SIZE)

    x = inputs
    for i in range(len(FILTERS)):
        x = tf.keras.layers.Conv2D(
            filters=FILTERS[i], kernel_size=KERNEL[i], strides=STRIDE[i], padding="same", name="encoder_conv_" + str(i)
        )(x)

        if USE_BATCH_NORM:
            x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=POOL[i], padding="same")(x)

        if USE_DROPOUT:
            x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    output = x
    return tf.keras.Model(inputs, output)


class ComparationLayer(tf.keras.layers.Layer):
    def __init__(self, comparison_matrix, layer_type, **kwargs):
        self.IM2_compress_list = comparison_matrix
        self.layer_type = layer_type
        super(ComparationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x = inputs / tf.keras.backend.sqrt(tf.keras.layers.Dot(1)([inputs, inputs]))
        x_l = []
        for i in range(len(self.IM2_compress_list)):
            x_l.append(tf.keras.backend.dot(x, tf.transpose(self.IM2_compress_list[i])))
        if self.layer_type == 0:
            x = tf.concat(x_l, axis=1)
            x = tf.keras.backend.argmax(x, axis=1)
            out = tf.math.floormod(x, 16)
            return out
        else:
            out = tf.concat([tf.expand_dims(x_l[i], axis=1) for i in range(len(x_l))], axis=1)
            return out

    def get_config(self):
        return {"IM2_compress_c" + str(i + 1): self.IM2_compress_list[i] for i in range(len(self.IM2_compress_list))}


class Pav_Model(Config_Basic):
    def __init__(self, device="/device:CPU:0", config_file=pavimentados_path / "configs" / "models_general.json"):
        self.load_config(config_file)
        self.general_path = pavimentados_path / Path(self.config["general_path"])
        self.model = None
        self.device = device

    def predict(self, data):
        with tf.device(self.device):
            return self.model.predict(data)

    def load_model(self):
        pass


class Yolo_Model(Pav_Model):
    def __init__(
        self,
        device=None,
        config_file=pavimentados_path / "configs" / "yolo_config.json",
        general_config_file=pavimentados_path / "configs" / "models_general.json",
        artifacts_path=None,
    ):
        super().__init__(device, config_file=general_config_file)
        self.general_config = self.config.copy()
        self.load_config(config_file)
        self.config["yolo_pav_dict_clases"] = {int(k): v for k, v in self.config["yolo_pav_dict_clases"].items()}
        if artifacts_path:
            self.yolo_paviment_path = Path(artifacts_path) / self.config["yolo_paviment_path"]
            self.yolo_signal_path = Path(artifacts_path) / self.config["yolo_signal_path"]
        else:
            self.yolo_paviment_path = self.general_path / self.config["yolo_paviment_path"]
            self.yolo_signal_path = self.general_path / self.config["yolo_signal_path"]
        self.load_model()

    def load_model(self):
        """
        Carga los modelos de YOLO.
        """
        with tf.device(self.device):
            # Carga las clases de modelo de pavimentos.
            classes_paviment = list(
                {
                    name: idx for idx, name in enumerate(open(self.yolo_paviment_path / "classes" / "classes.names").read().splitlines())
                }.keys()
            )
            self.num_classes_paviment = len(classes_paviment)

            # Carga las clases de modelo de señales,
            classes_signal = list(
                {name: idx for idx, name in enumerate(open(self.yolo_signal_path / "classes" / "classes.names").read().splitlines())}.keys()
            )
            self.num_classes_signal = len(classes_signal)

            # Une las clases de ambos modelos.
            self.full_classes = [*classes_paviment, *classes_signal]

            # Carga el path de los pesos de yolo de pavimentos y señales.
            path_weights_paviment = self.yolo_paviment_path / "checkpoints_model" / "yolov3_train_8.tf"
            path_weights_signal = self.yolo_signal_path / "checkpoints_model" / "yolov3_train_8.tf"

            # Instancia los modelos.
            yolo_paviment = YoloV3(classes=self.num_classes_paviment, model_name="yolov3_paviment")
            yolo_signal = YoloV3(classes=self.num_classes_signal, model_name="yolov3_signal")

            # Carga los pesos.
            yolo_paviment.load_weights(path_weights_paviment).expect_partial()
            yolo_signal.load_weights(path_weights_signal).expect_partial()

            # Genera modelo final.
            input_model = tf.keras.Input(shape=(416, 416, 3), name="image")
            paviment_output = yolo_paviment(input_model)
            signal_output = yolo_signal(input_model)
            self.model = tf.keras.models.Model(input_model, [paviment_output, signal_output])


class Siamese_Model(Pav_Model):
    def __init__(
        self,
        device=None,
        config_file=pavimentados_path / "configs" / "siamese_config.json",
        general_config_file=pavimentados_path / "configs" / "models_general.json",
        artifacts_path=None,
    ):
        super().__init__(device, config_file=general_config_file)
        self.general_config = self.config.copy()
        self.load_config(config_file)
        if artifacts_path:
            self.siamese_path = Path(artifacts_path) / self.config["siamese_path"]
        else:
            self.siamese_path = pavimentados_path / self.general_path / self.config["siamese_path"]
        self.load_model()

    def load_model(self):
        """
        Carga los modelos de YOLO.
        """
        with tf.device(self.device):
            FIRST_COMPARISON_EXAMPLES_NUMBER = self.config["FIRST_COMPARISON_EXAMPLES_NUMBER"]
            SECOND_COMPARISON_EXAMPLES_NUMBER = self.config["SECOND_COMPARISON_EXAMPLES_NUMBER"]
            FIRST_COMPARISON_FOLDER = self.config["FIRST_COMPARISON_FOLDER"]
            SECOND_COMPARISON_FOLDER = self.config["SECOND_COMPARISON_FOLDER"]

            SIAMESE_IMAGE_SIZE = tuple(self.config["SIAMESE_IMAGE_SIZE"])
            FILTERS = self.config["FILTERS"]
            KERNEL = self.config["KERNEL"]
            STRIDE = self.config["STRIDE"]
            POOL = self.config["POOL"]
            USE_BATCH_NORM = bool(self.config["USE_BATCH_NORM"])
            USE_DROPOUT = bool(self.config["USE_DROPOUT"])
            IM2_TOTAL_C = []

            for i in range(FIRST_COMPARISON_EXAMPLES_NUMBER):
                IM2_TOTAL_C.append([])

            class_names_first = {}
            self.class_names_last = {}
            c = 0
            search_path = self.siamese_path / FIRST_COMPARISON_FOLDER
            for path in tqdm(os.listdir(search_path)):
                class_names_first[path] = c
                self.class_names_last[path] = []
                c += 1
                item_search_path = search_path / path
                x = os.listdir(item_search_path)
                images_files = [item_search_path / i for i in x]
                if len(images_files) >= FIRST_COMPARISON_EXAMPLES_NUMBER:
                    for i in range(FIRST_COMPARISON_EXAMPLES_NUMBER):
                        IM2_TOTAL_C[i].append(
                            cv2.resize(cv2.imread(str(images_files[i])), SIAMESE_IMAGE_SIZE[:2], interpolation=cv2.INTER_AREA).astype(float)
                            / 255
                        )

            for i in range(FIRST_COMPARISON_EXAMPLES_NUMBER):
                IM2_TOTAL_C[i] = np.array(IM2_TOTAL_C[i])
            self.inv_class_names_first = {v: k for k, v in class_names_first.items()}

            dict_clases_sub = joblib.load(self.siamese_path / "dict_senales_clases.pickle")
            inv_dict_clases_sub = {item: k for k, v in dict_clases_sub.items() for item in v}

            IM2_LAST_C = []
            for i in range(SECOND_COMPARISON_EXAMPLES_NUMBER):
                IM2_LAST_C.append([])

            c = 0
            self.class_names_last_complete = {}
            search_path = self.siamese_path / SECOND_COMPARISON_FOLDER
            for path in tqdm(os.listdir(search_path)):
                if inv_dict_clases_sub.get(path, None):
                    self.class_names_last[inv_dict_clases_sub[path]].append(c)
                    self.class_names_last_complete[c] = path
                    c += 1
                    item_search_path = search_path / path
                    x = os.listdir(item_search_path)
                    images_files = [item_search_path / i for i in x]
                    if len(images_files) >= SECOND_COMPARISON_EXAMPLES_NUMBER:
                        for i in range(SECOND_COMPARISON_EXAMPLES_NUMBER):
                            IM2_LAST_C[i].append(
                                cv2.resize(cv2.imread(str(images_files[i])), SIAMESE_IMAGE_SIZE[:2], interpolation=cv2.INTER_AREA).astype(
                                    float
                                )
                                / 255
                            )

            for i in range(SECOND_COMPARISON_EXAMPLES_NUMBER):
                IM2_LAST_C[i] = np.array(IM2_LAST_C[i])

            print("Createing siamese model")
            image_conv_encoder = image_encoder(FILTERS, KERNEL, STRIDE, POOL, USE_BATCH_NORM, USE_DROPOUT, SIAMESE_IMAGE_SIZE)
            image_conv_encoder_last = image_encoder(FILTERS, KERNEL, STRIDE, POOL, USE_BATCH_NORM, USE_DROPOUT, SIAMESE_IMAGE_SIZE)
            print("Loading Weights siamese")
            image_conv_encoder.load_weights(str(self.siamese_path / "image_encoder_weights_first") + "/")
            image_conv_encoder_last.load_weights(str(self.siamese_path / "image_conv_encoder_weights") + "/")
            print("Siamese model first loaded")
            IM2_compress_c = []
            for i in range(FIRST_COMPARISON_EXAMPLES_NUMBER):
                IM2_compress = image_conv_encoder.predict(IM2_TOTAL_C[i])
                IM2_compress_c.append(IM2_compress / tf.keras.backend.sqrt(tf.keras.layers.Dot(1)([IM2_compress, IM2_compress])))
            IM2_compress_last_c = []
            for i in range(SECOND_COMPARISON_EXAMPLES_NUMBER):
                IM2_compress_last = image_conv_encoder.predict(IM2_LAST_C[i])
                IM2_compress_last_c.append(
                    IM2_compress_last / tf.keras.backend.sqrt(tf.keras.layers.Dot(1)([IM2_compress_last, IM2_compress_last]))
                )

            input_model = tf.keras.layers.Input(SIAMESE_IMAGE_SIZE)
            x1 = image_conv_encoder(input_model)
            x2 = image_conv_encoder_last(input_model)
            out1 = ComparationLayer(IM2_compress_c, 0)(x1)
            out2 = ComparationLayer(IM2_compress_last_c, 1)(x2)
            self.model = tf.keras.Model(input_model, [out1, out2])

    def predict(self, data):
        with tf.device(self.device):
            pred = self.model.predict(data)
        prediction_first = list(map(self.inv_class_names_first.get, pred[0]))
        prediction_last_comp_layer = list(map(self.class_names_last.get, prediction_first))
        pred_class_last = list(
            map(
                lambda p_l_c_l, p: self.class_names_last_complete[p_l_c_l[np.argmax(p[:, p_l_c_l]) % len(p_l_c_l)]],
                prediction_last_comp_layer,
                pred[1],
            )
        )
        return pred, prediction_first, pred_class_last


class State_Signal_Model(Pav_Model):
    def __init__(
        self,
        device=None,
        config_file=pavimentados_path / "configs" / "state_signal_config.json",
        general_config_file=pavimentados_path / "configs" / "models_general.json",
        artifacts_path=None,
    ):
        super().__init__(device, config_file=general_config_file)
        self.general_config = self.config.copy()
        self.load_config(config_file)
        if artifacts_path:
            self.state_signal_model_path = Path(artifacts_path) / self.config["state_signal_model_path"]
        else:
            self.state_signal_model_path = pavimentados_path / self.general_path / self.config["state_signal_model_path"]
        self.load_model()

    def load_model(self):
        with tf.device(self.device):
            self.model = tf.keras.models.load_model(self.state_signal_model_path / os.listdir(self.state_signal_model_path)[0])
