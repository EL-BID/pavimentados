# import cv2
import tensorflow as tf


def transform_images(images, size):
    """
    Transforma las imagenes que seran introducidas en el modelo Yolov3
    """
    images = tf.image.resize(images, (size, size))
    images = images / 255.0
    return images
