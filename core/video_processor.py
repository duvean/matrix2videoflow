import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PySide6.QtCore import QObject, Signal, QThread


class VideoProcessor(QObject):
    def __init__(self):
        super().__init__()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = hub.load("https://tfhub.dev/google/film/1")
            self.film_available = True
        except Exception as e:
            print(f"FILM error: {e}")
            self.film_available = False

    def interpolate_film(self, frame1, frame2):
        if not self.film_available:
            return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

        img1 = tf.convert_to_tensor(frame1, dtype=tf.float32) / 255.0
        img2 = tf.convert_to_tensor(frame2, dtype=tf.float32) / 255.0

        inputs = {
            'x0': img1[tf.newaxis, ...],
            'x1': img2[tf.newaxis, ...],
            'time': tf.constant([[0.5]], dtype=tf.float32)
        }

        result = self.model(inputs, training=False)
        inter_frame = result['image'].numpy()[0]
        return (np.clip(inter_frame * 255.0, 0, 255)).astype(np.uint8)

    def split_batch(self, image_path, grid_size=4):
        img = cv2.imread(image_path)
        if img is None: return []
        h, w, _ = img.shape
        fh, fw = h // grid_size, w // grid_size
        frames = []
        for r in range(grid_size):
            for c in range(grid_size):
                frames.append(img[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw].copy())
        return frames
