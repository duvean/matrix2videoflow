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

    def interpolate_opencv(self, f1, f2):
        return cv2.addWeighted(f1, 0.5, f2, 0.5, 0)

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

    def process_sequence(self, frames, pipeline):
        """
        Применяет цепочку интерполяций ко всему массиву кадров.
        pipeline: список строк ['film', 'opencv']
        """
        current_frames = frames

        for step in pipeline:
            new_sequence = []
            for i in range(len(current_frames) - 1):
                f1 = current_frames[i]
                f2 = current_frames[i + 1]

                # Генерируем промежуточный
                if step == 'film':
                    mid = self.interpolate_film(f1, f2)
                else:
                    mid = self.interpolate_opencv(f1, f2)

                new_sequence.append(f1)
                new_sequence.append(mid)

            # Не забываем добавить самый последний кадр
            new_sequence.append(current_frames[-1])
            current_frames = new_sequence

        return current_frames

    def split_batch(self, image_path, grid_size=3):
        img = cv2.imread(image_path)
        if img is None: return []
        h, w, _ = img.shape
        fh, fw = h // grid_size, w // grid_size
        frames = []
        for r in range(grid_size):
            for c in range(grid_size):
                frames.append(img[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw].copy())
        return frames
