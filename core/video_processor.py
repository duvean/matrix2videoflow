import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import onnxruntime
from PySide6.QtCore import QObject


class LamaInpainter:
    def __init__(self, model_path=None):
        models_dir = Path(__file__).resolve().parent / "models"
        default_path = models_dir / "lama_fp32.onnx"
        legacy_path = Path(__file__).resolve().parent / "lama_fp32.onnx"
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = default_path if default_path.exists() else legacy_path
        self.session = None

    def _ensure_session(self):
        if self.session is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"LaMa model not found: {self.model_path}")
        self.session = onnxruntime.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

    def inpaint(self, image, mask):
        self._ensure_session()

        if image is None or mask is None:
            return image
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if np.max(mask) == 0:
            return image.copy()

        rows, cols = np.where(mask > 0)
        y1, y2 = np.min(rows), np.max(rows)
        x1, x2 = np.min(cols), np.max(cols)

        pad = 200
        y1 = max(0, y1 - pad)
        y2 = min(image.shape[0], y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(image.shape[1], x2 + pad)

        crop_img = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]
        crop_h, crop_w = crop_img.shape[:2]

        target_size = (512, 512)
        img_resized = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(crop_mask, target_size, interpolation=cv2.INTER_NEAREST)

        img_onnx = img_resized.astype(np.float32) / 255.0
        img_onnx = img_onnx.transpose(2, 0, 1)[None, ...]

        mask_onnx = (mask_resized.astype(np.float32) / 255.0 > 0).astype(np.float32)
        mask_onnx = mask_onnx[None, None, ...]

        outputs = self.session.run(None, {"image": img_onnx, "mask": mask_onnx})
        output = outputs[0][0].transpose(1, 2, 0)

        if output.max() > 2.0:
            output = output.clip(0, 255).astype(np.uint8)
        else:
            output = (output * 255.0).clip(0, 255).astype(np.uint8)

        output_resized = cv2.resize(output, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
        result = image.copy()
        result[y1:y2, x1:x2] = output_resized
        return result


class VideoProcessor(QObject):
    def __init__(self):
        super().__init__()
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.model = None
        self.load_model()
        self.inpainter = LamaInpainter()

    def _resolve_local_film_path(self):
        return (Path(__file__).resolve().parent / "models")

    def load_model(self):
        try:
            local_film = self._resolve_local_film_path()
            if local_film is None:
                raise FileNotFoundError("Local FILM archive was not found in core/models")
            self.model = hub.load(str(local_film))
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
        current_frames = frames
        for step in pipeline:
            current_frames = self.apply_step(current_frames, step)
        return current_frames

    def process_sequence_fast_cv(self, frames, interpolation_depth):
        current_frames = frames
        for _ in range(interpolation_depth):
            new_sequence = []
            for i in range(len(current_frames) - 1):
                f1 = current_frames[i]
                f2 = current_frames[i + 1]
                mid = self.interpolate_opencv(f1, f2)
                new_sequence.append(f1)
                new_sequence.append(mid)
            new_sequence.append(current_frames[-1])
            current_frames = new_sequence
        return current_frames

    def apply_step(self, current_frames, step):
        if not current_frames:
            return []
        if step in ('film', 'cv'):
            new_sequence = []
            for i in range(len(current_frames) - 1):
                f1 = current_frames[i]
                f2 = current_frames[i + 1]
                mid = self.interpolate_film(f1, f2) if step == 'film' else self.interpolate_opencv(f1, f2)
                new_sequence.append(f1)
                new_sequence.append(mid)
            new_sequence.append(current_frames[-1])
            return new_sequence
        if step == "pixel_sort":
            total = max(1, len(current_frames) - 1)
            return [self.pixel_sort_frame(frame, i / total) for i, frame in enumerate(current_frames)]
        return current_frames

    def process_sequence_with_graph(self, frames, graph_manager):
        steps = graph_manager.list_processing_steps()
        current_frames = frames
        for step in steps:
            current_frames = self.apply_step(current_frames, step)
        return current_frames

    def pixel_sort_frame(self, frame, time_norm=0.0, threshold=0.45, direction="horizontal", strength=0.8):
        if frame is None:
            return frame
        out = frame.copy()
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        animated_threshold = float(np.clip(threshold + 0.2 * np.sin(time_norm * np.pi * 2.0), 0.05, 0.95))
        trigger = gray > animated_threshold
        if direction not in ("horizontal", "vertical"):
            direction = "horizontal"

        if direction == "vertical":
            out = np.transpose(out, (1, 0, 2))
            trigger = trigger.T

        for r in range(out.shape[0]):
            idx = np.where(trigger[r])[0]
            if idx.size < 2:
                continue
            row = out[r, idx, :]
            brightness = np.sum(row.astype(np.int32), axis=1)
            order = np.argsort(brightness)
            blend = float(np.clip(strength, 0.0, 1.0))
            sorted_row = row[order]
            out[r, idx, :] = (row * (1.0 - blend) + sorted_row * blend).astype(np.uint8)

        if direction == "vertical":
            out = np.transpose(out, (1, 0, 2))
        return out

    def split_batch(self, image_path, grid_size=3):
        img = cv2.imread(image_path)
        if img is None:
            return []
        h, w, _ = img.shape
        fh, fw = h // grid_size, w // grid_size
        frames = []
        for r in range(grid_size):
            for c in range(grid_size):
                frames.append(img[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw].copy())
        return frames

    def inpaint_project_frames(self, batches, masks, progress_cb=None):
        total = sum(len(m['frame_ids']) for m in masks)
        if total == 0:
            return 0

        frame_lookup = {}
        for b_idx, batch in enumerate(batches):
            if not hasattr(batch, 'frame_ids'):
                continue
            for f_idx, frame_id in enumerate(batch.frame_ids):
                frame_lookup[frame_id] = (b_idx, f_idx)

        processed = 0
        for mask_item in masks:
            mask = mask_item.get('mask')
            frame_ids = mask_item.get('frame_ids', set())
            for frame_id in frame_ids:
                loc = frame_lookup.get(frame_id)
                if loc is None:
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue
                b_idx, f_idx = loc
                img = batches[b_idx].frames[f_idx]

                resized_mask = mask
                if mask is None:
                    processed += 1
                    if progress_cb:
                        progress_cb(processed, total)
                    continue
                if mask.shape[:2] != img.shape[:2]:
                    resized_mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                batches[b_idx].frames[f_idx] = self.inpainter.inpaint(img, resized_mask)
                processed += 1
                if progress_cb:
                    progress_cb(processed, total)

        return processed
