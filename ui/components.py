import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QListWidget, QScrollArea, QLabel, QPushButton,
                               QFileDialog, QSplitter)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from core import VideoProcessor, ProjectModel, BatchData


class FrameWidget(QLabel):
    def __init__(self, np_img):
        super().__init__()
        self.setFixedSize(160, 90)
        self.setScaledContents(True)
        self.setStyleSheet("border: 1px solid #555; background: #222;")

        # Конвертация BGR -> RGB для корректного отображения
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(q_img))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = VideoProcessor()
        self.all_frames = []  # Список всех кадров на таймлайне (для экспорта)

        self.setWindowTitle("Batch Video Editor MVP")
        self.resize(1100, 700)
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Сплиттер для списка батчей и превью
        splitter = QSplitter(Qt.Horizontal)

        self.batch_list = QListWidget()
        self.preview = QLabel("Выберите кадр для превью")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: black; color: gray; font-size: 18px;")

        splitter.addWidget(self.batch_list)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(1, 3)

        layout.addWidget(splitter)

        # Таймлайн
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(180)

        self.timeline_container = QWidget()
        self.timeline_layout = QHBoxLayout(self.timeline_container)
        self.timeline_layout.setAlignment(Qt.AlignLeft)
        self.scroll.setWidget(self.timeline_container)

        layout.addWidget(self.scroll)

        # Кнопки
        btns = QHBoxLayout()
        btn_add = QPushButton("Добавить батчи (.jfif)")
        btn_add.clicked.connect(self.load_images)

        btn_export = QPushButton("Экспортировать видео (MP4)")
        btn_export.clicked.connect(self.export_video)
        btn_export.setStyleSheet("background-color: #2b5e2b; color: white; font-weight: bold;")

        btns.addWidget(btn_add)
        btns.addStretch()
        btns.addWidget(btn_export)
        layout.addLayout(btns)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выбор батчей", "", "Images (*.jfif *.jpg *.png)")
        if not files: return

        for f in files:
            frames = self.processor.split_batch(f)
            if not frames: continue

            self.batch_list.addItem(f.split('/')[-1])

            for img in frames:
                self.all_frames.append(img)
                fw = FrameWidget(img)
                # Добавляем обработку клика для превью
                fw.mousePressEvent = lambda e, i=img: self.set_preview(i)
                self.timeline_layout.addWidget(fw)

    def set_preview(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        # Масштабируем под размер окна превью
        pixmap = QPixmap.fromImage(q_img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    def export_video(self):
        if not self.all_frames: return

        save_path, _ = QFileDialog.getSaveFileName(self, "Сохранить видео", "output_film.mp4", "Video (*.mp4)")
        if not save_path: return

        h, w, _ = self.all_frames[0].shape

        new_fps = 32.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, new_fps, (w, h))

        prev_frame = None

        print("Запуск рендеринга с интерполяцией FILM...")

        for curr_frame in self.all_frames:
            if prev_frame is not None:
                # Генерируем промежуточный кадр через FILM
                mid_frame = self.processor.interpolate_film(prev_frame, curr_frame)
                out.write(mid_frame)

            # Пишем основной кадр
            out.write(curr_frame)
            prev_frame = curr_frame.copy()

        out.release()
        print("Экспорт завершен успешно.")
