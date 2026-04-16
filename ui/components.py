import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QListWidget, QScrollArea, QLabel, QPushButton,
                               QFileDialog, QSplitter, QGroupBox, QFormLayout, QSpinBox)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from pathlib import Path
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


class ExportSettings(QGroupBox):
    def __init__(self):
        super().__init__("Настройки экспорта")
        self.layout = QFormLayout(self)

        # Настройка FPS
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(32)
        self.layout.addRow("Целевой FPS:", self.fps_spin)

        # Список шагов интерполяции
        self.interp_steps = QListWidget()
        self.layout.addRow("Цепочка интерполяции:", self.interp_steps)

        # Кнопки управления шагами
        btn_layer = QHBoxLayout()
        self.add_film_btn = QPushButton("+ FILM")
        self.add_cv_btn = QPushButton("+ OpenCV")
        self.clear_btn = QPushButton("Сброс")

        btn_layer.addWidget(self.add_film_btn)
        btn_layer.addWidget(self.add_cv_btn)
        btn_layer.addWidget(self.clear_btn)
        self.layout.addRow(btn_layer)


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
        main_layout = QVBoxLayout(central)

        # 1. ГЛАВНЫЙ СПЛИТТЕР (Горизонтальный: Список | Превью | Настройки)
        self.main_splitter = QSplitter(Qt.Horizontal)

        # Левая часть: Список батчей
        self.batch_list = QListWidget()

        # Центральная часть: Превью
        self.preview = QLabel("Выберите кадр для превью")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background: black; color: gray; font-size: 18px; border: 2px solid #333;")
        self.preview.setMinimumWidth(400)

        # Правая часть: Настройки (создаем отдельный виджет-контейнер)
        self.settings_panel = QGroupBox("Настройки экспорта")
        settings_layout = QFormLayout(self.settings_panel)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(32)
        settings_layout.addRow("FPS:", self.fps_spin)

        self.interp_steps = QListWidget()
        self.interp_steps.setMaximumHeight(150)
        settings_layout.addRow("Цепочка:", self.interp_steps)

        add_btns = QHBoxLayout()
        self.btn_add_film = QPushButton("+ FILM")
        self.btn_add_cv = QPushButton("+ CV")
        self.btn_clear_steps = QPushButton("Clear")
        add_btns.addWidget(self.btn_add_film)
        add_btns.addWidget(self.btn_add_cv)
        add_btns.addWidget(self.btn_clear_steps)
        settings_layout.addRow(add_btns)

        # Добавляем всё в сплиттер
        self.main_splitter.addWidget(self.batch_list)
        self.main_splitter.addWidget(self.preview)
        self.main_splitter.addWidget(self.settings_panel)

        # Настраиваем пропорции (0 - батчи, 1 - превью, 2 - настройки)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 1)

        # 2. ТАЙМЛАЙН (Нижняя часть)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(180)
        self.timeline_container = QWidget()
        self.timeline_layout = QHBoxLayout(self.timeline_container)
        self.timeline_layout.setAlignment(Qt.AlignLeft)
        self.scroll.setWidget(self.timeline_container)

        # 3. КНОПКИ ДЕЙСТВИЙ (Самый низ)
        action_btns = QHBoxLayout()
        btn_load = QPushButton("Добавить батчи (.jfif)")
        btn_load.clicked.connect(self.load_images)

        self.btn_export = QPushButton("Экспортировать видео")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setStyleSheet("background-color: #2b5e2b; color: white; padding: 5px;")

        action_btns.addWidget(btn_load)
        action_btns.addStretch()
        action_btns.addWidget(self.btn_export)

        # Собираем всё в главный вертикальный слой
        main_layout.addWidget(self.main_splitter, stretch=1)  # Сплиттер занимает всё свободное место
        main_layout.addWidget(self.scroll)
        main_layout.addLayout(action_btns)

        # Подключаем кнопки настроек
        self.btn_add_film.clicked.connect(lambda: self.interp_steps.addItem("film"))
        self.btn_add_cv.clicked.connect(lambda: self.interp_steps.addItem("opencv"))
        self.btn_clear_steps.clicked.connect(self.interp_steps.clear)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выбор батчей", "", "Images (*.jfif *.jpg *.png)")
        if not files: return

        for f in files:
            frames = self.processor.split_batch(f)
            if not frames: continue

            self.batch_list.addItem(Path(f).name)

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

        # Читаем настройки из UI
        pipeline = [self.interp_steps.item(i).text() for i in range(self.interp_steps.count())]
        target_fps = self.fps_spin.value()
        save_path, _ = QFileDialog.getSaveFileName(self, "Экспорт", "output.mp4", "Video (*.mp4)")

        if not save_path: return

        # 1. Сначала подготавливаем все кадры (включая интерполяцию)
        print(f"Обработка интерполяции: {pipeline}...")
        final_frames = self.processor.process_sequence(self.all_frames, pipeline)

        # 2. Записываем в файл
        h, w, _ = final_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, float(target_fps), (w, h))

        for f in final_frames:
            out.write(f)

        out.release()
        print(f"Готово! Итого кадров: {len(final_frames)}")
