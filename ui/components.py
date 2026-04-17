import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QListWidget, QScrollArea, QLabel, QPushButton,
                               QFileDialog, QSplitter, QGroupBox, QFormLayout, QSpinBox, QToolButton, QStyle, QFrame,
                               QAbstractItemView, QSizePolicy)
from PySide6.QtGui import QImage, QPixmap, QDrag
from PySide6.QtCore import Qt, QMimeData, QEvent
from pathlib import Path
from core import VideoProcessor, ProjectManager, Batch


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


class TimelineFrame(QFrame):
    def __init__(self, np_img, original_id, batch_idx, frame_idx, parent_ui):
        super().__init__()
        # Возвращаем атрибуты для DND
        self.batch_idx = batch_idx
        self.frame_idx = frame_idx
        self.parent_ui = parent_ui
        self.np_img = np_img
        self.original_id = original_id

        self.setFixedSize(160, 200)  # Увеличили высоту под все метки
        self.setStyleSheet("QFrame { border: 1px solid #444; background: #2b2b2b; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Изображение (Превью кадра)
        self.img_label = QLabel()
        self.img_label.setFixedSize(150, 85)  # Фиксируем размер, чтобы не росло окно
        self.img_label.setScaledContents(True)
        self.update_pixmap(np_img)
        layout.addWidget(self.img_label)

        # Метка ID
        self.lbl_id = QLabel(f"ID: {original_id}")
        self.lbl_id.setStyleSheet("color: #777; font-size: 9px; border: none;")
        layout.addWidget(self.lbl_id)

        # Метка Времени
        self.lbl_time = QLabel("00:00:000")
        self.lbl_time.setStyleSheet("color: #00ff00; font-family: 'Courier New'; font-size: 10px; border: none;")
        layout.addWidget(self.lbl_time)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.btn_del = self._create_btn(QStyle.SP_TrashIcon, self.on_delete)
        self.btn_save = self._create_btn(QStyle.SP_DialogSaveButton, self.on_save)
        self.btn_replace = self._create_btn(QStyle.SP_BrowserReload, self.on_replace)

        btn_layout.addWidget(self.btn_del)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_replace)
        layout.addLayout(btn_layout)

    def set_time(self, frame_index, fps):
        if fps <= 0: fps = 1
        total_ms = int((frame_index / fps) * 1000)
        minutes = (total_ms // 60000)
        seconds = (total_ms % 60000) // 1000
        ms = total_ms % 1000
        self.lbl_time.setText(f"TIME: {minutes:02}:{seconds:02}:{ms:03}")

    def _create_btn(self, icon_style, slot):
        btn = QToolButton()
        btn.setFixedSize(22, 22)
        btn.setIcon(self.style().standardIcon(icon_style))
        btn.clicked.connect(slot)
        return btn

    def update_pixmap(self, np_img):
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

    def on_delete(self):
        self.parent_ui.delete_frame(self.batch_idx, self.frame_idx)

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save", "", "PNG (*.png)")
        if path: cv2.imwrite(path, self.np_img)

    def on_replace(self):
        path, _ = QFileDialog.getOpenFileName(None, "Replace", "", "Images (*.jpg *.png *.jfif)")
        if path:
            img = cv2.imread(path)
            self.parent_ui.replace_frame(self.batch_idx, self.frame_idx, img)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(f"{self.batch_idx}:{self.frame_idx}")
            drag.setMimeData(mime)
            drag.exec_(Qt.MoveAction)

    # --- DRAG AND DROP ---
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            # Передаем координаты кадра в данных
            mime.setText(f"{self.batch_idx}:{self.frame_idx}")
            drag.setMimeData(mime)
            drag.exec_(Qt.MoveAction)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = VideoProcessor()
        self.project = ProjectManager()
        self.all_frames = []  # Список всех кадров на таймлайне (для экспорта)

        # Флаги отображения
        self.show_batch_names = True
        self.show_frame_ids = True
        self.show_timestamps = True

        self.setWindowTitle("Batch Video Editor")
        self.resize(1300, 850)
        self.init_ui()

        self.batch_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.batch_list.model().rowsMoved.connect(self.on_batches_reordered)

        # Разрешаем Drop на контейнер таймлайна
        self.timeline_container.setAcceptDrops(True)
        self.timeline_container.dragEnterEvent = self.t_dragEnter
        self.timeline_container.dropEvent = self.t_drop

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
        self.fps_spin.valueChanged.connect(self.refresh_timeline)
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

        # Кнопки-переключатели рядом с "Добавить"
        self.tgl_batch = QToolButton()
        self.tgl_batch.setIcon(self.style().standardIcon(QStyle.SP_FileDialogContentsView))
        self.tgl_batch.setCheckable(True)
        self.tgl_batch.setChecked(True)
        self.tgl_batch.setToolTip("Имена батчей")
        self.tgl_batch.clicked.connect(self.toggle_visibility)

        self.tgl_id = QToolButton()
        self.tgl_id.setIcon(self.style().standardIcon(QStyle.SP_FileDialogListView))
        self.tgl_id.setCheckable(True)
        self.tgl_id.setChecked(True)
        self.tgl_id.setToolTip("ID кадров")
        self.tgl_id.clicked.connect(self.toggle_visibility)

        self.tgl_time = QToolButton()
        self.tgl_time.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.tgl_time.setCheckable(True)
        self.tgl_time.setChecked(True)
        self.tgl_time.setToolTip("Время")
        self.tgl_time.clicked.connect(self.toggle_visibility)

        # Добавляем их в action_btns (там где кнопка Добавить)
        action_btns.insertWidget(1, self.tgl_batch)
        action_btns.insertWidget(2, self.tgl_id)
        action_btns.insertWidget(3, self.tgl_time)

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

        # Фиксируем превью, чтобы оно не раздувалось
        self.preview.setMinimumSize(640, 360)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Таймлайн: Увеличиваем высоту и убираем вертикальный скролл
        self.scroll.setFixedHeight(280)  # Больше места для подписей и скобок
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Установка фильтра событий для прокрутки колесиком мыши
        self.scroll.viewport().installEventFilter(self)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выбор батчей", "", "Images (*.jfif *.jpg *.png)")
        if not files: return

        for f in files:
            frames = self.processor.split_batch(f)
            if not frames: continue

            # Создаем объект батча и сохраняем в проект
            new_batch = Batch(Path(f).name, frames)
            self.project.batches.append(new_batch)

            # Добавляем только имя в визуальный список
            self.batch_list.addItem(new_batch.name)

        # После загрузки всех файлов обновляем таймлайн
        self.refresh_timeline()

    def set_preview(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        q_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format_RGB888)
        # Масштабируем под размер окна превью
        pixmap = QPixmap.fromImage(q_img).scaled(self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pixmap)

    def toggle_visibility(self):
        self.show_batch_names = self.tgl_batch.isChecked()
        self.show_frame_ids = self.tgl_id.isChecked()
        self.show_timestamps = self.tgl_time.isChecked()
        self.refresh_timeline()

    def on_batches_reordered(self, parent, start, end, destination, row):
        # Извлекаем индексы как целые числа
        start_idx = start
        dest_idx = row

        # В некоторых версиях PySide/Qt индексы приходят как QModelIndex
        if hasattr(start, 'row'): start_idx = start.row()
        if hasattr(row, 'row'): dest_idx = row  # индекс строки назначения

        if not self.project.batches:
            return

        # Логика корректного перемещения в Python-списке
        # Если мы тянем элемент сверху вниз, индекс вставки смещается
        moved_item = self.project.batches.pop(start_idx)

        # Если мы вставляем ниже по списку, индекс уже уменьшился на 1 после pop
        actual_insert_idx = dest_idx
        if start_idx < dest_idx:
            actual_insert_idx -= 1

        self.project.batches.insert(actual_insert_idx, moved_item)

        # Полная перерисовка таймлайна
        self.refresh_timeline()

    def eventFilter(self, source, event):
        # Реализация горизонтальной прокрутки колесиком
        if event.type() == QEvent.Wheel and source is self.scroll.viewport():
            delta = event.angleDelta().y()
            self.scroll.horizontalScrollBar().setValue(
                self.scroll.horizontalScrollBar().value() - delta
            )
            return True
        return super().eventFilter(source, event)

    def refresh_timeline(self):
        while self.timeline_layout.count():
            item = self.timeline_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        fps = self.fps_spin.value()
        interp_count = self.interp_steps.count()
        # Каждая интерполяция удваивает количество кадров
        frame_multiplier = 2 ** interp_count

        global_idx = 0
        for b_idx, batch in enumerate(self.project.batches):
            # Контейнер батча
            batch_widget = QWidget()
            batch_v_layout = QVBoxLayout(batch_widget)

            frames_h_layout = QHBoxLayout()
            for f_idx, img in enumerate(batch.frames):
                # Гарантируем наличие ID
                if not hasattr(batch, 'frame_ids'): batch.frame_ids = []
                while len(batch.frame_ids) <= f_idx:
                    batch.frame_ids.append(f"{f_idx}_{batch.name}")

                fw = TimelineFrame(img, batch.frame_ids[f_idx], b_idx, f_idx, self)

                # Учет интерполяции в таймстампе
                # t = (GlobalIndex * Multiplier) / FPS
                fw.set_time(global_idx * frame_multiplier, fps)

                fw.lbl_id.setVisible(self.show_frame_ids)
                fw.lbl_time.setVisible(self.show_timestamps)
                fw.img_label.mousePressEvent = lambda e, i=img: self.set_preview(i)

                frames_h_layout.addWidget(fw)
                global_idx += 1

            batch_v_layout.addLayout(frames_h_layout)

            # Квадратная скобка
            bracket = QFrame()
            bracket.setFixedHeight(8)
            bracket.setStyleSheet("border: 2px solid #555; border-top: none; margin-top: -2px;")
            batch_v_layout.addWidget(bracket)

            if self.show_batch_names:
                lbl = QLabel(batch.name)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet("color: #5cacee; font-weight: bold;")
                batch_v_layout.addWidget(lbl)

            self.timeline_layout.addWidget(batch_widget)

    # --- Логика изменения кадров ---
    def delete_frame(self, b_idx, f_idx):
        # Удаляем кадр из данных
        if b_idx < len(self.project.batches):
            batch = self.project.batches[b_idx]
            if f_idx < len(batch.frames):
                batch.frames.pop(f_idx)

                # Опционально: если батч стал пустым, можно его удалить
                if not batch.frames:
                    self.project.batches.pop(b_idx)
                    self.batch_list.takeItem(b_idx)

        self.refresh_timeline()

    def replace_frame(self, b_idx, f_idx, new_img):
        self.project.batches[b_idx].frames[f_idx] = new_img
        self.refresh_timeline()

    # --- Drag & Drop кадров на таймлайне ---
    def t_dragEnter(self, e):
        e.accept()

    def t_drop(self, e):
        try:
            data = e.mimeData().text().split(':')
            src_b_idx, src_f_idx = int(data[0]), int(data[1])
        except:
            return

        # Определяем положение относительно вьюпорта скролла
        pos = e.position().toPoint()

        # Находим виджет, на который упал кадр
        target = self.timeline_container.childAt(pos)

        # Идем вверх по иерархии, пока не найдем наш TimelineFrame
        while target and not isinstance(target, TimelineFrame):
            target = target.parentWidget()

        # Данные для перемещения (Картинка + ID)
        moving_frame = self.project.batches[src_b_idx].frames.pop(src_f_idx)
        moving_id = self.project.batches[src_b_idx].frame_ids.pop(src_f_idx)

        if target:
            dst_b_idx = target.batch_idx
            dst_f_idx = target.frame_idx
            self.project.batches[dst_b_idx].frames.insert(dst_f_idx, moving_frame)
            self.project.batches[dst_b_idx].frame_ids.insert(dst_f_idx, moving_id)
        else:
            # В конец последнего батча
            self.project.batches[-1].frames.append(moving_frame)
            self.project.batches[-1].frame_ids.append(moving_id)

        self.refresh_timeline()

    def export_video(self):
        # 0. Собираем актуальный плоский список кадров из батчей
        current_frames = []
        for b in self.project.batches:
            current_frames.extend(b.frames)

        if not current_frames:
            print("Нет кадров для экспорта")
            return

        pipeline = [self.interp_steps.item(i).text() for i in range(self.interp_steps.count())]
        target_fps = self.fps_spin.value()
        save_path, _ = QFileDialog.getSaveFileName(self, "Экспорт", "output.mp4", "Video (*.mp4)")

        if not save_path: return

        # 1. Обработка (процессор должен принимать массив и список шагов)
        print(f"Запуск конвейера интерполяции: {pipeline}")
        final_frames = self.processor.process_sequence(current_frames, pipeline)

        # 2. Запись
        h, w, _ = final_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, float(target_fps), (w, h))

        for f in final_frames:
            out.write(f)

        out.release()
        print(f"Экспорт завершен. Итого кадров: {len(final_frames)}")
