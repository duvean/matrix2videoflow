import cv2
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                               QListWidget, QScrollArea, QLabel, QPushButton,
                               QFileDialog, QSplitter, QGroupBox, QFormLayout, QSpinBox, QToolButton, QStyle, QFrame,
                               QAbstractItemView)
from PySide6.QtGui import QImage, QPixmap, QDrag
from PySide6.QtCore import Qt, QMimeData
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
    def __init__(self, np_img, batch_idx, frame_idx, parent_ui):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.batch_idx = batch_idx
        self.frame_idx = frame_idx
        self.parent_ui = parent_ui
        self.np_img = np_img

        self.setFixedSize(160, 120)
        self.setFrameStyle(QFrame.Panel | QFrame.Plain)
        self.setLineWidth(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        # Иконка кадра
        self.img_label = QLabel()
        self.img_label.setScaledContents(True)
        self.update_pixmap(np_img)
        layout.addWidget(self.img_label)

        # Панель кнопок (Overlay-стиль или снизу)
        btn_layout = QHBoxLayout()

        self.btn_del = QToolButton()
        self.btn_del.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        self.btn_del.clicked.connect(self.on_delete)

        self.btn_save = QToolButton()
        self.btn_save.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.btn_save.clicked.connect(self.on_save)

        self.btn_replace = QToolButton()
        self.btn_replace.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.btn_replace.clicked.connect(self.on_replace)

        for b in [self.btn_del, self.btn_save, self.btn_replace]:
            b.setFixedSize(24, 24)
            btn_layout.addWidget(b)

        layout.addLayout(btn_layout)

    def update_pixmap(self, np_img):
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(qimg))

    # Логика кнопок
    def on_delete(self):
        self.parent_ui.delete_frame(self.batch_idx, self.frame_idx)

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(None, "Сохранить кадр", "", "Images (*.png *.jpg)")
        if path: cv2.imwrite(path, self.np_img)

    def on_replace(self):
        path, _ = QFileDialog.getOpenFileName(None, "Заменить кадр", "", "Images (*.jfif *.jpg *.png)")
        if path:
            new_img = cv2.imread(path)
            self.parent_ui.replace_frame(self.batch_idx, self.frame_idx, new_img)

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

        self.setWindowTitle("Batch Video Editor MVP")
        self.resize(1100, 700)
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

    def refresh_timeline(self):
        # Очистка
        while self.timeline_layout.count():
            item = self.timeline_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        # Отрисовка
        for b_idx, batch in enumerate(self.project.batches):
            for f_idx, img in enumerate(batch.frames):
                fw = TimelineFrame(img, b_idx, f_idx, self)
                # Важно: восстанавливаем привязку превью, которую мы потеряли
                fw.img_label.mousePressEvent = lambda e, i=img: self.set_preview(i)
                self.timeline_layout.addWidget(fw)

            # Добавим небольшой отступ между батчами для наглядности
            self.timeline_layout.addSpacing(15)

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

        # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ищем виджет внутри timeline_container, а не всего окна
        pos = e.position().toPoint()
        target_widget = self.timeline_container.childAt(pos)

        while target_widget and not isinstance(target_widget, TimelineFrame):
            target_widget = target_widget.parentWidget()

        # Удаляем кадр из старого места
        frame_to_move = self.project.batches[src_b_idx].frames.pop(src_f_idx)

        if target_widget:
            dst_b_idx = target_widget.batch_idx
            dst_f_idx = target_widget.frame_idx

            # Если перемещаем внутри одного и того же батча и тянем "вправо",
            # индекс смещается из-за pop(). Нужна корректировка:
            if src_b_idx == dst_b_idx and src_f_idx < dst_f_idx:
                # Мы не корректируем здесь, так как insert вставит ПЕРЕД целевым кадром,
                # что логично для "вставки между". Но если хочешь вставить ПОСЛЕ,
                # можно добавить логику проверки половины ширины виджета.
                pass

            self.project.batches[dst_b_idx].frames.insert(dst_f_idx, frame_to_move)
        else:
            # Если не попали в виджет, кидаем в конец последнего батча
            if self.project.batches:
                self.project.batches[-1].frames.append(frame_to_move)

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
