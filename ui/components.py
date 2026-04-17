import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QScrollArea, QLabel, QPushButton,
    QFileDialog, QSplitter, QGroupBox, QFormLayout, QSpinBox, QToolButton, QStyle, QFrame,
    QAbstractItemView, QSizePolicy, QStackedWidget, QMenu, QListWidgetItem, QProgressBar, QMessageBox
)
from PySide6.QtGui import QImage, QPixmap, QDrag, QAction, QPainter, QPen, QColor
from PySide6.QtCore import Qt, QMimeData, QEvent, QTimer, QPoint

from core import VideoProcessor, ProjectManager, Batch


class MaskPreviewLabel(QLabel):
    def __init__(self, parent_ui):
        super().__init__("Выберите кадр для превью")
        self.parent_ui = parent_ui
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: black; color: gray; font-size: 18px; border: 2px solid #333;")
        self.setMinimumWidth(400)
        self.setMouseTracking(True)

        self.base_img = None
        self.mask = None
        self.edit_mode = False
        self.brush_size = 20
        self.cursor_pos = QPoint(-1, -1)

    def set_content(self, img, mask=None, show_overlay=True):
        self.base_img = img.copy() if img is not None else None
        self.mask = None if mask is None else mask.copy()
        self._render(show_overlay)

    def _render(self, show_overlay=True):
        if self.base_img is None:
            self.setPixmap(QPixmap())
            self.setText("Выберите кадр для превью")
            return

        rgb_img = cv2.cvtColor(self.base_img, cv2.COLOR_BGR2RGB)
        render = rgb_img.copy()

        if show_overlay and self.mask is not None and np.max(self.mask) > 0:
            if self.mask.shape[:2] != render.shape[:2]:
                mask = cv2.resize(self.mask, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask = self.mask
            overlay = render.copy()
            overlay[mask > 0] = [255, 0, 0]
            render = cv2.addWeighted(render, 0.75, overlay, 0.25, 0)

        h, w, ch = render.shape
        q_img = QImage(render.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pixmap)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render(self.parent_ui.should_show_mask_overlay())

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.edit_mode or self.base_img is None or self.cursor_pos.x() < 0:
            return

        mapped = self._map_label_to_image(self.cursor_pos)
        if mapped is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        _, _, scale = mapped
        radius = max(1, int(self.brush_size * scale))
        painter.drawEllipse(self.cursor_pos, radius, radius)
        painter.end()

    def _map_label_to_image(self, pos):
        if self.base_img is None:
            return None
        label_w, label_h = self.width(), self.height()
        img_h, img_w = self.base_img.shape[:2]

        scale = min(label_w / img_w, label_h / img_h)
        draw_w, draw_h = int(img_w * scale), int(img_h * scale)
        x_off = (label_w - draw_w) // 2
        y_off = (label_h - draw_h) // 2

        x = pos.x() - x_off
        y = pos.y() - y_off
        if x < 0 or y < 0 or x >= draw_w or y >= draw_h:
            return None

        ix = int(x / scale)
        iy = int(y / scale)
        return ix, iy, scale

    def _draw_to_mask(self, pos, erase=False):
        if not self.edit_mode or self.base_img is None or self.mask is None:
            return

        mapped = self._map_label_to_image(pos)
        if mapped is None:
            return

        ix, iy, _ = mapped
        value = 0 if erase else 255
        cv2.circle(self.mask, (ix, iy), max(1, int(self.brush_size)), value, -1)
        self._render(True)

    def mousePressEvent(self, event):
        self.cursor_pos = event.position().toPoint()
        if self.edit_mode:
            if event.button() == Qt.LeftButton:
                self._draw_to_mask(self.cursor_pos, erase=False)
                return
            if event.button() == Qt.RightButton:
                self._draw_to_mask(self.cursor_pos, erase=True)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.cursor_pos = event.position().toPoint()
        if self.edit_mode:
            if event.buttons() & Qt.LeftButton:
                self._draw_to_mask(self.cursor_pos, erase=False)
                return
            if event.buttons() & Qt.RightButton:
                self._draw_to_mask(self.cursor_pos, erase=True)
                return
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.cursor_pos = QPoint(-1, -1)
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self.edit_mode:
            delta = event.angleDelta().y()
            if delta > 0:
                self.brush_size = min(200, self.brush_size + 2)
            elif delta < 0:
                self.brush_size = max(1, self.brush_size - 2)
            self.update()
            return
        super().wheelEvent(event)


class TimelineFrame(QFrame):
    def __init__(self, np_img, original_id, batch_idx, frame_idx, parent_ui):
        super().__init__()
        self.batch_idx = batch_idx
        self.frame_idx = frame_idx
        self.parent_ui = parent_ui
        self.np_img = np_img
        self.original_id = original_id
        self.mask_highlighted = False
        self.playback_highlighted = False

        self.setFixedSize(160, 220)
        self.apply_frame_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.addStretch()
        self.mask_toggle_btn = QToolButton()
        self.mask_toggle_btn.setFixedSize(22, 22)
        self.mask_toggle_btn.clicked.connect(self.toggle_frame_in_mask)
        self.mask_toggle_btn.setVisible(False)
        top.addWidget(self.mask_toggle_btn)
        layout.addLayout(top)

        self.img_label = QLabel()
        self.img_label.setFixedSize(150, 85)
        self.img_label.setScaledContents(True)
        self.update_pixmap(np_img)
        layout.addWidget(self.img_label)

        self.lbl_id = QLabel(f"ID: {original_id}")
        self.lbl_id.setStyleSheet("color: #777; font-size: 9px; border: none;")
        layout.addWidget(self.lbl_id)

        self.lbl_time = QLabel("00:00:000")
        self.lbl_time.setStyleSheet("color: #00ff00; font-family: 'Courier New'; font-size: 10px; border: none;")
        layout.addWidget(self.lbl_time)

        btn_layout = QHBoxLayout()
        self.btn_del = self._create_btn(QStyle.SP_TrashIcon, self.on_delete)
        self.btn_save = self._create_btn(QStyle.SP_DialogSaveButton, self.on_save)
        self.btn_replace = self._create_btn(QStyle.SP_BrowserReload, self.on_replace)
        btn_layout.addWidget(self.btn_del)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_replace)
        layout.addLayout(btn_layout)

    def apply_frame_style(self):
        if self.playback_highlighted:
            border_color = "#f1c40f"
        elif self.mask_highlighted:
            border_color = "#2ecc71"
        else:
            border_color = "#444"
        self.setStyleSheet(f"QFrame {{ border: 2px solid {border_color}; background: #2b2b2b; }}")

    def set_mask_ui_state(self, show_toggle, in_mask):
        self.mask_toggle_btn.setVisible(show_toggle)
        if show_toggle:
            self.mask_toggle_btn.setText("-" if in_mask else "+")
            self.mask_toggle_btn.setStyleSheet("color: #2ecc71; font-weight: bold;")

    def toggle_frame_in_mask(self):
        self.parent_ui.toggle_current_mask_frame(self.original_id)

    def set_time(self, frame_index, fps):
        if fps <= 0:
            fps = 1
        total_ms = int((frame_index / fps) * 1000)
        minutes = total_ms // 60000
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
        if path:
            cv2.imwrite(path, self.np_img)

    def on_replace(self):
        path, _ = QFileDialog.getOpenFileName(None, "Replace", "", "Images (*.jpg *.png *.jfif)")
        if path:
            img = cv2.imread(path)
            self.parent_ui.replace_frame(self.batch_idx, self.frame_idx, img)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_ui.select_timeline_frame(self.original_id, self.np_img)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and not self.parent_ui.mode_mask_add_frames:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(f"{self.batch_idx}:{self.frame_idx}")
            drag.setMimeData(mime)
            drag.exec_(Qt.MoveAction)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = VideoProcessor()
        self.project = ProjectManager()

        self.grid_size = 4
        self.show_batch_names = True
        self.show_frame_ids = True
        self.show_timestamps = True

        self.masks = []
        self.current_mask_index = None
        self.mode_mask_edit = False
        self.mode_mask_add_frames = False
        self.frame_lookup = {}
        self.timeline_widgets = {}

        self.current_preview_frame = None
        self.current_preview_frame_id = None

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_step)
        self.prerender_frames = []
        self.prerender_source_frame_ids = []
        self.current_playback_index = 0
        self.is_playing = False

        self.setWindowTitle("Batch Video Editor")
        self.resize(1300, 900)

        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)

        self.welcome_screen = self.init_welcome_ui()
        self.editor_screen = self.init_editor_ui()

        self.central_stack.addWidget(self.welcome_screen)
        self.central_stack.addWidget(self.editor_screen)
        self.central_stack.setCurrentWidget(self.welcome_screen)

    def init_editor_ui(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.batch_list.customContextMenuRequested.connect(self.show_batch_context_menu)
        self.batch_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.batch_list.model().rowsMoved.connect(self.on_batches_reordered)

        self.preview = MaskPreviewLabel(self)

        self.settings_panel = QGroupBox("Mask + Export")
        settings_layout = QVBoxLayout(self.settings_panel)

        mask_group = QGroupBox("Маски")
        mask_layout = QVBoxLayout(mask_group)

        lists_layout = QHBoxLayout()
        self.mask_list = QListWidget()
        self.mask_frames_list = QListWidget()
        self.mask_frames_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.mask_list.currentRowChanged.connect(self.select_mask)
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self.show_mask_context_menu)

        self.mask_frames_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_frames_list.customContextMenuRequested.connect(self.show_mask_frames_context_menu)

        lists_layout.addWidget(self.mask_list)
        lists_layout.addWidget(self.mask_frames_list)
        mask_layout.addLayout(lists_layout)

        mask_btns = QHBoxLayout()
        self.btn_new_mask = QPushButton("Новая маска")
        self.btn_run_inpaint = QPushButton("Inpaint выбранные кадры")
        self.btn_new_mask.clicked.connect(self.create_mask)
        self.btn_run_inpaint.clicked.connect(self.run_inpaint)
        mask_btns.addWidget(self.btn_new_mask)
        mask_btns.addWidget(self.btn_run_inpaint)
        mask_layout.addLayout(mask_btns)

        export_group = QGroupBox("Настройки экспорта")
        export_layout = QFormLayout(export_group)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(32)
        self.fps_spin.valueChanged.connect(self.on_export_settings_changed)
        export_layout.addRow("FPS:", self.fps_spin)

        self.interp_steps = QListWidget()
        self.interp_steps.setMaximumHeight(150)
        export_layout.addRow("Цепочка:", self.interp_steps)

        add_btns = QHBoxLayout()
        self.btn_add_film = QPushButton("+ FILM")
        self.btn_add_cv = QPushButton("+ CV")
        self.btn_clear_steps = QPushButton("Clear")
        add_btns.addWidget(self.btn_add_film)
        add_btns.addWidget(self.btn_add_cv)
        add_btns.addWidget(self.btn_clear_steps)
        export_layout.addRow(add_btns)

        settings_layout.addWidget(mask_group)
        settings_layout.addWidget(export_group)

        self.main_splitter.addWidget(self.batch_list)
        self.main_splitter.addWidget(self.preview)
        self.main_splitter.addWidget(self.settings_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(280)

        self.timeline_container = QWidget()
        self.timeline_container.setAcceptDrops(True)
        self.timeline_container.dragEnterEvent = self.t_dragEnter
        self.timeline_container.dropEvent = self.t_drop

        self.timeline_layout = QHBoxLayout(self.timeline_container)
        self.timeline_layout.setAlignment(Qt.AlignLeft)

        self.scroll.viewport().installEventFilter(self)
        self.scroll.setWidget(self.timeline_container)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Inpaint: %p%")

        playback_row = QHBoxLayout()
        self.btn_play = QPushButton("▶ Start")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.playback_progress = QProgressBar()
        self.playback_progress.setRange(0, 1000)
        self.playback_progress.setValue(0)
        self.playback_progress.setFormat("Preview timeline: %p%")
        playback_row.addWidget(self.btn_play)
        playback_row.addWidget(self.playback_progress)

        action_btns = QHBoxLayout()
        btn_load = QPushButton("Добавить батчи (.jfif)")
        btn_load.clicked.connect(self.load_images)

        self.btn_export = QPushButton("Экспортировать видео")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setStyleSheet("background-color: #2b5e2b; color: white; padding: 5px;")

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

        action_btns.addWidget(btn_load)
        action_btns.addWidget(self.tgl_batch)
        action_btns.addWidget(self.tgl_id)
        action_btns.addWidget(self.tgl_time)
        action_btns.addStretch()
        action_btns.addWidget(self.btn_export)

        main_layout.addWidget(self.main_splitter, stretch=1)
        main_layout.addWidget(self.scroll)
        main_layout.addWidget(self.progress_bar)
        main_layout.addLayout(playback_row)
        main_layout.addLayout(action_btns)

        self.btn_add_film.clicked.connect(lambda: self.add_interpolation_step("film"))
        self.btn_add_cv.clicked.connect(lambda: self.add_interpolation_step("opencv"))
        self.btn_clear_steps.clicked.connect(self.clear_interpolation_steps)

        self.preview.setMinimumSize(640, 360)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return widget

    def init_welcome_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("Batch Video Editor")
        title.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)

        btn_open = QPushButton("Открыть проект...")
        btn_open.setFixedSize(200, 40)
        layout.addWidget(btn_open)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #444; margin: 20px;")
        layout.addWidget(line)

        layout.addWidget(QLabel("Создать новый проект (выберите сетку):"))

        grid_btns = QHBoxLayout()
        for size in [2, 3, 4]:
            btn = QPushButton(f"{size ** 2} кадров\n({int(size)}x{int(size)})")
            btn.setFixedSize(100, 60)
            btn.clicked.connect(lambda checked, s=size: self.start_new_project(s))
            grid_btns.addWidget(btn)

        layout.addLayout(grid_btns)
        return widget

    def start_new_project(self, size):
        self.grid_size = size
        self.central_stack.setCurrentWidget(self.editor_screen)

    def on_export_settings_changed(self):
        self.refresh_timeline()
        self.invalidate_prerender()

    def add_interpolation_step(self, step_name):
        self.interp_steps.addItem(step_name)
        self.on_export_settings_changed()

    def clear_interpolation_steps(self):
        self.interp_steps.clear()
        self.on_export_settings_changed()

    def invalidate_prerender(self):
        self.prerender_frames = []
        self.prerender_source_frame_ids = []
        self.current_playback_index = 0
        self.playback_progress.setValue(0)

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выбор батчей", "", "Images (*.jfif *.jpg *.png)")
        if not files:
            return

        for f in files:
            frames = self.processor.split_batch(f, self.grid_size)
            if not frames:
                continue
            new_batch = Batch(Path(f).name, frames)
            self.project.batches.append(new_batch)
            self.batch_list.addItem(new_batch.name)

        self.refresh_timeline()
        self.invalidate_prerender()

    def should_show_mask_overlay(self):
        return self.current_mask_index is not None

    def set_preview(self, img, frame_id=None):
        self.current_preview_frame = img.copy()
        self.current_preview_frame_id = frame_id

        current_mask = None
        if self.current_mask_index is not None:
            current_mask = self.masks[self.current_mask_index].get('mask')

        self.preview.set_content(img, current_mask, show_overlay=self.should_show_mask_overlay())

    def select_timeline_frame(self, frame_id, img):
        self.set_preview(img, frame_id)
        self.seek_playback_to_frame(frame_id)

    def toggle_visibility(self):
        self.show_batch_names = self.tgl_batch.isChecked()
        self.show_frame_ids = self.tgl_id.isChecked()
        self.show_timestamps = self.tgl_time.isChecked()
        self.refresh_timeline()

    def on_batches_reordered(self, parent, start, end, destination, row):
        start_idx = start.row() if hasattr(start, 'row') else start
        dest_idx = row

        if not self.project.batches:
            return

        moved_item = self.project.batches.pop(start_idx)
        actual_insert_idx = dest_idx - 1 if start_idx < dest_idx else dest_idx
        self.project.batches.insert(actual_insert_idx, moved_item)
        self.refresh_timeline()
        self.invalidate_prerender()

    def show_batch_context_menu(self, pos):
        menu = QMenu()
        del_action = QAction("Удалить", self)
        del_action.triggered.connect(self.delete_selected_batches)
        menu.addAction(del_action)
        menu.exec_(self.batch_list.mapToGlobal(pos))

    def delete_selected_batches(self):
        selected_items = self.batch_list.selectedItems()
        if not selected_items:
            return

        indices = sorted([self.batch_list.row(item) for item in selected_items], reverse=True)
        for index in indices:
            if index < len(self.project.batches):
                self.project.batches.pop(index)
            self.batch_list.takeItem(index)

        self.cleanup_masks_after_frame_changes()
        self.refresh_timeline()
        self.invalidate_prerender()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if self.batch_list.hasFocus():
                self.delete_selected_batches()

        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode_mask_edit:
                self.finish_mask_edit_mode()
                return
            if self.mode_mask_add_frames:
                self.finish_mask_add_mode()
                return

        if event.key() == Qt.Key_Space:
            self.toggle_playback()
            return

        super().keyPressEvent(event)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel and source is self.scroll.viewport():
            delta = event.angleDelta().y()
            self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - delta)
            return True
        return super().eventFilter(source, event)

    def refresh_timeline(self):
        while self.timeline_layout.count():
            item = self.timeline_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        fps = self.fps_spin.value()
        interp_count = self.interp_steps.count()
        frame_multiplier = 2 ** interp_count

        self.frame_lookup = {}
        self.timeline_widgets = {}
        global_idx = 0
        current_mask_frames = set()
        if self.current_mask_index is not None:
            current_mask_frames = self.masks[self.current_mask_index]['frame_ids']

        for b_idx, batch in enumerate(self.project.batches):
            batch_widget = QWidget()
            batch_v_layout = QVBoxLayout(batch_widget)

            frames_h_layout = QHBoxLayout()
            if not hasattr(batch, 'frame_ids'):
                batch.frame_ids = []

            for f_idx, img in enumerate(batch.frames):
                while len(batch.frame_ids) <= f_idx:
                    batch.frame_ids.append(f"{batch.name}:{len(batch.frame_ids)}")

                frame_id = batch.frame_ids[f_idx]
                self.frame_lookup[frame_id] = (b_idx, f_idx)

                fw = TimelineFrame(img, frame_id, b_idx, f_idx, self)
                self.timeline_widgets[frame_id] = fw
                fw.set_time(global_idx * frame_multiplier, fps)
                fw.lbl_id.setVisible(self.show_frame_ids)
                fw.lbl_time.setVisible(self.show_timestamps)

                in_current_mask = frame_id in current_mask_frames
                fw.mask_highlighted = in_current_mask
                fw.set_mask_ui_state(self.mode_mask_add_frames, in_current_mask)
                fw.playback_highlighted = (frame_id == self.get_current_playback_frame_id())
                fw.apply_frame_style()

                frames_h_layout.addWidget(fw)
                global_idx += 1

            batch_v_layout.addLayout(frames_h_layout)

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

        self.refresh_current_mask_frames_list()

    def delete_frame(self, b_idx, f_idx):
        if b_idx < len(self.project.batches):
            batch = self.project.batches[b_idx]
            if not hasattr(batch, 'frame_ids'):
                batch.frame_ids = [f"{batch.name}:{i}" for i in range(len(batch.frames))]
            if f_idx < len(batch.frames):
                batch.frames.pop(f_idx)
                if f_idx < len(batch.frame_ids):
                    frame_id = batch.frame_ids.pop(f_idx)
                    for m in self.masks:
                        m['frame_ids'].discard(frame_id)
                if not batch.frames:
                    self.project.batches.pop(b_idx)
                    self.batch_list.takeItem(b_idx)

        self.refresh_timeline()
        self.invalidate_prerender()

    def replace_frame(self, b_idx, f_idx, new_img):
        if new_img is None:
            return
        self.project.batches[b_idx].frames[f_idx] = new_img
        self.refresh_timeline()
        self.invalidate_prerender()

    def t_dragEnter(self, e):
        e.accept()

    def t_drop(self, e):
        try:
            data = e.mimeData().text().split(':')
            src_b_idx, src_f_idx = int(data[0]), int(data[1])
        except Exception:
            return

        pos = e.position().toPoint()
        target = self.timeline_container.childAt(pos)
        while target and not isinstance(target, TimelineFrame):
            target = target.parentWidget()

        src_batch = self.project.batches[src_b_idx]
        moving_frame = src_batch.frames.pop(src_f_idx)
        moving_id = src_batch.frame_ids.pop(src_f_idx)

        if target:
            dst_b_idx = target.batch_idx
            dst_f_idx = target.frame_idx
            self.project.batches[dst_b_idx].frames.insert(dst_f_idx, moving_frame)
            self.project.batches[dst_b_idx].frame_ids.insert(dst_f_idx, moving_id)
        else:
            self.project.batches[-1].frames.append(moving_frame)
            self.project.batches[-1].frame_ids.append(moving_id)

        self.refresh_timeline()
        self.invalidate_prerender()

    def export_video(self):
        current_frames = []
        for b in self.project.batches:
            current_frames.extend(b.frames)

        if not current_frames:
            print("Нет кадров для экспорта")
            return

        pipeline = [self.interp_steps.item(i).text() for i in range(self.interp_steps.count())]
        target_fps = self.fps_spin.value()
        save_path, _ = QFileDialog.getSaveFileName(self, "Экспорт", "output.mp4", "Video (*.mp4)")
        if not save_path:
            return

        final_frames = self.processor.process_sequence(current_frames, pipeline)
        h, w, _ = final_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, float(target_fps), (w, h))
        for f in final_frames:
            out.write(f)
        out.release()
        print(f"Экспорт завершен. Итого кадров: {len(final_frames)}")

    # ---------- Playback ----------
    def get_flat_frames_with_ids(self):
        frames, frame_ids = [], []
        for batch in self.project.batches:
            if not hasattr(batch, 'frame_ids'):
                batch.frame_ids = [f"{batch.name}:{i}" for i in range(len(batch.frames))]
            for i, frame in enumerate(batch.frames):
                frames.append(frame)
                frame_ids.append(batch.frame_ids[i])
        return frames, frame_ids

    def build_prerender(self):
        frames, frame_ids = self.get_flat_frames_with_ids()
        if not frames:
            self.prerender_frames = []
            self.prerender_source_frame_ids = []
            return

        interpolation_depth = self.interp_steps.count()
        self.prerender_frames = self.processor.process_sequence_fast_cv(frames, interpolation_depth)

        # map rendered frame index back to nearest source frame id
        # source frames are inserted every 2**depth position
        stride = 2 ** interpolation_depth if interpolation_depth > 0 else 1
        mapped_ids = []
        for idx in range(len(self.prerender_frames)):
            src_idx = min(len(frame_ids) - 1, idx // stride)
            mapped_ids.append(frame_ids[src_idx])
        self.prerender_source_frame_ids = mapped_ids

    def get_current_playback_frame_id(self):
        if not self.prerender_source_frame_ids:
            return None
        if self.current_playback_index >= len(self.prerender_source_frame_ids):
            return None
        return self.prerender_source_frame_ids[self.current_playback_index]

    def update_playback_visuals(self):
        frame_id = self.get_current_playback_frame_id()
        for fid, widget in self.timeline_widgets.items():
            widget.playback_highlighted = (fid == frame_id)
            widget.apply_frame_style()

        total = max(1, len(self.prerender_frames) - 1)
        value = int((self.current_playback_index / total) * 1000) if self.prerender_frames else 0
        self.playback_progress.setValue(value)

    def playback_step(self):
        if not self.prerender_frames:
            self.stop_playback()
            return

        if self.current_playback_index >= len(self.prerender_frames):
            self.stop_playback()
            return

        frame = self.prerender_frames[self.current_playback_index]
        frame_id = self.get_current_playback_frame_id()
        self.set_preview(frame, frame_id)
        self.update_playback_visuals()
        self.current_playback_index += 1

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        if not self.prerender_frames:
            self.build_prerender()
        if not self.prerender_frames:
            return

        self.is_playing = True
        self.btn_play.setText("⏸ Stop")
        fps = max(1, self.fps_spin.value())
        self.playback_timer.start(int(1000 / fps))

    def stop_playback(self):
        self.is_playing = False
        self.btn_play.setText("▶ Start")
        self.playback_timer.stop()

    def seek_playback_to_frame(self, frame_id):
        if not self.prerender_frames:
            self.build_prerender()
        if not self.prerender_source_frame_ids:
            return
        try:
            idx = self.prerender_source_frame_ids.index(frame_id)
        except ValueError:
            return

        self.current_playback_index = idx
        frame = self.prerender_frames[self.current_playback_index]
        self.set_preview(frame, frame_id)
        self.update_playback_visuals()

    # -------------------- Mask management --------------------
    def create_mask(self):
        idx = len(self.masks) + 1
        name = f"mask_{idx}"
        self.masks.append({'name': name, 'mask': None, 'frame_ids': set()})
        self.mask_list.addItem(name)
        self.mask_list.setCurrentRow(self.mask_list.count() - 1)

    def show_mask_context_menu(self, pos):
        row = self.mask_list.indexAt(pos).row()
        if row < 0:
            return

        menu = QMenu()
        delete_action = QAction("Удалить маску", self)
        edit_action = QAction("Редактировать маску", self)
        add_frames_action = QAction("Добавить кадры", self)

        delete_action.triggered.connect(lambda: self.delete_mask(row))
        edit_action.triggered.connect(self.start_mask_edit_mode)
        add_frames_action.triggered.connect(self.start_mask_add_mode)

        menu.addAction(delete_action)
        menu.addAction(edit_action)
        menu.addAction(add_frames_action)
        menu.exec_(self.mask_list.mapToGlobal(pos))

    def show_mask_frames_context_menu(self, pos):
        if self.current_mask_index is None:
            return
        item = self.mask_frames_list.itemAt(pos)
        if item is None:
            return

        menu = QMenu()
        remove_action = QAction("Удалить кадры из маски", self)
        remove_action.triggered.connect(self.remove_selected_frames_from_current_mask)
        menu.addAction(remove_action)
        menu.exec_(self.mask_frames_list.mapToGlobal(pos))

    def delete_mask(self, row):
        if row < 0 or row >= len(self.masks):
            return
        self.masks.pop(row)
        self.mask_list.takeItem(row)
        if self.current_mask_index == row:
            self.current_mask_index = None
        elif self.current_mask_index is not None and self.current_mask_index > row:
            self.current_mask_index -= 1
        self.refresh_current_mask_frames_list()
        self.refresh_timeline()

    def select_mask(self, row):
        if row < 0 or row >= len(self.masks):
            self.current_mask_index = None
            self.refresh_current_mask_frames_list()
            self.preview._render(False)
            self.refresh_timeline()
            return

        self.current_mask_index = row
        self.refresh_current_mask_frames_list()
        if self.current_preview_frame is not None:
            self.set_preview(self.current_preview_frame, self.current_preview_frame_id)
        self.refresh_timeline()

    def refresh_current_mask_frames_list(self):
        self.mask_frames_list.clear()
        if self.current_mask_index is None:
            return

        frame_ids = sorted(self.masks[self.current_mask_index]['frame_ids'])
        for fid in frame_ids:
            self.mask_frames_list.addItem(QListWidgetItem(fid))

    def toggle_current_mask_frame(self, frame_id):
        if self.current_mask_index is None:
            return
        frame_ids = self.masks[self.current_mask_index]['frame_ids']
        if frame_id in frame_ids:
            frame_ids.remove(frame_id)
        else:
            frame_ids.add(frame_id)
        self.refresh_current_mask_frames_list()
        self.refresh_timeline()

    def remove_selected_frames_from_current_mask(self):
        if self.current_mask_index is None:
            return
        selected = self.mask_frames_list.selectedItems()
        if not selected:
            return
        frame_ids = self.masks[self.current_mask_index]['frame_ids']
        for it in selected:
            frame_ids.discard(it.text())
        self.refresh_current_mask_frames_list()
        self.refresh_timeline()

    def start_mask_edit_mode(self):
        if self.current_mask_index is None:
            return
        if self.current_preview_frame is None:
            QMessageBox.warning(self, "Mask", "Выберите кадр в таймлайне перед редактированием маски")
            return

        self.mode_mask_edit = True
        self.mode_mask_add_frames = False

        mask = self.masks[self.current_mask_index].get('mask')
        if mask is None:
            h, w = self.current_preview_frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            self.masks[self.current_mask_index]['mask'] = mask
        elif mask.shape[:2] != self.current_preview_frame.shape[:2]:
            mask = cv2.resize(mask, (self.current_preview_frame.shape[1], self.current_preview_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.masks[self.current_mask_index]['mask'] = mask

        self.preview.edit_mode = True
        self.preview.mask = self.masks[self.current_mask_index]['mask']
        self.preview._render(True)
        self.set_editor_controls_enabled(False)

    def finish_mask_edit_mode(self):
        self.mode_mask_edit = False
        self.preview.edit_mode = False
        self.set_editor_controls_enabled(True)
        if self.current_preview_frame is not None:
            self.set_preview(self.current_preview_frame, self.current_preview_frame_id)

    def start_mask_add_mode(self):
        if self.current_mask_index is None:
            return
        self.mode_mask_add_frames = True
        self.mode_mask_edit = False
        #self.set_editor_controls_enabled(False)
        self.refresh_timeline()

    def finish_mask_add_mode(self):
        self.mode_mask_add_frames = False
        self.set_editor_controls_enabled(True)
        self.refresh_timeline()

    def set_editor_controls_enabled(self, enabled):
        widgets = [
            self.batch_list, self.settings_panel, self.scroll,
            self.btn_export, self.btn_new_mask, self.btn_run_inpaint,
            self.mask_list, self.mask_frames_list,
            self.tgl_batch, self.tgl_id, self.tgl_time,
            self.btn_play
        ]
        for w in widgets:
            w.setEnabled(enabled)
        self.preview.setEnabled(True)

    def cleanup_masks_after_frame_changes(self):
        valid = set(self.frame_lookup.keys())
        for m in self.masks:
            m['frame_ids'] = {fid for fid in m['frame_ids'] if fid in valid}

    def run_inpaint(self):
        if not self.masks:
            QMessageBox.information(self, "Inpaint", "Нет масок для применения")
            return

        masks_to_apply = [m for m in self.masks if m.get('mask') is not None and len(m.get('frame_ids', [])) > 0]
        if not masks_to_apply:
            QMessageBox.information(self, "Inpaint", "Нет выбранных кадров с нарисованной маской")
            return

        self.progress_bar.setValue(0)

        def on_progress(done, total):
            percent = int((done / total) * 100) if total else 0
            self.progress_bar.setValue(percent)

        try:
            total = self.processor.inpaint_project_frames(self.project.batches, masks_to_apply, progress_cb=on_progress)
        except Exception as e:
            QMessageBox.critical(self, "Inpaint error", str(e))
            return

        self.refresh_timeline()
        self.invalidate_prerender()
        QMessageBox.information(self, "Inpaint", f"Обработано кадров: {total}")
