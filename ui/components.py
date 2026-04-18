import cv2
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QScrollArea, QLabel, QPushButton,
    QFileDialog, QSplitter, QGroupBox, QFormLayout, QSpinBox, QToolButton, QStyle, QFrame,
    QAbstractItemView, QSizePolicy, QStackedWidget, QMenu, QListWidgetItem, QProgressBar, QMessageBox,
    QTreeWidget, QTreeWidgetItem, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
    QApplication, QSpacerItem, QSlider
)
from PySide6.QtGui import QImage, QPixmap, QDrag, QAction, QPainter, QPen, QColor, QBrush, QFont
from PySide6.QtCore import Qt, QMimeData, QEvent, QTimer, QPointF, QRectF

from core import VideoProcessor, ProjectManager, Batch


class MaskListItemWidget(QWidget):
    def __init__(self, name, selected=True, visible=True):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self.lbl = QLabel(name)
        self.btn_select = QToolButton()
        self.btn_visible = QToolButton()

        self.btn_select.setCheckable(True)
        self.btn_select.setChecked(selected)
        self.btn_visible.setCheckable(True)
        self.btn_visible.setChecked(visible)

        self._sync_icons()
        layout.addWidget(self.lbl)
        layout.addStretch()
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_visible)

        self.btn_select.clicked.connect(self._sync_icons)
        self.btn_visible.clicked.connect(self._sync_icons)

    def _sync_icons(self):
        self.btn_select.setText("✓" if self.btn_select.isChecked() else "○")
        self.btn_visible.setText("👁" if self.btn_visible.isChecked() else "🚫")


class MaskPreviewLabel(QLabel):
    def __init__(self, parent_ui):
        super().__init__("Выберите кадр для превью")
        self.parent_ui = parent_ui
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumWidth(500)
        self.setMouseTracking(True)

        self.base_img = None
        self.mask = None
        self.edit_mode = False
        self.brush_size = 20
        self.cursor_pos = None

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
            mask = self.mask
            if mask.shape[:2] != render.shape[:2]:
                mask = cv2.resize(mask, (render.shape[1], render.shape[0]), interpolation=cv2.INTER_NEAREST)
            overlay = render.copy()
            overlay[mask > 0] = [255, 60, 60]
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
        if not self.edit_mode or self.base_img is None or self.cursor_pos is None:
            return

        mapped = self._map_label_to_image(self.cursor_pos)
        if mapped is None:
            return

        _, _, scale = mapped
        radius = max(1, int(self.brush_size * scale))

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor("#8dff73"), 2))
        painter.setBrush(Qt.NoBrush)
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
        self.cursor_pos = None
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


class GraphNodeItem(QGraphicsRectItem):
    def __init__(self, node_id, label, color):
        super().__init__(0, 0, 120, 44)
        self.node_id = node_id
        self.label = label
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#64748b"), 1.4))
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)

        self.txt = QGraphicsTextItem(label, self)
        self.txt.setDefaultTextColor(QColor("#e5e7eb"))
        self.txt.setPos(10, 10)


class ProcessGraphView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing)
        self.setMinimumHeight(230)
        self.edges = []
        self.node_items = {}
        self.connect_start = None

    def clear_graph(self):
        self.scene.clear()
        self.edges = []
        self.node_items = {}
        self.connect_start = None

    def rebuild_pipeline(self, process_chain):
        self.clear_graph()
        nodes = ["input", *process_chain, "output"]
        palette = {
            "input": "#334155",
            "output": "#334155",
            "film": "#0f766e",
            "cv": "#1d4ed8"
        }

        x = 20
        y = 40
        for i, name in enumerate(nodes):
            item = GraphNodeItem(f"n{i}", name.upper(), palette.get(name, "#3b4252"))
            item.setPos(x, y + (30 if i % 2 else 0))
            self.scene.addItem(item)
            self.node_items[item.node_id] = item
            x += 155

        ids = list(self.node_items.keys())
        for i in range(len(ids) - 1):
            self.add_edge(ids[i], ids[i + 1])

        self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-20, -20, 40, 40))

    def add_edge(self, src_id, dst_id):
        src = self.node_items.get(src_id)
        dst = self.node_items.get(dst_id)
        if not src or not dst:
            return

        sx = src.scenePos().x() + src.rect().width()
        sy = src.scenePos().y() + src.rect().height() / 2
        dx = dst.scenePos().x()
        dy = dst.scenePos().y() + dst.rect().height() / 2

        line = self.scene.addLine(sx, sy, dx, dy, QPen(QColor("#60a5fa"), 2))
        self.edges.append((src_id, dst_id, line))

    def refresh_edges(self):
        for src_id, dst_id, line in self.edges:
            src = self.node_items.get(src_id)
            dst = self.node_items.get(dst_id)
            if not src or not dst:
                continue
            sx = src.scenePos().x() + src.rect().width()
            sy = src.scenePos().y() + src.rect().height() / 2
            dx = dst.scenePos().x()
            dy = dst.scenePos().y() + dst.rect().height() / 2
            line.setLine(sx, sy, dx, dy)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, GraphNodeItem):
            if self.connect_start is None:
                self.connect_start = item.node_id
            else:
                if self.connect_start != item.node_id:
                    self.add_edge(self.connect_start, item.node_id)
                self.connect_start = None
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.refresh_edges()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.refresh_edges()


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

        self.setFixedSize(160, 160)
        self.apply_frame_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.addStretch()
        self.mask_toggle_btn = QToolButton()
        self.mask_toggle_btn.setFixedSize(20, 20)
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
        self.lbl_id.setStyleSheet("color:#94a3b8; font-size:9px; border:none;")
        layout.addWidget(self.lbl_id)

        self.lbl_time = QLabel("00:00:000")
        self.lbl_time.setStyleSheet("color:#86efac; font-family:'Consolas'; font-size:10px; border:none;")
        self.lbl_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_time)

        btns = QHBoxLayout()
        self.btn_del = self._btn(QStyle.SP_TrashIcon, self.on_delete)
        self.btn_save = self._btn(QStyle.SP_DialogSaveButton, self.on_save)
        self.btn_replace = self._btn(QStyle.SP_BrowserReload, self.on_replace)
        btns.addWidget(self.btn_del)
        btns.addWidget(self.btn_save)
        btns.addWidget(self.btn_replace)
        layout.addLayout(btns)

    def _btn(self, icon_style, slot):
        b = QToolButton()
        b.setFixedSize(22, 22)
        b.setIcon(self.style().standardIcon(icon_style))
        b.clicked.connect(slot)
        return b

    def apply_frame_style(self):
        if self.playback_highlighted:
            c = "#f59e0b"
        elif self.mask_highlighted:
            c = "#22c55e"
        else:
            c = "#374151"
        self.setStyleSheet(f"QFrame{{border:2px solid {c}; background:#131a26; border-radius:6px;}}")

    def set_mask_ui_state(self, show_toggle, in_mask):
        self.mask_toggle_btn.setVisible(show_toggle)
        if show_toggle:
            self.mask_toggle_btn.setText("-" if in_mask else "+")
            self.mask_toggle_btn.setStyleSheet("color:#22c55e; font-weight:bold;")

    def set_time(self, frame_index, fps):
        fps = max(1, fps)
        tms = int((frame_index / fps) * 1000)
        m = tms // 60000
        s = (tms % 60000) // 1000
        ms = tms % 1000
        self.lbl_time.setText(f"{m:02}:{s:02}:{ms:03}")

    def update_pixmap(self, np_img):
        rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.img_label.setPixmap(QPixmap.fromImage(q))

    def toggle_frame_in_mask(self):
        self.parent_ui.toggle_current_mask_frame(self.original_id)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent_ui.select_timeline_frame(self.original_id, self.np_img)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton and not self.parent_ui.mode_mask_add_frames:
            d = QDrag(self)
            m = QMimeData()
            m.setText(f"{self.batch_idx}:{self.frame_idx}")
            d.setMimeData(m)
            d.exec_(Qt.MoveAction)

    def on_delete(self):
        self.parent_ui.delete_frame(self.batch_idx, self.frame_idx)

    def on_save(self):
        path, _ = QFileDialog.getSaveFileName(None, "Save", "", "PNG (*.png)")
        if path:
            cv2.imwrite(path, self.np_img)

    def on_replace(self):
        path, _ = QFileDialog.getOpenFileName(None, "Replace", "", "Images (*.jpg *.png *.jfif)")
        if path:
            self.parent_ui.replace_frame(self.batch_idx, self.frame_idx, cv2.imread(path))


class BatchItemWidget(QWidget):
    def __init__(self, batch_name, first_frame_img, parent=None):
        super().__init__(parent)

        # Главный контейнер с рамкой для эффекта "карточки"
        self.container = QFrame()
        self.container.setObjectName("batchCard")
        self.container.setStyleSheet("""
            #batchCard {
                background-color: #1a2234;
                border: 1px solid #334155;
                border-radius: 8px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)

        card_layout = QHBoxLayout(self.container)
        card_layout.setContentsMargins(8, 4, 8, 8)

        # 1. Миниатюра (Превью первого кадра)
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(40, 40)
        self.thumb_label.setStyleSheet("border-radius: 4px; background: #000;")

        if first_frame_img is not None:
            # Конвертируем numpy в QPixmap
            h, w, _ = first_frame_img.shape
            qimg = QImage(first_frame_img.data, w, h, w * 3, QImage.Format_BGR888)
            pix = QPixmap.fromImage(qimg).scaled(50, 50, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.thumb_label.setPixmap(pix)
            self.thumb_label.setScaledContents(True)

            # 2. Название с авто-сокращением (Ellipsis)
            self.name_label = QLabel(batch_name)
            self.name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            font_metrics = self.name_label.fontMetrics()
            elided_text = font_metrics.elidedText(batch_name, Qt.ElideRight, 150)
            self.name_label.setText(elided_text)
            self.name_label.setToolTip(batch_name)

            # 3. Иконка перетаскивания
            self.grabber_label = QLabel("⠿")
            self.grabber_label.setFixedWidth(20)
            self.grabber_label.setAlignment(Qt.AlignCenter)
            self.grabber_label.setStyleSheet("color: #475569; font-size: 18px;")

            card_layout.addWidget(self.thumb_label)
            card_layout.addWidget(self.name_label)
            card_layout.addWidget(self.grabber_label)

            layout.addWidget(self.container)

    # Метод для обновления стиля при выделении (вызывается из списка)
    def set_selected(self, is_selected):
        if is_selected:
            self.container.setStyleSheet("""
                #batchCard {
                    background-color: #1f2937;
                    border: 1px solid #60a5fa;
                    border-radius: 8px;
                }
            """)
        else:
            self.container.setStyleSheet("""
                #batchCard {
                    background-color: #1a2234;
                    border: 1px solid #334155;
                    border-radius: 8px;
                }
            """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        QApplication.setFont(QFont("DejaVu Sans", 10))
        #self.setWindowFlags(Qt.FramelessWindowHint)

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

        self.process_chain = []

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self.playback_step)
        self.prerender_frames = []
        self.prerender_source_frame_ids = []
        self.current_playback_index = 0
        self.is_playing = False

        self.setWindowTitle("Batch Video Studio")
        self.resize(1520, 940)
        self.setStyleSheet("""
            QMainWindow{background:#0b1220;color:#e2e8f0;}
            QWidget{color:#e2e8f0;}
            QLabel{color:#cbd5e1;}
            QMenu{background:#111827;color:#e5e7eb;border:1px solid #374151;}
            QMenu::item:selected{background:#1f2937;}
            QGroupBox{border:1px solid #253047;border-radius:8px;margin-top:10px;background:#0f172a;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px;color:#f3f4f6;}
            QPushButton{background:#1f2937;border:1px solid #334155;border-radius:6px;padding:6px 10px;color:#e5e7eb;}
            QPushButton:hover{background:#263348;}
            
            QScrollBar:vertical {background: transparent;width: 8px;margin: 0px;}
            QScrollBar::handle:vertical {background: #dcdcdc;border-radius: 4px;min-height: 20px;}
            QScrollBar::add-line:vertical, 
            QScrollBar::sub-line:vertical {height: 0px;}
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {background: none;}
            QScrollBar:horizontal {background: transparent;height: 8px;margin: 0px;}
            QScrollBar::handle:horizontal {background: #dcdcdc;border-radius: 4px;min-width: 20px;}
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {width: 0px;}
            QScrollBar::add-page:horizontal,
            QScrollBar::sub-page:horizontal {background: none;}
            
            QToolButton{background:#1f2937;border:1px solid #334155;border-radius:4px;color:#e5e7eb;}
            QListWidget,QTreeWidget,QScrollArea,QGraphicsView,QSpinBox{background:#111827;border:1px solid #334155;border-radius:6px;color:#e5e7eb;}
            QProgressBar{background:#111827;border:1px solid #334155;border-radius:8px;}
            QProgressBar::chunk{background:#22c55e;border-radius:8px;}
            QSplitter::handle{background-color: #0b1220;width: 6px;}
            
            QTreeWidget {background-color: #111827;border: 1px solid #334155;border-radius: 6px;color: #e5e7eb;outline: none;}
            QTreeWidget::item {padding: 6px;border-radius: 4px;}
            QTreeWidget::item:selected {background-color: #1f2937; /* Используем тот же цвет, что в QPushButton:hover */color: #ffffff;border: 1px solid #475569;}
            QTreeWidget::item:hover {background-color: #1a2234;}
            QTreeWidget::item:selected:hover {background-color: #263348;}
            QTreeWidget::header {background-color: #1f2937;color: #e5e7eb;border: none;}
            
            QListWidget {background: #0f172a;border: none;outline: none;padding: 5px;}
            QListWidget::item {background: transparent;border: none;margin-bottom: 4px;}
            QListWidget::item:selected {background: transparent;border: none;}
        """)

        self.central_stack = QStackedWidget()
        self.setCentralWidget(self.central_stack)
        self.welcome_screen = self.init_welcome_ui()
        self.editor_screen = self.init_editor_ui()
        self.central_stack.addWidget(self.welcome_screen)
        self.central_stack.addWidget(self.editor_screen)
        self.central_stack.setCurrentWidget(self.welcome_screen)

    def init_welcome_ui(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.setAlignment(Qt.AlignCenter)
        title = QLabel("Batch Video Studio")
        title.setStyleSheet("font-size:32px;font-weight:bold;")
        l.addWidget(title)
        for size in [2, 3, 4]:
            b = QPushButton(f"Create {size}x{size} Project")
            b.clicked.connect(lambda _, s=size: self.start_new_project(s))
            l.addWidget(b)
        return w

    def init_editor_ui(self):
        w = QWidget()
        root = QVBoxLayout(w)

        top = QSplitter(Qt.Horizontal)

        # Left
        left = QGroupBox("Batches")
        ll = QVBoxLayout(left)
        self.batch_list = QListWidget()
        self.batch_list.setMinimumWidth(300)
        self.batch_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.batch_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.batch_list.customContextMenuRequested.connect(self.show_batch_context_menu)
        self.batch_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.batch_list.model().rowsMoved.connect(self.on_batches_reordered)
        ll.addWidget(self.batch_list)
        btn_add_batch = QPushButton("+ Add Batch")
        btn_add_batch.clicked.connect(self.load_images)
        ll.addWidget(btn_add_batch)

        # Center
        center = QWidget()
        cl = QVBoxLayout(center)
        prev_group = QGroupBox("Video Preview")
        pgl = QVBoxLayout(prev_group)
        self.preview = MaskPreviewLabel(self)
        pgl.addWidget(self.preview)

        controls = QFrame()
        controls_l = QHBoxLayout(controls)
        self.btn_play = QPushButton("▶")
        self.btn_play.clicked.connect(self.toggle_playback)

        class ReadOnlySlider(QSlider):
            def mousePressEvent(self, event):
                event.ignore()
            def mouseMoveEvent(self, event):
                event.ignore()

        self.playback_progress = ReadOnlySlider(Qt.Horizontal)
        self.playback_progress.setRange(0, 1000)
        self.playback_progress.setStyleSheet("""
                    QSlider::groove:horizontal {
                        border: 1px solid #333;
                        height: 6px;
                        background: #333; /* Цвет правой части (непройденной) */
                        margin: 2px 0;
                        border-radius: 3px;
                    }

                    /* Цвет левой части (прошедшей) */
                    QSlider::sub-page:horizontal {
                        background: #6c006c; /* Фиолетовый */
                        border-radius: 3px;
                    }

                    QSlider::handle:horizontal {
                        background: #FFFFFF;
                        border: 1px solid #FFFFFF;
                        width: 14px;
                        height: 14px;
                        margin: -5px 0;
                        border-radius: 7px;
                    }

                    QSlider::handle:horizontal:hover {
                        /* Если слайдер Read-Only, hover лучше отключить */
                        background: #FFFFFF;
                        border: 1px solid #FFFFFF;
                    }
                """)
        self.lbl_play_time = QLabel("00:00:000")
        controls_l.addWidget(self.btn_play)
        controls_l.addWidget(self.playback_progress, 1)
        controls_l.addWidget(self.lbl_play_time)
        pgl.addWidget(controls)

        cl.addWidget(prev_group)

        # Right
        right = QWidget()
        rl = QVBoxLayout(right)

        graph_group = QGroupBox("Processing Pipeline")
        ggl = QVBoxLayout(graph_group)
        self.graph_view = ProcessGraphView()
        ggl.addWidget(self.graph_view)

        row = QHBoxLayout()

        masks_group = QGroupBox("Masks")
        mgl = QVBoxLayout(masks_group)
        self.mask_list = QListWidget()
        self.mask_list.currentRowChanged.connect(self.select_mask)
        self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_list.customContextMenuRequested.connect(self.show_mask_context_menu)
        mgl.addWidget(self.mask_list)

        self.mask_frames_list = QListWidget()
        self.mask_frames_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.mask_frames_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.mask_frames_list.customContextMenuRequested.connect(self.show_mask_frames_context_menu)
        mgl.addWidget(self.mask_frames_list)

        self.batch_index_btns_wrap = QWidget()
        self.batch_index_btns_layout = QHBoxLayout(self.batch_index_btns_wrap)
        self.batch_index_btns_layout.setContentsMargins(0, 0, 0, 0)
        self.batch_index_btns_wrap.setVisible(False)
        mgl.addWidget(self.batch_index_btns_wrap)

        mb = QHBoxLayout()
        self.btn_new_mask = QPushButton("New Mask")
        self.btn_new_mask.clicked.connect(self.create_mask)
        self.btn_run_inpaint = QPushButton("Inpaint Selected")
        self.btn_run_inpaint.clicked.connect(self.run_inpaint)
        mb.addWidget(self.btn_new_mask)
        mb.addWidget(self.btn_run_inpaint)
        mgl.addLayout(mb)

        lib_group = QGroupBox("Processor Library")
        lgl = QVBoxLayout(lib_group)
        self.explorer_tree = QTreeWidget()
        self.explorer_tree.setHeaderHidden(True)
        self.populate_explorer_tree()
        self.explorer_tree.itemDoubleClicked.connect(self.on_explorer_item_activated)
        lgl.addWidget(self.explorer_tree)

        row.addWidget(masks_group)
        row.addWidget(lib_group)

        rl.addLayout(row)
        rl.addWidget(graph_group)

        top.addWidget(left)
        top.addWidget(center)
        top.addWidget(right)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 3)
        top.setStretchFactor(2, 2)

        root.addWidget(top, 1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFixedHeight(220)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.timeline_container = QWidget()
        self.timeline_container.setAcceptDrops(True)
        self.timeline_container.dragEnterEvent = self.t_dragEnter
        self.timeline_container.dropEvent = self.t_drop
        self.timeline_layout = QHBoxLayout(self.timeline_container)
        self.timeline_layout.setAlignment(Qt.AlignLeft)
        self.timeline_container.setStyleSheet("background-color: #0f172a;")
        self.scroll.viewport().installEventFilter(self)
        self.scroll.setWidget(self.timeline_container)
        root.addWidget(self.scroll)

        bottom_controls = QHBoxLayout()
        self.tgl_batch = QToolButton()
        self.tgl_batch.setText("Batch names")
        self.tgl_batch.setCheckable(True)
        self.tgl_batch.setChecked(True)
        self.tgl_batch.clicked.connect(self.toggle_visibility)

        self.tgl_id = QToolButton()
        self.tgl_id.setText("Frame IDs")
        self.tgl_id.setCheckable(True)
        self.tgl_id.setChecked(True)
        self.tgl_id.clicked.connect(self.toggle_visibility)

        self.tgl_time = QToolButton()
        self.tgl_time.setText("Timestamps")
        self.tgl_time.setCheckable(True)
        self.tgl_time.setChecked(True)
        self.tgl_time.clicked.connect(self.toggle_visibility)

        self.global_progress = QProgressBar()
        self.global_progress.setRange(0, 100)
        self.global_progress.setValue(0)
        self.global_progress.setTextVisible(False)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(32)
        self.fps_spin.valueChanged.connect(self.on_export_settings_changed)

        self.btn_export = QPushButton("Export Video")
        self.btn_export.clicked.connect(self.export_video)
        self.btn_export.setStyleSheet("background-color: #2b5e2b; color: white;")

        self.frames_count_lbl = QLabel("0")

        bottom_controls.addWidget(self.tgl_batch)
        bottom_controls.addWidget(self.tgl_id)
        bottom_controls.addWidget(self.tgl_time)
        bottom_controls.addWidget(self.global_progress, 1)
        bottom_controls.addWidget(QLabel("Frames:"))
        bottom_controls.addWidget(self.frames_count_lbl)
        bottom_controls.addWidget(QLabel(" |  FPS:"))
        bottom_controls.addWidget(self.fps_spin)
        bottom_controls.addSpacerItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        bottom_controls.addWidget(self.btn_export)

        root.addLayout(bottom_controls)

        self.graph_view.rebuild_pipeline(self.process_chain)
        return w

    def populate_explorer_tree(self):
        self.explorer_tree.clear()
        interp = QTreeWidgetItem(["interpolation"])
        interp.addChild(QTreeWidgetItem(["cv"]))
        interp.addChild(QTreeWidgetItem(["film"]))
        upscalers = QTreeWidgetItem(["upscalers"])
        self.explorer_tree.addTopLevelItem(interp)
        self.explorer_tree.addTopLevelItem(upscalers)
        self.explorer_tree.expandAll()

    def on_explorer_item_activated(self, item, _):
        parent = item.parent()
        if not parent:
            return
        folder = parent.text(0)
        leaf = item.text(0)
        if folder == "interpolation" and leaf in ("cv", "film"):
            self.process_chain.append(leaf)
            self.graph_view.rebuild_pipeline(self.process_chain)
            self.invalidate_prerender()

    def start_new_project(self, size):
        self.grid_size = size
        self.central_stack.setCurrentWidget(self.editor_screen)

    def on_export_settings_changed(self):
        self.refresh_timeline()
        self.invalidate_prerender()

    def toggle_visibility(self):
        self.show_batch_names = self.tgl_batch.isChecked()
        self.show_frame_ids = self.tgl_id.isChecked()
        self.show_timestamps = self.tgl_time.isChecked()
        self.refresh_timeline()

    def invalidate_prerender(self):
        self.prerender_frames = []
        self.prerender_source_frame_ids = []
        self.current_playback_index = 0
        self.playback_progress.setValue(0)
        self.lbl_play_time.setText("00:00:000")

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Выбор батчей", "", "Images (*.jfif *.jpg *.png)")
        if not files:
            return
        for f in files:
            frames = self.processor.split_batch(f, self.grid_size)
            if not frames:
                continue
            b = Batch(Path(f).name, frames)
            self.project.batches.append(b)
            self.batch_list.addItem(b.name)
            self.refresh_batch_list_ui()
        self.refresh_timeline()
        self.invalidate_prerender()

    def should_show_mask_overlay(self):
        if self.current_mask_index is None:
            return False
        m = self.masks[self.current_mask_index]
        return m.get('visible', True)

    def set_preview(self, img, frame_id=None):
        self.current_preview_frame = img.copy()
        self.current_preview_frame_id = frame_id
        mask = None
        if self.current_mask_index is not None and self.masks[self.current_mask_index].get('visible', True):
            mask = self.masks[self.current_mask_index].get('mask')
        self.preview.set_content(img, mask, show_overlay=self.should_show_mask_overlay())

    def select_timeline_frame(self, frame_id, img):
        self.set_preview(img, frame_id)
        self.seek_playback_to_frame(frame_id)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.mode_mask_edit:
                self.finish_mask_edit_mode()
                return
            if self.mode_mask_add_frames:
                self.finish_mask_add_mode()
                return
        if event.key() == Qt.Key_Space:
            if self.prerender_frames and self.current_playback_index >= len(self.prerender_frames):
                self.current_playback_index = 0
            self.toggle_playback()
            return
        super().keyPressEvent(event)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel and source is self.scroll.viewport() and not self.mode_mask_edit:
            delta = event.angleDelta().y()
            self.scroll.horizontalScrollBar().setValue(self.scroll.horizontalScrollBar().value() - delta)
            return True
        return super().eventFilter(source, event)

    def refresh_timeline(self):
        while self.timeline_layout.count():
            i = self.timeline_layout.takeAt(0)
            if i.widget():
                i.widget().deleteLater()

        self.frame_lookup = {}
        self.timeline_widgets = {}
        fps = self.fps_spin.value()
        mult = 2 ** len(self.process_chain)
        global_idx = 0
        current_mask_frames = set()
        if self.current_mask_index is not None:
            current_mask_frames = self.masks[self.current_mask_index]['frame_ids']

        flat_count = 0
        for b_idx, batch in enumerate(self.project.batches):
            bw = QWidget()
            bl = QVBoxLayout(bw)
            hl = QHBoxLayout()
            if not hasattr(batch, 'frame_ids'):
                batch.frame_ids = []

            for f_idx, img in enumerate(batch.frames):
                while len(batch.frame_ids) <= f_idx:
                    batch.frame_ids.append(f"{len(batch.frame_ids)}:{batch.name}")
                fid = batch.frame_ids[f_idx]
                self.frame_lookup[fid] = (b_idx, f_idx)

                fw = TimelineFrame(img, fid, b_idx, f_idx, self)
                fw.set_time(global_idx * mult, fps)
                fw.lbl_id.setVisible(self.show_frame_ids)
                fw.lbl_time.setVisible(self.show_timestamps)
                fw.mask_highlighted = fid in current_mask_frames
                fw.playback_highlighted = (fid == self.get_current_playback_frame_id())
                fw.set_mask_ui_state(self.mode_mask_add_frames, fid in current_mask_frames)
                fw.apply_frame_style()

                self.timeline_widgets[fid] = fw
                hl.addWidget(fw)
                global_idx += 1
                flat_count += 1

            bl.addLayout(hl)
            if self.show_batch_names:
                lb = QLabel(batch.name)
                lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lb.setStyleSheet("color:#60a5fa;font-weight:bold;")
                bl.addWidget(lb)
            self.timeline_layout.addWidget(bw)

        self.frames_count_lbl.setText(str(flat_count))
        self.refresh_current_mask_frames_list()
        self.rebuild_batch_index_buttons()

    def rebuild_batch_index_buttons(self):
        while self.batch_index_btns_layout.count():
            i = self.batch_index_btns_layout.takeAt(0)
            if i.widget():
                i.widget().deleteLater()

        max_frames = max((len(b.frames) for b in self.project.batches), default=0)
        for idx in range(max_frames):
            btn = QToolButton()
            btn.setText(str(idx + 1))
            btn.clicked.connect(lambda _, i=idx: self.toggle_mask_frame_index_for_all_batches(i))
            self.batch_index_btns_layout.addWidget(btn)

    def refresh_batch_list_ui(self):
        self.batch_list.clear()

        for batch in self.project.batches:
            item = QListWidgetItem(self.batch_list)
            first_frame = batch.frames[0] if len(batch.frames) > 0 else None
            custom_widget = BatchItemWidget(batch.name, first_frame)
            item.setSizeHint(custom_widget.sizeHint())

            self.batch_list.addItem(item)
            self.batch_list.setItemWidget(item, custom_widget)

        self.batch_list.itemSelectionChanged.connect(self.update_batch_styles)

    def update_batch_styles(self):
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            widget = self.batch_list.itemWidget(item)
            if widget:
                widget.set_selected(item.isSelected())

    def toggle_mask_frame_index_for_all_batches(self, frame_idx):
        if self.current_mask_index is None:
            return
        mask = self.masks[self.current_mask_index]
        targets = []
        for b in self.project.batches:
            if not hasattr(b, 'frame_ids'):
                continue
            if frame_idx < len(b.frame_ids):
                targets.append(b.frame_ids[frame_idx])

        if not targets:
            return

        all_present = all(t in mask['frame_ids'] for t in targets)
        if all_present:
            for t in targets:
                mask['frame_ids'].discard(t)
        else:
            for t in targets:
                mask['frame_ids'].add(t)

        self.refresh_current_mask_frames_list()
        self.refresh_timeline()

    def delete_frame(self, b_idx, f_idx):
        if b_idx < len(self.project.batches):
            b = self.project.batches[b_idx]
            if not hasattr(b, 'frame_ids'):
                b.frame_ids = [f"{b.name}:{i}" for i in range(len(b.frames))]
            if f_idx < len(b.frames):
                b.frames.pop(f_idx)
                if f_idx < len(b.frame_ids):
                    removed = b.frame_ids.pop(f_idx)
                    for m in self.masks:
                        m['frame_ids'].discard(removed)
                if not b.frames:
                    self.project.batches.pop(b_idx)
                    self.batch_list.takeItem(b_idx)

        self.refresh_timeline()
        self.invalidate_prerender()

    def replace_frame(self, b_idx, f_idx, img):
        if img is None:
            return
        self.project.batches[b_idx].frames[f_idx] = img
        self.refresh_timeline()
        self.invalidate_prerender()

    def show_batch_context_menu(self, pos):
        m = QMenu()
        a = QAction("Удалить", self)
        a.triggered.connect(self.delete_selected_batches)
        m.addAction(a)
        m.exec_(self.batch_list.mapToGlobal(pos))

    def delete_selected_batches(self):
        items = self.batch_list.selectedItems()
        if not items:
            return
        idxs = sorted([self.batch_list.row(i) for i in items], reverse=True)
        for idx in idxs:
            if idx < len(self.project.batches):
                self.project.batches.pop(idx)
            self.batch_list.takeItem(idx)
        self.cleanup_masks_after_frame_changes()
        self.refresh_timeline()
        self.invalidate_prerender()

    def on_batches_reordered(self, parent, start, end, destination, row):
        start_idx = start.row() if hasattr(start, 'row') else start
        dest_idx = row
        if not self.project.batches:
            return
        moved = self.project.batches.pop(start_idx)
        actual = dest_idx - 1 if start_idx < dest_idx else dest_idx
        self.project.batches.insert(actual, moved)
        self.refresh_timeline()
        self.invalidate_prerender()

    def t_dragEnter(self, e):
        e.accept()

    def t_drop(self, e):
        try:
            src_b, src_f = map(int, e.mimeData().text().split(':'))
        except Exception:
            return

        pos = e.position().toPoint()
        target = self.timeline_container.childAt(pos)
        while target and not isinstance(target, TimelineFrame):
            target = target.parentWidget()

        sb = self.project.batches[src_b]
        frame = sb.frames.pop(src_f)
        fid = sb.frame_ids.pop(src_f)

        if target:
            db, df = target.batch_idx, target.frame_idx
            self.project.batches[db].frames.insert(df, frame)
            self.project.batches[db].frame_ids.insert(df, fid)
        else:
            self.project.batches[-1].frames.append(frame)
            self.project.batches[-1].frame_ids.append(fid)

        self.refresh_timeline()
        self.invalidate_prerender()

    # playback
    def get_flat_frames_with_ids(self):
        frames, ids = [], []
        for b in self.project.batches:
            if not hasattr(b, 'frame_ids'):
                b.frame_ids = [f"{b.name}:{i}" for i in range(len(b.frames))]
            for i, fr in enumerate(b.frames):
                frames.append(fr)
                ids.append(b.frame_ids[i])
        return frames, ids

    def build_prerender(self):
        frames, ids = self.get_flat_frames_with_ids()
        if not frames:
            self.prerender_frames, self.prerender_source_frame_ids = [], []
            return
        depth = len(self.process_chain)
        self.prerender_frames = self.processor.process_sequence_fast_cv(frames, depth)
        stride = 2 ** depth if depth > 0 else 1
        self.prerender_source_frame_ids = [ids[min(len(ids)-1, i // stride)] for i in range(len(self.prerender_frames))]

    def get_current_playback_frame_id(self):
        if not self.prerender_source_frame_ids or self.current_playback_index >= len(self.prerender_source_frame_ids):
            return None
        return self.prerender_source_frame_ids[self.current_playback_index]

    def _scroll_timeline_to_frame(self, frame_id):
        w = self.timeline_widgets.get(frame_id)
        if not w:
            return
        x = w.mapTo(self.timeline_container, w.rect().topLeft()).x()
        center = x + w.width() // 2
        bar = self.scroll.horizontalScrollBar()
        bar.setValue(max(0, center - self.scroll.viewport().width() // 2))

    def update_playback_visuals(self):
        fid = self.get_current_playback_frame_id()
        for k, w in self.timeline_widgets.items():
            w.playback_highlighted = (k == fid)
            w.apply_frame_style()

        total = max(1, len(self.prerender_frames) - 1)
        self.playback_progress.setValue(int((self.current_playback_index / total) * 1000) if self.prerender_frames else 0)

        fps = max(1, self.fps_spin.value())
        tms = int((self.current_playback_index / fps) * 1000)
        m = tms // 60000
        s = (tms % 60000) // 1000
        ms = tms % 1000
        self.lbl_play_time.setText(f"{m:02}:{s:02}:{ms:03}")

        if fid:
            self._scroll_timeline_to_frame(fid)

    def playback_step(self):
        if not self.prerender_frames or self.current_playback_index >= len(self.prerender_frames):
            self.stop_playback()
            return
        frame = self.prerender_frames[self.current_playback_index]
        fid = self.get_current_playback_frame_id()
        self.set_preview(frame, fid)
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
        if self.current_playback_index >= len(self.prerender_frames):
            self.current_playback_index = 0
        self.is_playing = True
        self.btn_play.setText("⏸")
        self.playback_timer.start(int(1000 / max(1, self.fps_spin.value())))

    def stop_playback(self):
        self.is_playing = False
        self.btn_play.setText("▶")
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
        self.set_preview(self.prerender_frames[idx], frame_id)
        self.update_playback_visuals()

    # mask
    def create_mask(self):
        idx = len(self.masks) + 1
        self.masks.append({'name': f"mask {idx}", 'mask': None, 'frame_ids': set(), 'selected': True, 'visible': True})
        self.refresh_mask_list()
        self.mask_list.setCurrentRow(len(self.masks) - 1)

    def refresh_mask_list(self):
        self.mask_list.clear()
        for i, m in enumerate(self.masks):
            item = QListWidgetItem()
            w = MaskListItemWidget(m['name'], m.get('selected', True), m.get('visible', True))
            w.btn_select.clicked.connect(lambda _, idx=i: self.toggle_mask_selected(idx))
            w.btn_visible.clicked.connect(lambda _, idx=i: self.toggle_mask_visible(idx))
            item.setSizeHint(w.sizeHint())
            self.mask_list.addItem(item)
            self.mask_list.setItemWidget(item, w)

    def toggle_mask_selected(self, idx):
        self.masks[idx]['selected'] = not self.masks[idx].get('selected', True)

    def toggle_mask_visible(self, idx):
        self.masks[idx]['visible'] = not self.masks[idx].get('visible', True)
        if idx == self.current_mask_index and self.current_preview_frame is not None:
            self.set_preview(self.current_preview_frame, self.current_preview_frame_id)

    def show_mask_context_menu(self, pos):
        row = self.mask_list.indexAt(pos).row()
        if row < 0:
            return
        m = QMenu()
        a_del = QAction("Удалить маску", self)
        a_edit = QAction("Редактировать маску", self)
        a_add = QAction("Добавить кадры", self)
        a_del.triggered.connect(lambda: self.delete_mask(row))
        a_edit.triggered.connect(self.start_mask_edit_mode)
        a_add.triggered.connect(self.start_mask_add_mode)
        m.addAction(a_del)
        m.addAction(a_edit)
        m.addAction(a_add)
        m.exec_(self.mask_list.mapToGlobal(pos))

    def show_mask_frames_context_menu(self, pos):
        if self.current_mask_index is None or self.mask_frames_list.itemAt(pos) is None:
            return
        m = QMenu()
        a = QAction("Удалить кадры из маски", self)
        a.triggered.connect(self.remove_selected_frames_from_current_mask)
        m.addAction(a)
        m.exec_(self.mask_frames_list.mapToGlobal(pos))

    def delete_mask(self, row):
        if 0 <= row < len(self.masks):
            self.masks.pop(row)
            if self.current_mask_index == row:
                self.current_mask_index = None
            elif self.current_mask_index is not None and self.current_mask_index > row:
                self.current_mask_index -= 1
            self.refresh_mask_list()
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
        for fid in sorted(self.masks[self.current_mask_index]['frame_ids']):
            self.mask_frames_list.addItem(QListWidgetItem(fid))

    def toggle_current_mask_frame(self, frame_id):
        if self.current_mask_index is None:
            return
        s = self.masks[self.current_mask_index]['frame_ids']
        if frame_id in s:
            s.remove(frame_id)
        else:
            s.add(frame_id)
        self.refresh_current_mask_frames_list()
        self.refresh_timeline()

    def remove_selected_frames_from_current_mask(self):
        if self.current_mask_index is None:
            return
        s = self.masks[self.current_mask_index]['frame_ids']
        for it in self.mask_frames_list.selectedItems():
            s.discard(it.text())
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

        m = self.masks[self.current_mask_index]
        mask = m.get('mask')
        if mask is None:
            h, w = self.current_preview_frame.shape[:2]
            m['mask'] = np.zeros((h, w), dtype=np.uint8)
        elif mask.shape[:2] != self.current_preview_frame.shape[:2]:
            m['mask'] = cv2.resize(mask, (self.current_preview_frame.shape[1], self.current_preview_frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        self.preview.edit_mode = True
        self.preview.mask = m['mask']
        self.preview._render(True)
        self.set_editor_controls_for_mode("mask_edit")

    def finish_mask_edit_mode(self):
        self.mode_mask_edit = False
        self.preview.edit_mode = False
        self.set_editor_controls_for_mode(None)
        if self.current_preview_frame is not None:
            self.set_preview(self.current_preview_frame, self.current_preview_frame_id)

    def start_mask_add_mode(self):
        if self.current_mask_index is None:
            return
        self.mode_mask_add_frames = True
        self.mode_mask_edit = False
        self.batch_index_btns_wrap.setVisible(True)
        self.set_editor_controls_for_mode("mask_add")
        self.refresh_timeline()

    def finish_mask_add_mode(self):
        self.mode_mask_add_frames = False
        self.batch_index_btns_wrap.setVisible(False)
        self.set_editor_controls_for_mode(None)
        self.refresh_timeline()

    def set_editor_controls_for_mode(self, mode):
        full_lock = [
            self.batch_list, self.btn_new_mask, self.btn_run_inpaint,
            self.explorer_tree, self.btn_export, self.fps_spin,
            self.btn_play
        ]
        if mode == "mask_edit":
            for w in full_lock + [self.scroll, self.mask_list, self.mask_frames_list, self.batch_index_btns_wrap]:
                w.setEnabled(False)
            self.preview.setEnabled(True)
        elif mode == "mask_add":
            for w in full_lock + [self.mask_list, self.mask_frames_list]:
                w.setEnabled(False)
            self.scroll.setEnabled(True)
            self.batch_index_btns_wrap.setEnabled(True)
            self.preview.setEnabled(True)
        else:
            for w in full_lock + [self.scroll, self.mask_list, self.mask_frames_list, self.batch_index_btns_wrap]:
                w.setEnabled(True)
            self.preview.setEnabled(True)

    def cleanup_masks_after_frame_changes(self):
        valid = set(self.frame_lookup.keys())
        for m in self.masks:
            m['frame_ids'] = {fid for fid in m['frame_ids'] if fid in valid}

    def run_inpaint(self):
        selected_masks = [m for m in self.masks if m.get('selected', True)]
        if not selected_masks:
            QMessageBox.information(self, "Inpaint", "Нет выбранных масок")
            return

        masks_to_apply = [m for m in selected_masks if m.get('mask') is not None and len(m.get('frame_ids', [])) > 0]
        if not masks_to_apply:
            QMessageBox.information(self, "Inpaint", "Нет выбранных кадров с нарисованной маской")
            return

        self.global_progress.setValue(0)

        def on_progress(done, total):
            self.global_progress.setValue(int((done / total) * 100) if total else 0)

        try:
            self.processor.inpaint_project_frames(self.project.batches, masks_to_apply, progress_cb=on_progress)
        except Exception as e:
            QMessageBox.critical(self, "Inpaint error", str(e))
            return

        self.refresh_timeline()
        self.invalidate_prerender()

    def export_video(self):
        frames = []
        for b in self.project.batches:
            frames.extend(b.frames)
        if not frames:
            return

        pipeline = self.process_chain
        save_path, _ = QFileDialog.getSaveFileName(self, "Экспорт", "output.mp4", "Video (*.mp4)")
        if not save_path:
            return

        self.global_progress.setValue(0)
        final_frames = self.processor.process_sequence(frames, pipeline)
        h, w, _ = final_frames[0].shape
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), float(self.fps_spin.value()), (w, h))
        total = len(final_frames)
        for i, f in enumerate(final_frames, 1):
            out.write(f)
            self.global_progress.setValue(int((i / total) * 100))
        out.release()
