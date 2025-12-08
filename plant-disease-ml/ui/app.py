# ui/app.py
import sys, argparse, os, json
from typing import Optional, Iterable, Set, Tuple, List
from PyQt5 import QtWidgets, QtGui, QtCore

HERE = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(HERE, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from plant_disease.inference import InferenceEngine  


import cv2
import numpy as np
from PIL import Image


# Helpers
def load_class_names(path: Optional[str]):
    if not path:
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # Try JSON first
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                keys = sorted(obj.keys(), key=lambda k: int(k))
                return [obj[k] for k in keys]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        # Fallback: YAML
        import yaml
        y = yaml.safe_load(txt)
        names = y.get('names')
        if isinstance(names, dict):
            keys = sorted(names.keys(), key=lambda k: int(k))
            return [names[k] for k in keys]
        return names
    except Exception:
        return None


def make_allowed_ids(class_names: Optional[Iterable[str]], keep_names_csv: str) -> Optional[Set[int]]:
    if not class_names:
        return None
    keep = [s.strip().lower() for s in str(keep_names_csv).split(",") if s.strip()]
    if not keep:
        return None
    return {i for i, n in enumerate(class_names) if str(n).lower() in keep}


def is_healthy_prediction(selected_labels: List[str]) -> bool:
    if not selected_labels:
        return False
    return any("healthy" in str(lbl).lower() for lbl in selected_labels)


def detect_leaf_bboxes_bgr(bgr, min_area_frac=0.01, min_aspect=0.2, max_aspect=5.0, downscale=320):

    H, W = bgr.shape[:2]
    scale = 1.0
    small = bgr
    m = max(H, W)
    if m > downscale:
        scale = m / float(downscale)
        new_w, new_h = int(round(W/scale)), int(round(H/scale))
        small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Tweakable green range
    lower = np.array([25, 40, 40], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    area_thresh = min_area_frac * (mask.shape[0] * mask.shape[1])
    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_thresh:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(1, h)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        # scale back to original coords
        x1 = int(round(x * scale)); y1 = int(round(y * scale))
        x2 = int(round((x + w) * scale)); y2 = int(round((y + h) * scale))
        # clamp
        x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2)); y2 = max(0, min(H-1, y2))
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    return boxes


def detect_leaf_bbox(pil_img, min_area_frac=0.02) -> Tuple[Image.Image, Optional[Tuple[int,int,int,int]]]:
    rgb = np.array(pil_img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxes = detect_leaf_bboxes_bgr(bgr, min_area_frac=min_area_frac)
    if not boxes:
        return pil_img, None
    x1, y1, x2, y2 = boxes[0]
    crop = rgb[y1:y2, x1:x2]
    return Image.fromarray(crop), (x1, y1, x2, y2)


def draw_bbox_on_image(pil_img, bbox, color_bgr=(0, 255, 0), thickness=3) -> Image.Image:
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(bgr, (x1, y1), (x2, y2), color_bgr, thickness)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# Worker
class WorkerSignals(QtCore.QObject):
    result = QtCore.pyqtSignal(dict, list, dict)  # results, selected, meta
    error = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()


class InferenceTask(QtCore.QRunnable):
    def __init__(self, engine: InferenceEngine, pil_img: Image.Image):
        super().__init__()
        self.engine = engine
        self.pil_img = pil_img
        self.signals = WorkerSignals()

    @QtCore.pyqtSlot()
    def run(self):
        try:
            res, sel, meta = self.engine.predict_pil(self.pil_img)
            self.signals.result.emit(res, sel, meta)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


# Camera Widget
class CameraWidget(QtWidgets.QWidget):
    frameCaptured = QtCore.pyqtSignal(object, object)  # (np.ndarray BGR, bbox tuple|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.preview = QtWidgets.QLabel("Camera preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(360)
        self.preview.setStyleSheet("border: 2px dashed #8a8a8a;")

        self.idx_spin = QtWidgets.QSpinBox(); self.idx_spin.setRange(0, 5); self.idx_spin.setValue(0)
        self.idx_spin.setToolTip("Camera index")

        self.start_btn = QtWidgets.QPushButton("Start Camera")
        self.stop_btn = QtWidgets.QPushButton("Stop Camera")
        self.capture_btn = QtWidgets.QPushButton("Capture & Diagnose")
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)
        self.live_boxes_chk = QtWidgets.QCheckBox("Live boxes"); self.live_boxes_chk.setChecked(True)

        self._last_bboxes: List[Tuple] = []
        self._last_frame: Optional[np.ndarray] = None
        self.detector = None            # set by MainWindow
        self.det_allowed_ids = None 

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(QtWidgets.QLabel("Index:"))
        btns.addWidget(self.idx_spin)
        btns.addStretch(1)
        btns.addWidget(self.live_boxes_chk)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.capture_btn)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.preview)
        layout.addLayout(btns)

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._grab)

        self.start_btn.clicked.connect(lambda: self.start(self.idx_spin.value()))
        self.stop_btn.clicked.connect(self.stop)
        self.capture_btn.clicked.connect(self.capture)

    def start(self, index=0):
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(int(index), api)
            if cap and cap.isOpened():
                self.cap = cap
                break
            if cap: cap.release()
        if not getattr(self, "cap", None):
            QtWidgets.QMessageBox.warning(self, "Camera", "Could not open camera. Try a different index.")
            return

        self.timer.start(33)  # ~30 FPS
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.capture_btn.setEnabled(True)

    def stop(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.preview.setText("Camera preview")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)

    def _best_bbox(self) -> Optional[Tuple[int,int,int,int]]:
        if not self._last_bboxes:
            return None
        def area(b):
            x1,y1,x2,y2 = b[:4]
            return (x2-x1)*(y2-y1)
        b = max(self._last_bboxes, key=area)
        return tuple(b[:4])

    def _grab(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        disp = frame
        if self.live_boxes_chk.isChecked():
            if self.detector is not None:
                boxes = self.detector.detect_bgr(frame)
                if self.det_allowed_ids is not None:
                    boxes = [b for b in boxes if b[5] in self.det_allowed_ids]
                self._last_bboxes = boxes
                if boxes:
                    disp = frame.copy()
                    for (x1,y1,x2,y2,conf,cls) in boxes:
                        # Yellow preview
                        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2)
                        cv2.putText(disp, f"{conf:.2f}", (x1, max(0,y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
            else:
                boxes = detect_leaf_bboxes_bgr(frame, min_area_frac=0.01)
                self._last_bboxes = boxes
                if boxes:
                    disp = frame.copy()
                    for (x1,y1,x2,y2) in boxes:
                        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,255,255), 2)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch*w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.preview.setPixmap(pix.scaled(
            self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        ))
        self._last_frame = frame 

    def capture(self):
        if self._last_frame is not None:
            self.frameCaptured.emit(self._last_frame.copy(), self._best_bbox())


# Main
class MainWindow(QtWidgets.QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("Plant Disease App")
        self.resize(1150, 720)
        self.threadpool = QtCore.QThreadPool()
        self.args = args

        self.detector_names = load_class_names(args.det_names)
        self.det_allowed_ids = make_allowed_ids(self.detector_names, args.det_classes)

        # Classifier
        self.engine = InferenceEngine(
            args.weights, args.label_map, model_name=args.model_name, device=args.device,
            img_size=args.img_size, threshold=args.threshold, temperature=args.temperature,
            multiclass=(str(args.multiclass).lower() in ["1","true","yes","y"])
        )

        # Detector (ONNX)
        self.detector = None
        if args.detector_onnx:
            try:
                from plant_disease.detect.yolo_onnx import YOLOv8ONNX
                self.detector = YOLOv8ONNX(
                    args.detector_onnx, conf_thr=args.det_conf, iou_thr=args.det_iou,
                    input_size=args.det_input
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Detector", f"Failed to load detector: {e}")

        # Styles
        self.setStyleSheet("""
            QWidget { font-family: Segoe UI, Arial; font-size: 11pt; color: #202124; }
            QFrame#Card { background: #ffffff; border: 1px solid #e6e6e6; border-radius: 12px; }
            QLabel#Title { font-size: 18pt; font-weight: 700; }
            QPushButton { padding: 8px 14px; border-radius: 8px; }
            QPushButton:hover { background: #f1f3f4; }
            QPushButton:disabled { color: #9aa0a6; }
            QProgressBar { border: 1px solid #e6e6e6; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background-color: #66bb6a; border-radius: 6px; }
            QTabWidget::pane { border: 0; }
            QTabBar::tab { padding: 8px 16px; margin: 2px; border-radius: 8px; }
            QTabBar::tab:selected { background: #e8f5e9; font-weight: 600; }
            QTabBar::tab:!selected { background: #f8f9fa; }
            QListWidget { border: 1px solid #e6e6e6; border-radius: 8px; }
        """)

        # Header controls
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(" Plant Disease App"); title.setObjectName("Title")
        header.addWidget(title); header.addStretch(1)

        self.alpha_spin = QtWidgets.QDoubleSpinBox(); self.alpha_spin.setDecimals(2); self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setRange(0.05, 0.99); self.alpha_spin.setValue(self.engine.unknown_maxp)
        header.addWidget(QtWidgets.QLabel("Unknown α (max p):")); header.addWidget(self.alpha_spin)

        self.beta_spin = QtWidgets.QDoubleSpinBox(); self.beta_spin.setDecimals(2); self.beta_spin.setSingleStep(0.1)
        self.beta_spin.setRange(0.1, 4.0); self.beta_spin.setValue(self.engine.unknown_entropy)
        header.addWidget(QtWidgets.QLabel("β (entropy):")); header.addWidget(self.beta_spin)

        self.temp_spin = QtWidgets.QDoubleSpinBox(); self.temp_spin.setDecimals(2); self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setRange(0.1, 10.0); self.temp_spin.setValue(self.engine.temperature)
        header.addWidget(QtWidgets.QLabel("Temp:")); header.addWidget(self.temp_spin)

        self.explain_chk = QtWidgets.QCheckBox("Explain (Grad-CAM)")
        self.save_uncertain_chk = QtWidgets.QCheckBox("Save uncertain"); self.save_uncertain_chk.setChecked(True)
        header.addWidget(self.explain_chk); header.addWidget(self.save_uncertain_chk)

        # Tabs (Image / Camera)
        self.tabs = QtWidgets.QTabWidget()
        self._build_image_tab()
        self._build_camera_tab()  # creates self.camera

        # Pass detector to camera
        self.camera.detector = self.detector
        self.camera.det_allowed_ids = self.det_allowed_ids

        left_card = QtWidgets.QFrame(); left_card.setObjectName("Card")
        lc = QtWidgets.QVBoxLayout(left_card); lc.addWidget(self.tabs)

        # Results card
        right_card = QtWidgets.QFrame(); right_card.setObjectName("Card")
        rc = QtWidgets.QVBoxLayout(right_card)

        self.result_title = QtWidgets.QLabel("Result"); self.result_title.setStyleSheet("font-weight: 600;")
        rc.addWidget(self.result_title)

        self.top_label = QtWidgets.QLabel("—")
        self.conf_bar = QtWidgets.QProgressBar(); self.conf_bar.setRange(0,100); self.conf_bar.setValue(0)
        rc.addWidget(self.top_label); rc.addWidget(self.conf_bar)

        self.unknown_label = QtWidgets.QLabel(""); rc.addWidget(self.unknown_label)
        self.severity_label = QtWidgets.QLabel(""); rc.addWidget(self.severity_label)

        self.prob_list = QtWidgets.QListWidget(); rc.addWidget(self.prob_list)

        body = QtWidgets.QHBoxLayout()
        body.addWidget(left_card, 2); body.addWidget(right_card, 1)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(header); root.addLayout(body)

        # Bindings
        self.alpha_spin.valueChanged.connect(self._sync_unknown_params)
        self.beta_spin.valueChanged.connect(self._sync_unknown_params)
        self.temp_spin.valueChanged.connect(self._sync_temperature)

        # State
        self.current_image_path: Optional[str] = None
        self._last_pil: Optional[Image.Image] = None
        self._last_bbox: Optional[Tuple[int,int,int,int]] = None
        self._last_crop: Optional[Image.Image] = None
        self._active_task = None

    # Tabs
    def _build_image_tab(self):
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)

        self.image_preview = QtWidgets.QLabel("Drop an image or click 'Open Image'")
        self.image_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.image_preview.setMinimumHeight(420)
        self.image_preview.setStyleSheet("border: 2px dashed #8a8a8a;")
        self.image_preview.setAcceptDrops(True)

        page.setAcceptDrops(True)
        open_btn = QtWidgets.QPushButton("Open Image"); open_btn.clicked.connect(self._open_image)
        self.predict_btn = QtWidgets.QPushButton("Diagnose"); self.predict_btn.clicked.connect(self._diagnose_current); self.predict_btn.setEnabled(False)

        btns = QtWidgets.QHBoxLayout(); btns.addWidget(open_btn); btns.addWidget(self.predict_btn); btns.addStretch(1)

        v.addWidget(self.image_preview); v.addLayout(btns)

        def dragEnterEvent(event):
            if event.mimeData().hasUrls(): event.acceptProposedAction()
        def dropEvent(event):
            urls = event.mimeData().urls()
            if urls:
                self._load_image(urls[0].toLocalFile())
        page.dragEnterEvent = dragEnterEvent
        page.dropEvent = dropEvent

        self.tabs.addTab(page, "Image")

    def _build_camera_tab(self):
        self.camera = CameraWidget()
        self.camera.frameCaptured.connect(self._diagnose_frame)  
        self.tabs.addTab(self.camera, "Live Camera")

    # Settings
    def _sync_unknown_params(self):
        self.engine.unknown_maxp = float(self.alpha_spin.value())
        self.engine.unknown_entropy = float(self.beta_spin.value())

    def _sync_temperature(self):
        self.engine.temperature = float(self.temp_spin.value())

    # Loading image
    def _open_image(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if path: self._load_image(path)

    def _load_image(self, path: str):
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            QtWidgets.QMessageBox.warning(self, "Error", "Failed to load image.")
            return
        self.current_image_path = path
        self.image_preview.setPixmap(pix.scaled(self.image_preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.predict_btn.setEnabled(True)
        # Reset
        self.top_label.setText("—"); self.conf_bar.setValue(0)
        self.unknown_label.setText(""); self.severity_label.setText(""); self.prob_list.clear()

    # Inference
    def _diagnose_current(self):
        if not self.current_image_path:
            return
        with Image.open(self.current_image_path) as im:
            pil = im.convert("RGB").copy()
        self._run_inference(pil)  

    def _diagnose_frame(self, bgr: np.ndarray, live_bbox=None):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._run_inference(pil, forced_bbox=live_bbox)

    def _choose_bbox(self, pil_img: Image.Image) -> Optional[Tuple[int,int,int,int]]:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        bbox = None
        if self.detector is not None:
            det_boxes = self.detector.detect_bgr(bgr)  
            if self.det_allowed_ids is not None:
                det_boxes = [b for b in det_boxes if b[5] in self.det_allowed_ids]
            if det_boxes:
                det_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                x1,y1,x2,y2,conf,cls = det_boxes[0]
                bbox = (x1,y1,x2,y2)
        if bbox is None:
            _, bbox = detect_leaf_bbox(pil_img)
        return bbox

    def _run_inference(self, pil_img: Image.Image, forced_bbox: Optional[Tuple[int,int,int,int]] = None):
        # Reset UI
        self.prob_list.clear()
        self.top_label.setText("—")
        self.conf_bar.setValue(0)
        self.unknown_label.setText("")
        self.severity_label.setText("")


        bbox = forced_bbox if forced_bbox is not None else self._choose_bbox(pil_img)
        self._last_pil = pil_img
        self._last_bbox = bbox


        crop_pil = pil_img if bbox is None else pil_img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        self._last_crop = crop_pil

        task = InferenceTask(self.engine, crop_pil)
        task.signals.result.connect(self._show_results)
        task.signals.error.connect(self._show_error)
        task.signals.finished.connect(self._on_task_finished)
        self._active_task = task

        self._set_busy(True)
        self.threadpool.start(task)

    def _set_busy(self, busy: bool):
        if busy:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.predict_btn.setEnabled(not busy and self.current_image_path is not None)
        self.camera.capture_btn.setEnabled(not busy)

    def _on_task_finished(self):
        self._active_task = None
        self._set_busy(False)
        self.prob_list.viewport().update()

    # Results UI
    def _show_results(self, results: dict, selected: list, meta: dict):
        
        top = max(results.items(), key=lambda kv: kv[1])
        self.top_label.setText(f"Top-1: {top[0]}")
        self.conf_bar.setValue(int(round(top[1] * 100)))
        self.unknown_label.setText("⚠️ Unknown / Low confidence" if meta.get("unknown", False) else "")

        base_img = self._last_pil

        # Grad-CAM overlay
        if self.explain_chk.isChecked() and self._last_pil is not None:
            if self._last_bbox is not None and self._last_crop is not None:
                heat = self.engine.grad_cam(self._last_crop, target_class=None)
                if heat is not None:
                    heat_u8 = np.array(heat)
                    if heat_u8.dtype != np.uint8:
                        heat_u8 = cv2.normalize(heat_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    if heat_u8.ndim == 3:
                        heat_u8 = cv2.cvtColor(heat_u8, cv2.COLOR_BGR2GRAY)
                    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

                    crop_rgb = np.array(self._last_crop)
                    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                    overlay = cv2.addWeighted(heat_color, 0.45, crop_bgr, 0.55, 0)

                    full_bgr = cv2.cvtColor(np.array(self._last_pil), cv2.COLOR_RGB2BGR)
                    x1,y1,x2,y2 = self._last_bbox
                    if overlay.shape[:2] != (y2-y1, x2-x1):
                        overlay = cv2.resize(overlay, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
                    full_bgr[y1:y2, x1:x2] = overlay
                    base_img = Image.fromarray(cv2.cvtColor(full_bgr, cv2.COLOR_BGR2RGB))

                    try:
                        lvl, cov = self.engine.severity_from_cam(heat_u8)
                        self.severity_label.setText(f"Severity: {lvl} (~{cov*100:.1f}%)")
                    except Exception:
                        pass
                else:
                    self.severity_label.setText("")
            else:
                # Grad-CAM fallback
                heat = self.engine.grad_cam(self._last_pil, target_class=None)
                if heat is not None:
                    heat_u8 = np.array(heat)
                    if heat_u8.dtype != np.uint8:
                        heat_u8 = cv2.normalize(heat_u8, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    if heat_u8.ndim == 3:
                        heat_u8 = cv2.cvtColor(heat_u8, cv2.COLOR_BGR2GRAY)
                    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                    rgb = np.array(self._last_pil)
                    overlay = cv2.addWeighted(heat_color, 0.45, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, 0)
                    base_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                else:
                    self.severity_label.setText("")

        # Box colour
        healthy = is_healthy_prediction(selected)
        color_bgr = (0,255,0) if healthy else (0,0,255)  # green / red

        # Draw the final box
        if self._last_bbox is None:
            H, W = base_img.size[1], base_img.size[0]
            bbox = (2, 2, W-3, H-3)
        else:
            bbox = self._last_bbox
        boxed = draw_bbox_on_image(base_img, bbox, color_bgr=color_bgr, thickness=3)

        # Show result image in active tab
        qimg = QtGui.QImage(boxed.tobytes("raw", "RGB"), boxed.size[0], boxed.size[1],
                            boxed.size[0]*3, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        target_label = self.image_preview if self.tabs.currentIndex()==0 else self.camera.preview
        target_label.setPixmap(pix.scaled(
            target_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        ))

        # probabilities
        self.prob_list.clear()
        self.prob_list.setSpacing(6)
        self.prob_list.setUniformItemSizes(True)
        for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(8, 2, 8, 2)
            name = QtWidgets.QLabel(k); name.setMinimumWidth(260); name.setStyleSheet("font-weight: 500;")
            bar = QtWidgets.QProgressBar(); bar.setRange(0,100); bar.setValue(int(round(v*100))); bar.setFormat(f"{v*100:.1f}%")
            h.addWidget(name); h.addWidget(bar)
            item = QtWidgets.QListWidgetItem(); item.setSizeHint(row.sizeHint())
            self.prob_list.addItem(item); self.prob_list.setItemWidget(item, row)

        self.predict_btn.setEnabled(True)
        self.camera.capture_btn.setEnabled(True)

    def _show_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Inference Error", msg)
        self.predict_btn.setEnabled(True)
        self.camera.capture_btn.setEnabled(True)


# Entry
def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--label-map", required=True)
    ap.add_argument("--model-name", default="efficientnet_b0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--multiclass", type=str, default="true") 
    # Detector args
    ap.add_argument("--detector-onnx", default=None, help="Path to YOLOv8 ONNX detector")
    ap.add_argument("--det-names", default=None, help="Path to class names file (JSON or YAML)")
    ap.add_argument("--det-classes", default="leaf", help="Comma-separated class names to KEEP (default: leaf)")
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou",  type=float, default=0.45)
    ap.add_argument("--det-input", type=int, default=480)
    return ap


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    args = build_arg_parser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(args)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
