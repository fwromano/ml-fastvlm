#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageTk


# Runtime selection of device happens inside ultralytics via torch; we avoid torch import here
DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolov8s-seg.pt")
DEFAULT_VIDEO = os.environ.get("VIDEO_PATH", "/Users/agc/Documents/output.mp4")


def _ensure_rgb(pil: Image.Image) -> Image.Image:
    if pil.mode != "RGB":
        return pil.convert("RGB")
    return pil


def _color_for_class(cls_id: int) -> Tuple[int, int, int, int]:
    # RGBA colors with alpha for overlay
    # person=0 -> red, car=2 -> blue, others -> green
    if cls_id == 0:
        return (255, 50, 50, 110)
    if cls_id == 2:
        return (50, 120, 255, 110)
    return (60, 200, 80, 90)


def _draw_overlay(frame_rgb: np.ndarray, boxes: np.ndarray, classes: List[int], confs: List[float], masks: np.ndarray, names: List[str]) -> np.ndarray:
    h, w, _ = frame_rgb.shape
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ol = np.array(overlay)

    if masks is not None and len(masks) > 0:
        # masks: (N, h, w) boolean/float
        for i, m in enumerate(masks):
            cls_id = int(classes[i])
            color = np.array(_color_for_class(cls_id), dtype=np.uint8)  # RGBA
            if m.shape != (h, w):
                # Resize to match frame size
                m_resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                m = m_resized
            if m.dtype != np.uint8:
                m_bin = (m > 0.5).astype(np.uint8)
            else:
                m_bin = (m > 0).astype(np.uint8)
            # Broadcast to RGBA
            mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            mask_rgba[m_bin == 1] = color
            ol = cv2.add(ol, mask_rgba)

    out = Image.fromarray(frame_rgb).convert("RGBA")
    out = Image.alpha_composite(out, Image.fromarray(ol))
    out_bgr = cv2.cvtColor(np.array(out.convert("RGB")), cv2.COLOR_RGB2BGR)

    # Draw boxes and labels on top (use OpenCV for speed)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(classes[i])
        name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)
        conf = confs[i]
        color_rgb = _color_for_class(cls_id)[:3]
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        cv2.rectangle(out_bgr, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(out_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out_bgr


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Segmentation: Video Detector")
        self.geometry("980x760")

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(frm, textvariable=self.status_var, foreground="#555").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))

        ttk.Label(frm, text="Video:").grid(row=1, column=0, sticky=tk.W)
        self.video_var = tk.StringVar(value=DEFAULT_VIDEO)
        self.video_entry = ttk.Entry(frm, textvariable=self.video_var, width=70)
        self.video_entry.grid(row=1, column=1, sticky=tk.EW, padx=(4, 4))
        ttk.Button(frm, text="Browse", command=self.browse_video).grid(row=1, column=2)

        ttk.Label(frm, text="Model:").grid(row=2, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_entry = ttk.Entry(frm, textvariable=self.model_var, width=70)
        self.model_entry.grid(row=2, column=1, sticky=tk.EW, padx=(4, 4))
        ttk.Button(frm, text="Change", command=self.change_model).grid(row=2, column=2)

        # Controls
        btns = ttk.Frame(frm)
        btns.grid(row=3, column=1, pady=(8, 8))
        self.play_btn = ttk.Button(btns, text="Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=4)
        self.inf_btn = ttk.Button(btns, text="Start Detection", command=self.toggle_infer)
        self.inf_btn.grid(row=0, column=1, padx=4)

        # Settings
        self.person_var = tk.BooleanVar(value=True)
        self.car_var = tk.BooleanVar(value=True)
        self.conf_var = tk.DoubleVar(value=0.25)
        opt = ttk.Frame(frm)
        opt.grid(row=4, column=1, pady=(2, 8))
        ttk.Checkbutton(opt, text="Person", variable=self.person_var).grid(row=0, column=0, padx=6)
        ttk.Checkbutton(opt, text="Car", variable=self.car_var).grid(row=0, column=1, padx=6)
        ttk.Label(opt, text="Conf:").grid(row=0, column=2)
        ttk.Scale(opt, from_=0.1, to=0.8, variable=self.conf_var, orient=tk.HORIZONTAL, length=180).grid(row=0, column=3, padx=6)

        # Canvas
        ttk.Label(frm, text="Current frame:").grid(row=5, column=0, sticky=tk.NW)
        self.preview_label = ttk.Label(frm)
        self.preview_label.grid(row=5, column=1, sticky=tk.NSEW)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)

        self.cap = None
        self.playing = False
        self.infer_on = False
        self.infer_busy = False
        self.current_frame = None
        self.model = None
        self.names = []
        self.last_det_image = None  # PIL.Image with overlays

        # Preload video
        if os.path.isfile(self.video_var.get()):
            self._open_capture(self.video_var.get())

        # Async model load
        self._load_model_async(self.model_var.get())

    def browse_video(self):
        p = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        if p:
            self.video_var.set(p)
            self._open_capture(p)

    def change_model(self):
        p = filedialog.askopenfilename(title="Select YOLO weights or enter hub id", filetypes=[("Model", "*.pt *.onnx *.engine"), ("All files", "*.*")])
        if p:
            self.model_var.set(p)
            self._load_model_async(p)

    def toggle_play(self):
        if self.cap is None:
            path = self.video_var.get().strip()
            if not os.path.isfile(path):
                messagebox.showerror("Error", f"Video not found:\n{path}")
                return
            self._open_capture(path)
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._video_loop()

    def toggle_infer(self):
        self.infer_on = not self.infer_on
        self.inf_btn.configure(text="Stop Detection" if self.infer_on else "Start Detection")
        if not self.infer_on:
            self.last_det_image = None
        if self.infer_on and not self.infer_busy and self.current_frame is not None and self.model is not None:
            self._run_detection_on_current()

    def _open_capture(self, path: str):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Error", f"Cannot open video: {path}")
            return
        self.status_var.set("Ready • video loaded")
        self.last_det_image = None

    def _video_loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = Image.fromarray(rgb)
            # If detection is ON and model is ready, run synchronously for reliable overlays
            if self.infer_on and self.model is not None:
                try:
                    disp = self._detect_sync(self.current_frame)
                    self.last_det_image = disp
                except Exception as _:
                    # Fall back to raw frame if detection fails
                    disp = self.current_frame
            else:
                # Prefer showing the last detection overlay if available
                disp = self.last_det_image if (self.infer_on and self.last_det_image is not None) else self.current_frame
            pw, ph = disp.size
            s = min(900 / max(pw, 1), 600 / max(ph, 1), 1.0)
            show = disp.resize((int(pw * s), int(ph * s)))
            photo = ImageTk.PhotoImage(show)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        if self.playing:
            # Schedule next frame with a small delay to keep UI responsive (~60 FPS)
            self.after(16, self._video_loop)

    def _load_model_async(self, weights: str):
        self.status_var.set("Loading YOLO model...")
        def worker():
            try:
                # Local import to let launcher manage dependencies
                from ultralytics import YOLO
                model = YOLO(weights)
                names = model.names if hasattr(model, 'names') else []
            except Exception as e:
                err = str(e)
                def fail():
                    self.status_var.set("Error loading model")
                    messagebox.showerror("Model load failed", err)
                self.after(0, fail)
                return
            def done():
                self.model = model
                self.names = [names[k] for k in sorted(names.keys())] if isinstance(names, dict) else list(names)
                if not self.names:
                    # Fallback common COCO order
                    self.names = [
                        'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
                        'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
                        'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
                        'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
                        'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
                        'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
                        'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
                        'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
                    ]
                self.status_var.set("Ready • model loaded")
            self.after(0, done)
        threading.Thread(target=worker, daemon=True).start()

    def _detect_sync(self, pil_img: Image.Image) -> Image.Image:
        """Run YOLO predict synchronously on the given image and return overlaid PIL image."""
        conf = float(self.conf_var.get())
        want_person = self.person_var.get()
        want_car = self.car_var.get()

        pil_img = _ensure_rgb(pil_img)
        im_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        res_list = self.model.predict(source=im_bgr, conf=conf, retina_masks=True, verbose=False)
        if not res_list:
            return pil_img
        res = res_list[0]
        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        clses = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        masks = None
        if getattr(res, 'masks', None) is not None and res.masks is not None and res.masks.data is not None:
            masks = res.masks.data.cpu().numpy()  # (N, h, w)

        keep = []
        for i, c in enumerate(clses):
            if (c == 0 and want_person) or (c == 2 and want_car):
                keep.append(i)
        if keep:
            boxes_xyxy = boxes_xyxy[keep]
            clses = clses[keep]
            confs = confs[keep]
            if masks is not None:
                masks = masks[keep]
        else:
            # If nothing matched filters but detections exist, show all to verify pipeline
            if boxes_xyxy.shape[0] == 0:
                return pil_img

        out_bgr = _draw_overlay(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), boxes_xyxy, clses.tolist(), confs.tolist(), masks, self.names)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)


if __name__ == "__main__":
    App().mainloop()
