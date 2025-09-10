#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import torch

# Ensure local 'llava' package is importable without pip installing the repo
import sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# FastVLM (LLaVA) imports for visual captioning
try:
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.conversation import conv_templates
    _HAS_VLM = True
except Exception:
    _HAS_VLM = False


# Runtime selection for YOLO happens inside ultralytics/torch
DEFAULT_MODEL = os.environ.get("YOLO_MODEL", "yolov8s-seg.pt")
DEFAULT_VIDEO = os.environ.get("VIDEO_PATH", "/Users/agc/Documents/output.mp4")
DEFAULT_VLM_DIR = os.environ.get("MODEL_DIR", "checkpoints/llava-fastvithd_1.5b_stage3")
# Prefer CPU for VLM by default on macOS to avoid MPS matmul crashes
VLM_FORCE_CPU = os.environ.get("VLM_FORCE_CPU", "1") not in ("0", "false", "False")

def _pick_device() -> str:
    try:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


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


def _draw_overlay(frame_rgb: np.ndarray, boxes: np.ndarray, classes: List[int], confs: List[float], masks: np.ndarray, names: List[str], captions: List[str] | None = None) -> np.ndarray:
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
        cap = None if captions is None or i >= len(captions) else captions[i]
        # Only append caption if it's non-empty and not a placeholder
        if cap and cap != 'pending':
            label = f"{name} {conf:.2f} - {cap}"
        else:
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
        # Side panel for VLM outputs
        ttk.Label(frm, text="VLM descriptions:").grid(row=5, column=3, sticky=tk.NW, padx=(10,0))
        self.captions_text = tk.Text(frm, height=30, width=40, wrap='word')
        self.captions_text.grid(row=5, column=4, sticky=tk.NSEW)

        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(4, weight=1)
        frm.rowconfigure(5, weight=1)

        self.cap = None
        self.playing = False
        self.infer_on = False
        self.infer_busy = False
        self.current_frame = None
        self.model = None
        self.names = []
        self.last_det_image = None  # PIL.Image with overlays
        # VLM components
        self.vlm = None  # dict(tokenizer, model, proc)
        self.vlm_device = 'cpu' if VLM_FORCE_CPU else _pick_device()
        self.vlm_cache = {}  # key -> (caption, timestamp)
        self.vlm_sem = threading.Semaphore(1)
        self.vlm_lock = threading.Lock()
        self.tracks = []  # [{id, cls, box}]
        self.next_track_id = 1

        # Preload video
        if os.path.isfile(self.video_var.get()):
            self._open_capture(self.video_var.get())

        # Async model load
        self._load_model_async(self.model_var.get())

        # VLM controls
        ttk.Label(frm, text="VLM dir:").grid(row=2, column=3, sticky=tk.W, padx=(10,0))
        self.vlm_var = tk.StringVar(value=DEFAULT_VLM_DIR)
        self.vlm_entry = ttk.Entry(frm, textvariable=self.vlm_var, width=50)
        self.vlm_entry.grid(row=2, column=4, sticky=tk.EW, padx=(4,4))
        ttk.Button(frm, text="Load VLM", command=self.reload_vlm).grid(row=2, column=5)
        frm.columnconfigure(4, weight=1)

        if _HAS_VLM and os.path.isdir(DEFAULT_VLM_DIR):
            # Use CPU by default to be robust on macOS
            self._load_vlm_async(DEFAULT_VLM_DIR, force_device=('cpu' if VLM_FORCE_CPU else None))

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
        # If turning on and we have a frame + model, run one detection immediately to show overlays
        if self.infer_on and self.current_frame is not None and self.model is not None:
            try:
                disp = self._detect_sync(self.current_frame)
                self.last_det_image = disp
                pw, ph = disp.size
                s = min(900 / max(pw, 1), 600 / max(ph, 1), 1.0)
                show = disp.resize((int(pw * s), int(ph * s)))
                photo = ImageTk.PhotoImage(show)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo
            except Exception:
                pass

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

    def reload_vlm(self):
        path = self.vlm_var.get().strip()
        if not _HAS_VLM:
            messagebox.showerror("VLM unavailable", "llava is not installed in this environment.")
            return
        if not os.path.isdir(path):
            messagebox.showerror("Invalid path", f"VLM model directory not found:\n{path}")
            return
        self._load_vlm_async(path)

    def _load_vlm_async(self, model_dir: str, force_device: str | None = None):
        self.status_var.set("Loading VLM model…")
        def worker():
            try:
                dev = force_device or self.vlm_device
                model_name = get_model_name_from_path(model_dir)
                tokenizer, model, image_processor, _ = load_pretrained_model(model_dir, None, model_name, device=dev)
                model.eval()
                if hasattr(model.config, "mm_use_im_start_end"):
                    model.config.mm_use_im_start_end = False
            except Exception as e:
                err = str(e)
                def fail():
                    messagebox.showerror("VLM load failed", err)
                    self.status_var.set("Ready")
                self.after(0, fail)
                return
            def done():
                self.vlm = {"tokenizer": tokenizer, "model": model, "proc": image_processor}
                if force_device is not None:
                    self.vlm_device = force_device
                self.status_var.set("Ready • VLM loaded")
            self.after(0, done)
        threading.Thread(target=worker, daemon=True).start()

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

        # Filter for person/car
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
            if boxes_xyxy.shape[0] == 0:
                # Nothing to show
                self._update_captions_panel([], [], [], [])
                return pil_img

        # Associate detections to tracks (IoU) for persistence of captions
        def iou(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            aarea = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            barea = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = aarea + barea - inter
            return inter / union if union > 0 else 0.0

        matched_track_ids = []
        captions = [None] * len(boxes_xyxy)
        new_tracks = []
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
            box = (int(x1), int(y1), int(x2), int(y2))
            cls_i = int(clses[i])
            # find best track match by IoU for same class
            best_tid, best_iou = None, 0.0
            for tr in self.tracks:
                if tr['cls'] != cls_i:
                    continue
                ov = iou(box, tr['box'])
                if ov > best_iou:
                    best_iou, best_tid = ov, tr['id']
            if best_tid is not None and best_iou >= 0.3:
                tid = best_tid
                # update track box
                for tr in self.tracks:
                    if tr['id'] == tid:
                        tr['box'] = box
                        break
                matched_track_ids.append(tid)
            else:
                tid = self.next_track_id; self.next_track_id += 1
                new_tracks.append({'id': tid, 'cls': cls_i, 'box': box})
                matched_track_ids.append(tid)
            # caption fetch/enqueue
            cap = self._vlm_get_or_enqueue(pil_img, box, f"tid:{tid}")
            captions[i] = cap
        # keep only matched + new tracks
        keep_ids = set(matched_track_ids)
        self.tracks = [tr for tr in self.tracks if tr['id'] in keep_ids] + new_tracks

        out_bgr = _draw_overlay(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), boxes_xyxy, clses.tolist(), confs.tolist(), masks, self.names, captions=captions)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        # Update side panel
        try:
            self._update_captions_panel(boxes_xyxy, clses, confs, captions)
        except Exception:
            pass
        return Image.fromarray(out_rgb)

    def _update_captions_panel(self, boxes, clses, confs, captions):
        lines = []
        for i in range(len(captions)):
            name = self.names[int(clses[i])] if 0 <= int(clses[i]) < len(self.names) else str(int(clses[i]))
            cap = captions[i]
            if not cap:
                cap = 'pending'
            lines.append(f"- {name} ({confs[i]:.2f}): {cap}")
        text = "\n".join(lines) if lines else "(no detections)"
        self.captions_text.delete('1.0', tk.END)
        self.captions_text.insert(tk.END, text)

    def _vlm_get_or_enqueue(self, pil_img: Image.Image, box: tuple, key: str) -> str:
        now = time.time()
        val = self.vlm_cache.get(key)
        if val and (now - val[1] < 10.0):
            return val[0]
        if self.vlm is None or not _HAS_VLM:
            return '(VLM not loaded)'
        if self.vlm_sem.acquire(blocking=False):
            x1, y1, x2, y2 = box
            w, h = pil_img.size
            dx = int(0.05 * (x2 - x1 + 1)); dy = int(0.05 * (y2 - y1 + 1))
            x1e = max(0, x1 - dx); y1e = max(0, y1 - dy); x2e = min(w, x2 + dx); y2e = min(h, y2 + dy)
            crop = pil_img.crop((x1e, y1e, x2e, y2e)).convert("RGB")
            threading.Thread(target=self._vlm_caption_worker, args=(crop, key), daemon=True).start()
        return 'pending'

    def _vlm_caption_worker(self, crop: Image.Image, key: str):
        try:
            prompt = "Concisely describe this."
            tokenizer = self.vlm["tokenizer"]
            model = self.vlm["model"]
            proc = self.vlm["proc"]
            # Mirror demo_video_fastvlm.py pattern exactly
            conv = conv_templates["llava_v1"].copy()
            user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt.strip()}"
            conv.append_message(conv.roles[0], user_prompt)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            images = [crop.convert("RGB")]
            dtype = torch.float16 if self.vlm_device != "cpu" else torch.float32
            image_tensor = process_images(images, proc, model.config).to(model.device, dtype=dtype)
            input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
            with torch.inference_mode():
                with self.vlm_lock:
                    output_ids = model.generate(
                        inputs=input_ids,
                        images=image_tensor,
                        image_sizes=[images[0].size],
                        do_sample=False,
                        max_new_tokens=48,
                        use_cache=True,
                    )
            raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            ans = raw
            try:
                ptxt = prompt.strip()
                if ans.lower().startswith(ptxt.lower()):
                    ans = ans[len(ptxt):].lstrip(':\n ').strip()
            except Exception:
                pass
            self.vlm_cache[key] = (ans, time.time())
        except Exception as e:
            import traceback
            msg = ''.join(traceback.format_exc())
            print(f"[vlm] caption error for key={key}: {msg}")
            # Auto-fallback: if on GPU/MPS, reload VLM on CPU and retry once
            if self.vlm_device != 'cpu':
                try:
                    self._load_vlm_async(self.vlm_var.get().strip() or DEFAULT_VLM_DIR, force_device='cpu')
                except Exception:
                    pass
            self.vlm_cache[key] = (f"error: {str(e)[:60]}", time.time())
        finally:
            try:
                self.vlm_sem.release()
            except Exception:
                pass


if __name__ == "__main__":
    App().mainloop()
