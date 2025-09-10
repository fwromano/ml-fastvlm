#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tempfile
import subprocess
import cv2
from PIL import Image, ImageTk

# Try to use MLX-VLM Python API for maximum speed (avoid subprocess)
USE_API = True
try:
    from mlx_vlm.utils import load as mlx_load, load_config as mlx_load_config, generate as mlx_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat
except Exception:
    USE_API = False


def run_mlx_generate_subprocess(model_dir: str, image_path: str, prompt: str, max_tokens: int = 100) -> str:
    cmds = [
        ["python3", "-m", "mlx_vlm.generate", "--model", model_dir, "--image", image_path, "--prompt", prompt, "--max-tokens", str(max_tokens), "--verbose"],
        ["python3", "-m", "mlx_vlm", "generate", "--model", model_dir, "--image", image_path, "--prompt", prompt, "--max-tokens", str(max_tokens), "--verbose"],
    ]
    last_err = None
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            # Filter noisy lines
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            noise = (
                "Peak memory:",
                "Torch version",
                "beta version",
                "Looking for CoreML vision tower",
                "Loading ",
                "INFO:",
                "WARNING:",
                "Generation:",
                "Speed:",
            )
            filt = [ln for ln in lines if not any(m in ln for m in noise)]
            return (filt[-1] if filt else (lines[-1] if lines else "")).strip()
        except subprocess.CalledProcessError as e:
            last_err = e.output
    raise RuntimeError(last_err or "mlx_vlm.generate failed")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FastVLM (MLX) Video → Text")
        self.geometry("900x700")

        default_video = os.environ.get("VIDEO_PATH", "/Users/agc/Documents/output.mp4")
        default_model = os.environ.get("MODEL_DIR", "exported/fastvlm_1.5b_mlx")

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready (MLX)")
        ttk.Label(frm, textvariable=self.status_var, foreground="#555").grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0,6))

        ttk.Label(frm, text="Video:").grid(row=1, column=0, sticky=tk.W)
        self.video_var = tk.StringVar(value=default_video)
        self.video_entry = ttk.Entry(frm, textvariable=self.video_var, width=70)
        self.video_entry.grid(row=1, column=1, sticky=tk.EW, padx=(4,4))
        ttk.Button(frm, text="Browse", command=self.browse_video).grid(row=1, column=2)

        ttk.Label(frm, text="Model dir:").grid(row=2, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=default_model)
        self.model_entry = ttk.Entry(frm, textvariable=self.model_var, width=70)
        self.model_entry.grid(row=2, column=1, sticky=tk.EW, padx=(4,4))
        ttk.Button(frm, text="Browse", command=self.browse_model).grid(row=2, column=2)

        ttk.Label(frm, text="Prompt:").grid(row=3, column=0, sticky=tk.NW)
        self.prompt_text = tk.Text(frm, height=4, width=60)
        self.prompt_text.insert("1.0", os.environ.get("PROMPT_DEFAULT", "Describe the environment and any agents inside it."))
        self.prompt_text.grid(row=3, column=1, sticky=tk.EW, padx=(4,4))

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=1, pady=(8,8))
        self.play_btn = ttk.Button(btns, text="Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=4)
        self.inf_btn = ttk.Button(btns, text="Start Inference", command=self.toggle_infer)
        self.inf_btn.grid(row=0, column=1, padx=4)

        ttk.Label(frm, text="Model Output:").grid(row=5, column=0, sticky=tk.NW)
        self.out_text = tk.Text(frm, height=10, width=60)
        self.out_text.grid(row=5, column=1, sticky=tk.NSEW, padx=(4,4))

        ttk.Label(frm, text="Current frame:").grid(row=6, column=0, sticky=tk.NW)
        self.preview_label = ttk.Label(frm)
        self.preview_label.grid(row=6, column=1, sticky=tk.NSEW)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(5, weight=1)
        frm.rowconfigure(6, weight=1)

        self.cap = None
        self.playing = False
        self.infer_on = False
        self.infer_busy = False
        self.current_frame = None
        self.model = None
        self.processor = None
        self.config = None

        # Preload video
        if os.path.isfile(default_video):
            self._open_capture(default_video)
        # Load model via MLX API for speed (if available)
        if USE_API:
            self._load_mlx_model_async(default_model)

    def _load_mlx_model_async(self, model_dir: str):
        self.status_var.set("Loading model (MLX)...")
        def worker():
            try:
                cfg = mlx_load_config(model_dir, trust_remote_code=True)
                model, processor = mlx_load(model_dir, adapter_path=None, lazy=False, trust_remote_code=True)
                # Warm-up with a tiny black image and short prompt
                try:
                    dummy = Image.new("RGB", (64, 64), (0, 0, 0))
                    prompt = "Describe."
                    prompt_fmt = mlx_apply_chat(processor, cfg, prompt, num_images=1)
                    _ = mlx_generate(model, processor, prompt_fmt, image=[dummy], temperature=0.0, max_tokens=4, verbose=False)
                except Exception:
                    pass
            except Exception as e:
                err = str(e)
                def fail():
                    messagebox.showerror("Model load failed", err)
                    self.status_var.set("Ready (subprocess fallback)")
                self.after(0, fail)
                return
            def done():
                self.model, self.processor, self.config = model, processor, cfg
                self.status_var.set("Ready (MLX API)")
            self.after(0, done)
        threading.Thread(target=worker, daemon=True).start()

    def browse_video(self):
        p = filedialog.askopenfilename(title="Select video", filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files","*.*")])
        if p:
            self.video_var.set(p)
            self._open_capture(p)

    def browse_model(self):
        p = filedialog.askdirectory(title="Select model directory")
        if p:
            self.model_var.set(p)

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

    def _video_loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.current_frame = pil
            pw, ph = pil.size
            s = min(720/max(pw,1), 360/max(ph,1), 1.0)
            disp = pil.resize((int(pw*s), int(ph*s)))
            photo = ImageTk.PhotoImage(disp)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            if self.infer_on and not self.infer_busy:
                self._run_inference_on_current()
        if self.playing:
            self.after(0, self._video_loop)

    def toggle_infer(self):
        self.infer_on = not self.infer_on
        self.inf_btn.configure(text="Stop Inference" if self.infer_on else "Start Inference")
        # Disable the toggle while a run is ongoing
        if self.infer_on:
            try:
                self.inf_btn.configure(state=tk.DISABLED)
            except Exception:
                pass
        else:
            try:
                self.inf_btn.configure(state=tk.NORMAL)
            except Exception:
                pass
        if self.infer_on and not self.infer_busy:
            self._run_inference_on_current()

    def _run_inference_on_current(self):
        if self.current_frame is None:
            return
        self.infer_busy = True
        self.status_var.set("Running inference...")
        prompt = (self.prompt_text.get("1.0", tk.END) or "").strip() or "Describe this frame."
        model_dir = self.model_var.get().strip()

        def worker(pil_img: Image.Image):
            try:
                if USE_API and self.model is not None and self.processor is not None and self.config is not None:
                    # Use in-memory call via MLX API (fast path)
                    prompt_fmt = mlx_apply_chat(self.processor, self.config, prompt, num_images=1)
                    ans = mlx_generate(self.model, self.processor, prompt_fmt, image=[pil_img], temperature=0.0, max_tokens=100, verbose=False)
                else:
                    # Subprocess fallback
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        pil_img.save(tmp.name, format="JPEG", quality=92)
                        ans = run_mlx_generate_subprocess(model_dir, tmp.name, prompt, max_tokens=100)
            except Exception as e:
                err = str(e)
                def fail():
                    self.status_var.set("Error")
                    messagebox.showerror("Error", err)
                    self.infer_busy = False
                self.after(0, fail)
                return
            def done():
                self.out_text.delete("1.0", tk.END)
                self.out_text.insert(tk.END, ans)
                self.status_var.set("Ready")
                self.infer_busy = False
                # Re-enable toggle if user hasn't turned it off
                try:
                    if self.infer_on:
                        self.inf_btn.configure(state=tk.NORMAL)
                except Exception:
                    pass
            self.after(0, done)

        threading.Thread(target=worker, args=(self.current_frame.copy(),), daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
