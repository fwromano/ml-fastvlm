#!/usr/bin/env python3
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import math
import cv2
import numpy as np
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from transformers import StoppingCriteriaList
from llava.conversation import conv_templates


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sample_frames(video_path: str, max_frames: int = 9):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1), num=min(max_frames, max(1, total))).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    if not frames:
        raise RuntimeError("No frames sampled from video.")
    return frames


def make_grid(pil_images, grid_cols=None):
    n = len(pil_images)
    if n == 0:
        raise RuntimeError("No frames to mosaic.")
    if grid_cols is None:
        grid_cols = int(math.ceil(math.sqrt(n)))
    grid_rows = int(math.ceil(n / grid_cols))
    w, h = pil_images[0].size
    pil_images = [im.resize((w, h)) for im in pil_images]
    canvas = Image.new("RGB", (grid_cols * w, grid_rows * h), (0, 0, 0))
    for i, im in enumerate(pil_images):
        r, c = divmod(i, grid_cols)
        canvas.paste(im, (c * w, r * h))
    return canvas


@torch.inference_mode()
def run_inference_bundle(bundle, video_path: str, prompt: str, frames: int, max_new_tokens: int, temperature: float):
    tokenizer, model, image_processor = bundle["tokenizer"], bundle["model"], bundle["image_processor"]
    frames_list = sample_frames(video_path, max_frames=frames)
    grid = make_grid(frames_list)

    conv = conv_templates["llava_v1"].copy()
    user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt.strip()}"
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    images = [grid.convert("RGB")]
    dtype = torch.float16 if str(model.device) != "cpu" else torch.float32
    image_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=dtype)
    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        images=image_tensor,
        do_sample=temperature > 0,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        use_cache=True,
    )
    ans = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    if ans.lower().startswith(prompt.lower()):
        ans = ans[len(prompt):].strip(": \n")
    return ans, grid


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FastVLM Video → Text")
        self.geometry("840x700")
        self.resizable(True, True)

        default_video = os.environ.get("VIDEO_PATH", "/Users/agc/Documents/output.mp4")
        # Fixed model path (hide model selection)
        self.default_model_dir = os.environ.get("MODEL_DIR", "checkpoints/llava-fastvithd_7b_stage3")
        DEFAULT_PROMPT = (
            "Return ONLY minimal JSON for visible vehicles.\n\n"
            "Schema (use exactly these keys):\n"
            "{\n"
            "  \"vehicles\": [\n"
            "    {\"id\":\"v1\",\"type\":\"<sedan|suv|truck|van|bus|motorcycle|bicycle|unknown>\",\"color\":\"<e.g., white>\",\"notes\":[\"<e.g., parked|moving>\"]}\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Output JSON only; no prose, no code fences.\n"
            "- If no vehicles, use \"vehicles\": [].\n"
            "- Use ids v1, v2, …; lowercase all strings; lists ≤3 items; omit a field if you cannot infer it; ensure valid JSON."
        )

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Status
        self.status_var = tk.StringVar(value="Loading model...")
        self.status_lbl = ttk.Label(frm, textvariable=self.status_var, foreground="#555")
        self.status_lbl.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0,6))

        # Video path
        ttk.Label(frm, text="Video:").grid(row=1, column=0, sticky=tk.W)
        self.video_var = tk.StringVar(value=default_video)
        self.video_entry = ttk.Entry(frm, textvariable=self.video_var, width=70)
        self.video_entry.grid(row=1, column=1, sticky=tk.EW, padx=(4, 4))
        self.video_btn = ttk.Button(frm, text="Browse", command=self.browse_video)
        self.video_btn.grid(row=1, column=2)

        # Model selection removed from UI; using fixed default (self.default_model_dir)

        # Prompt
        ttk.Label(frm, text="Prompt:").grid(row=3, column=0, sticky=tk.NW)
        self.prompt_text = tk.Text(frm, height=8, width=60)
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)
        self.prompt_text.grid(row=3, column=1, sticky=tk.EW, padx=(4, 4))
        frm.rowconfigure(3, weight=0)

        # Playback + Inference controls
        btns = ttk.Frame(frm)
        btns.grid(row=4, column=1, pady=(8,8))
        self.play_btn = ttk.Button(btns, text="Play", command=self.toggle_play)
        self.play_btn.grid(row=0, column=0, padx=4)
        self.inf_btn = ttk.Button(btns, text="Start Inference", command=self.toggle_infer)
        self.inf_btn.grid(row=0, column=1, padx=4)

        # Output
        ttk.Label(frm, text="Model Output:").grid(row=6, column=0, sticky=tk.NW)
        self.out_text = tk.Text(frm, height=10, width=60)
        self.out_text.grid(row=6, column=1, sticky=tk.NSEW, padx=(4, 4))

        # Current frame preview
        ttk.Label(frm, text="Current frame:").grid(row=7, column=0, sticky=tk.NW)
        self.preview_label = ttk.Label(frm)
        self.preview_label.grid(row=7, column=1, sticky=tk.NSEW)

        # Column expand
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(6, weight=1)
        frm.rowconfigure(7, weight=1)

        self._img_ref = None
        self._prev_ref = None
        self.cap = None
        self.playing = False
        self.infer_on = False
        self.infer_busy = False
        self.after_id = None
        self.current_frame_pil = None

        # Model bundle (loaded asynchronously)
        self.bundle = None
        self.loaded_model_dir = None
        self.device = pick_device()

        # Disable controls until model is loaded
        self.set_controls_state(False)
        threading.Thread(target=self._load_model_startup, daemon=True).start()
        # Auto-open capture if default exists
        if os.path.isfile(default_video):
            try:
                self._open_capture(default_video)
            except Exception:
                pass

    def set_controls_state(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        for w in [self.video_entry, self.video_btn, self.prompt_text, self.play_btn, self.inf_btn]:
            try:
                w.configure(state=state)
            except Exception:
                pass

    def _load_model_startup(self):
        try:
            self._load_model(self.default_model_dir)
            self.after(0, lambda: self.status_var.set(f"Model loaded • {self.device}"))
            self.after(0, lambda: self.set_controls_state(True))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Failed to load model:\n{e}"))

    def browse_video(self):
        path = filedialog.askopenfilename(title="Select video",
                                          filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*.*")])
        if path:
            self.video_var.set(path)
            try:
                self._open_capture(path)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    # Model browsing disabled; model is fixed to self.default_model_dir

    def on_run(self):
        # Single inference on current frame
        if self.current_frame_pil is None:
            messagebox.showerror("Error", "No frame to run on. Press Play to display a frame.")
            return
        self._run_inference_on_frame(self.current_frame_pil)

        # no-op for legacy path; continuous inference is handled via Play + Start Inference
        return

    def _load_model(self, model_dir: str):
        model_name = get_model_name_from_path(model_dir)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_dir, None, model_name, device=self.device)
        model.eval()
        if hasattr(model.config, "mm_use_im_start_end"):
            model.config.mm_use_im_start_end = False
        # Force simple image path to avoid fragile anyres/pad branches causing None shapes
        # Do not override image_aspect_ratio; honor model defaults (e.g., anyres) so
        # image_sizes from the original frame can be used correctly downstream.
        # Sanity check: ensure image_processor works on a simple RGB image via direct call
        if image_processor is None:
            raise RuntimeError("Image processor not initialized. Model vision tower did not load correctly.")
        try:
            from PIL import Image
            dummy = Image.new("RGB", (8, 8), (128, 128, 128))
            _ = image_processor([dummy], return_tensors='pt')['pixel_values']
        except Exception as e:
            raise RuntimeError(f"Image preprocessor check failed: {e}")
        # Set pad token id for safer generation
        try:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        except Exception:
            pass
        self.bundle = {"tokenizer": tokenizer, "model": model, "image_processor": image_processor}
        self.loaded_model_dir = model_dir
        # Reset prompt cache on model change
        self._cached_prompt_key = None
        self._cached_input_ids = None

    # --- Video playback and inference helpers ---
    def _open_capture(self, path: str):
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.cap = cap
        self.status_var.set("Ready • video loaded")

    def toggle_play(self):
        if self.cap is None:
            path = self.video_var.get().strip()
            if not os.path.isfile(path):
                messagebox.showerror("Error", f"Video not found:\n{path}")
                return
            try:
                self._open_capture(path)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing and self.cap is not None:
            self._video_loop()

    def _video_loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            # loop back to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.current_frame_pil = pil
            pw, ph = pil.size
            s = min(720 / max(pw, 1), 360 / max(ph, 1), 1.0)
            disp = pil.resize((int(pw*s), int(ph*s)))
            photo = ImageTk.PhotoImage(disp)
            self._prev_ref = photo
            self.preview_label.configure(image=photo)
            # Trigger inference if enabled and not busy
            if self.infer_on and not self.infer_busy and self.bundle is not None:
                self._run_inference_on_frame(pil)
        if self.playing:
            self.after(0, self._video_loop)

    def toggle_infer(self):
        self.infer_on = not self.infer_on
        self.inf_btn.configure(text="Stop Inference" if self.infer_on else "Start Inference")
        if self.infer_on and self.current_frame_pil is not None and not self.infer_busy and self.bundle is not None:
            self._run_inference_on_frame(self.current_frame_pil)

    def _run_inference_on_frame(self, pil_image: Image.Image):
        if self.bundle is None or pil_image is None:
            return
        # Validate frame
        try:
            w, h = pil_image.size
            if w <= 1 or h <= 1:
                raise ValueError(f"Invalid frame size: {w}x{h}")
            img_rgb = pil_image.convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid frame: {e}")
            return
        self.infer_busy = True
        self.status_var.set("Running inference on current frame...")
        prompt = self.prompt_text.get("1.0", tk.END).strip() or "Describe this frame succinctly."

        def worker(img=img_rgb.copy(), prompt_text=prompt):
            try:
                tokenizer = self.bundle["tokenizer"]
                model = self.bundle["model"]
                image_processor = self.bundle["image_processor"]

                # Build prompt similar to demo_video_fastvlm (keeps outputs concise)
                qs = prompt_text.strip()
                if getattr(model.config, "mm_use_im_start_end", False):
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
                conv_name = "llava_v1"
                cache_key = (prompt_text.strip(), conv_name, bool(getattr(model.config, "mm_use_im_start_end", False)))
                if self._cached_prompt_key == cache_key and self._cached_input_ids is not None:
                    input_ids = self._cached_input_ids
                else:
                    conv = conv_templates[conv_name].copy()
                    conv.append_message(conv.roles[0], qs)
                    conv.append_message(conv.roles[1], None)
                    full_prompt = conv.get_prompt()
                    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
                    self._cached_prompt_key = cache_key
                    self._cached_input_ids = input_ids
                dtype = torch.float16 if str(model.device) != "cpu" else torch.float32
                try:
                    # Follow repo's predict.py pattern for single image
                    image_tensor_single = process_images([img], image_processor, model.config)[0]
                except Exception as ex:
                    raise RuntimeError(f"process_images failed on frame: {ex}")
                images_batched = image_tensor_single.unsqueeze(0).to(model.device, dtype=dtype)
                # Stop if the model starts emitting role markers like USER: or ASSISTANT:
                stop_words = ["USER:", "User:", "ASSISTANT:", "Assistant:"]
                stopping = StoppingCriteriaList([KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)])
                output_ids = model.generate(
                    input_ids,
                    images=images_batched,
                    image_sizes=[img.size],
                    do_sample=False,
                    max_new_tokens=75,
                    use_cache=True,
                    stopping_criteria=stopping,
                )
                raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                ans = raw
                # Post-process to remove echoed prompt or QA wrappers
                try:
                    import re
                    ptxt = prompt_text.strip()
                    if ans.lower().startswith(ptxt.lower()):
                        ans = ans[len(ptxt):].lstrip(':\n ').strip()
                    # Extract text after the last '**Answer:**' or 'Answer:' if present
                    m = list(re.finditer(r'(\*\*Answer:?\*\*|\bAnswer:)', ans, flags=re.IGNORECASE))
                    if m:
                        ans = ans[m[-1].end():].strip()
                    # Remove leading '**' markdown remnants
                    ans = re.sub(r'^\*+\s*', '', ans).strip()
                    # Strip role labels at line starts
                    ans = re.sub(r'(?mi)^(?:ASSISTANT|Assistant|USER|User)\s*:\s*', '', ans).strip()
                except Exception:
                    pass
            except Exception as e:
                import traceback, sys
                traceback.print_exc()
                def fail(err=e):
                    self.status_var.set("Error")
                    messagebox.showerror("Error", str(err))
                    self.infer_busy = False
                self.after(0, fail)
                return

            def done(a=ans):
                self.out_text.delete("1.0", tk.END)
                self.out_text.insert(tk.END, a)
                self.status_var.set("Ready")
                self.infer_busy = False
            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
