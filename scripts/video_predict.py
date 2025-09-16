#!/usr/bin/env python
import argparse, os, math, sys
import cv2
import numpy as np
from PIL import Image
import torch

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
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
def infer_video(model_dir: str, video_path: str, prompt: str, frames: int, max_new_tokens: int, temperature: float, device: str):
    model_name = get_model_name_from_path(model_dir)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_dir, None, model_name, device=device)
    model.eval()
    if hasattr(model.config, "mm_use_im_start_end"):
        model.config.mm_use_im_start_end = False

    frames_list = sample_frames(video_path, max_frames=frames)
    grid = make_grid(frames_list)

    conv = conv_templates["llava_v1"].copy()
    user_prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt.strip()}"
    conv.append_message(conv.roles[0], user_prompt)
    conv.append_message(conv.roles[1], None)
    full_prompt = conv.get_prompt()

    images = [grid.convert("RGB")]
    dtype = torch.float16 if device != "cpu" else torch.float32
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


def main():
    parser = argparse.ArgumentParser(description="FastVLM video → text (CLI)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "checkpoints/llava-fastvithd_7b_stage3"))
    parser.add_argument(
        "--prompt",
        default=(
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
        ),
    )
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--max-new-tokens", type=int, default=75)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default=None, help="Force device: mps|cuda|cpu")
    parser.add_argument("--save-grid", default=None, help="Optional path to save the sampled-frames mosaic image")
    args = parser.parse_args()

    dev = args.device or pick_device()
    print(f"[info] Using device: {dev}")
    print(f"[info] Model dir: {args.model_dir}")
    print(f"[info] Video: {args.video}")

    ans, grid = infer_video(
        model_dir=args.model_dir,
        video_path=args.video,
        prompt=args.prompt,
        frames=args.frames,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=dev,
    )
    if args.save_grid:
        os.makedirs(os.path.dirname(args.save_grid) or ".", exist_ok=True)
        grid.save(args.save_grid)
        print(f"[info] Saved mosaic: {args.save_grid}")

    print("\n=== Model Output ===\n")
    print(ans)


if __name__ == "__main__":
    sys.exit(main())
