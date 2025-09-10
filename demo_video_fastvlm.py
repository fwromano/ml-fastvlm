# demo_video_fastvlm.py
import os, math, cv2, numpy as np
from PIL import Image
import gradio as gr
import torch
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates

MODEL_DIR = os.environ.get("MODEL_DIR") or "checkpoints/llava-fastvithd_1.5b_stage3"
device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
model_name = get_model_name_from_path(MODEL_DIR)
tokenizer, model, image_processor, context_len = load_pretrained_model(MODEL_DIR, None, model_name, device=device)
model.eval()
if hasattr(model.config, "mm_use_im_start_end"):
    model.config.mm_use_im_start_end = False

def sample_frames(video_path, max_frames=9):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video.")
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
    return frames

def make_grid(pil_images, grid_cols=None):
    n = len(pil_images)
    if n == 0:
        raise RuntimeError("No frames sampled.")
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
def run(video, prompt, frames=9, max_new_tokens=128, temperature=0.2):
    frames_list = sample_frames(video, max_frames=frames)
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

with gr.Blocks() as demo:
    gr.Markdown("FastVLM Video → Text Demo")
    with gr.Row():
        video = gr.Video(label="Video", sources=["upload"], height=300)
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="Describe the key events in this clip succinctly.")
            frames = gr.Slider(4, 16, value=9, step=1, label="Frames sampled")
            run_btn = gr.Button("Run")
    out_text = gr.Textbox(label="Model Output")
    out_image = gr.Image(label="Frames mosaic the model saw", type="pil")
    run_btn.click(run, inputs=[video, prompt, frames], outputs=[out_text, out_image])

# Note: We avoid gradio queue customization to keep behavior identical to the repo.

if __name__ == "__main__":
    # Allow forcing share via env var; also fallback to share=True if localhost isn't accessible.
    share_env = os.environ.get("DEMO_SHARE")
    share = None if share_env is None else (share_env not in ("0", "false", "False"))
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=(share or False))
    except ValueError as e:
        msg = str(e)
        if "shareable link must be created" in msg or "localhost is not accessible" in msg:
            demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, share=True)
        else:
            raise
