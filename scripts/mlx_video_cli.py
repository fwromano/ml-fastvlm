#!/usr/bin/env python3
import argparse, os, sys, subprocess, tempfile, time
import cv2
from PIL import Image

def main():
    ap = argparse.ArgumentParser(description="Run MLX-VLM on video frames (headless)")
    ap.add_argument("--model", required=True, help="Path to exported MLX model (must contain .mlpackage vision tower)")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--prompt", required=True, help="Prompt to run per (sampled) frame")
    ap.add_argument("--interval-ms", type=int, default=0, help="Min delay between frames (0 = as fast as possible)")
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--every-nth", type=int, default=1, help="Process every Nth decoded frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] Cannot open video: {args.video}", file=sys.stderr)
        return 2

    tmpdir = tempfile.TemporaryDirectory()
    tmp_img = os.path.join(tmpdir.name, "frame.jpg")
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.every_nth > 1 and (frame_idx % args.every_nth) != 0:
                continue
            # Write frame to temp file in RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(tmp_img, format="JPEG", quality=92)

            # Call MLX-VLM generator CLI
            # Prefer older invocation for compatibility with pinned mlx-vlm
            cmd = [sys.executable, "-m", "mlx_vlm.generate", "--model", args.model, "--image", tmp_img, "--prompt", args.prompt, "--verbose"]
            # Some mlx-vlm versions support --max-tokens; if not, the flag is ignored
            cmd += ["--max-tokens", str(args.max_new_tokens)]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
            except subprocess.CalledProcessError as e:
                # Fallback to new style CLI if available
                try:
                    cmd2 = [sys.executable, "-m", "mlx_vlm", "generate", "--model", args.model, "--image", tmp_img, "--prompt", args.prompt, "--max-tokens", str(args.max_new_tokens), "--verbose"]
                    out = subprocess.check_output(cmd2, stderr=subprocess.STDOUT, text=True)
                except subprocess.CalledProcessError as e2:
                    print(e.output + "\n" + e2.output, file=sys.stderr)
                    print("[error] mlx-vlm generate failed. Ensure mlx_vlm is installed and model is valid.", file=sys.stderr)
                    return 3
            # Print last meaningful line as the answer (filter noisy logs)
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            noise_markers = (
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
            filtered = [ln for ln in lines if not any(m in ln for m in noise_markers)]
            to_print = filtered[-1] if filtered else (lines[-1] if lines else "")
            if to_print:
                print(to_print)
            if args.interval_ms > 0:
                time.sleep(args.interval_ms/1000.0)
    finally:
        cap.release()
        tmpdir.cleanup()

    return 0

if __name__ == "__main__":
    sys.exit(main())
