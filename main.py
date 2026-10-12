#!/usr/bin/env python3
"""Caption-behind-subject renderer.

  python main.py --video in.mp4 --template dynamic --out out.mp4            # fully automatic captions
  python main.py --video in.mp4 --srt in.srt --template dynamic --out out.mp4  # or use your own .srt
  python main.py --video in.mp4 --srt in.srt --template luxe --preview 1.5     # one PNG, fast iteration
  python main.py --list
"""
import argparse, shutil, subprocess, sys, time
import cv2
import numpy as np

from captions import active_chunk, load_chunks, words_to_srt
from compositor import composite, find_head_top
from renderer.text_layer import TemplateRenderer
from segmentation import get_segmenter
from template_loader import list_templates, load_template


def to_np(layer):
    return None if layer is None else np.asarray(layer)


def process_frame(frame, t, chunks, renderer, segmenter):
    chunk = active_chunk(chunks, t)
    if chunk is None:
        return frame  # no caption right now: skip segmentation entirely (big speed-up)
    mask = segmenter.get_mask(frame)
    behind, front = renderer.render(chunk, t, find_head_top(mask))
    return composite(frame, mask, to_np(behind), to_np(front))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--srt", help="optional. If omitted, the video's speech is transcribed automatically")
    ap.add_argument("--model", default="base", help="Whisper size for auto captions: tiny|base|small|medium (bigger = more accurate, slower)")
    ap.add_argument("--language", help="e.g. en, ur, hi. Default: auto-detect")
    ap.add_argument("--transcribe-only", action="store_true", help="just create the transcript (+ .srt) and stop")
    ap.add_argument("--template", default="dynamic")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--segmenter", default="mediapipe", help="mediapipe | rembg | rvm | ellipse")
    ap.add_argument("--preview", type=float, help="render only this timestamp (s) to preview.png")
    ap.add_argument("--set", help='custom settings as JSON, merged over the template, e.g. \'{"colors":{"keyword":"#FF3366"},"layout":{"y_ratio":0.4}}\'')
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        print("\n".join(list_templates())); return
    if not a.video:
        ap.error("--video is required")
    if a.transcribe_only:
        from transcribe import transcribe
        import os
        words = transcribe(a.video, a.model, a.language)
        out = os.path.splitext(a.video)[0] + ".srt"
        words_to_srt(words, out); print(f"wrote {out}"); return

    import json
    tpl = load_template(a.template, json.loads(a.set) if a.set else None)
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    chunks = load_chunks(a.video, a.srt, tpl.get("words_per_chunk", 4), tpl.get("strip_punctuation", True), a.model, a.language)
    renderer = TemplateRenderer(tpl, W, H)
    segmenter = get_segmenter(a.segmenter)

    if a.preview is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, a.preview * 1000)
        ok, frame = cap.read()
        if not ok: sys.exit("could not read frame at that time")
        cv2.imwrite("preview.png", process_frame(frame, a.preview, chunks, renderer, segmenter))
        print("wrote preview.png"); return

    ffmpeg = shutil.which("ffmpeg") or sys.exit("ffmpeg not found on PATH")
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-i", a.video,                       # original file -> audio source
           "-map", "0:v", "-map", "1:a?",       # '?' = don't fail if the video has no audio
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
           "-c:a", "copy", "-shortest", "-movflags", "+faststart", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    t0, i = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        proc.stdin.write(process_frame(frame, i / fps, chunks, renderer, segmenter).tobytes())
        i += 1
        if i % 30 == 0:
            print(f"\r{i}/{total} frames  ({i / (time.time() - t0):.1f} fps)", end="", flush=True)
    proc.stdin.close(); proc.wait()
    print(f"\ndone -> {a.out}")


if __name__ == "__main__":
    main()
