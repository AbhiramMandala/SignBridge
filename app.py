"""SignBridge MVP Gradio app. Run: python app.py  (CPU-friendly, no GPU needed)."""
from __future__ import annotations

import cv2
import gradio as gr
import numpy as np

from src.config import CONFIDENCE_THRESHOLD, MODEL_PATH

CSS = ".big-letter { font-size: 96px; font-weight: 800; text-align: center; }"


def _bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img_rgb), cv2.COLOR_RGB2BGR)


def classify(image, threshold=CONFIDENCE_THRESHOLD):
    """Gradio entry: returns (annotated_rgb, letter_md, conf, top3_text, state)."""
    from src.predict import load, predict_array
    if image is None:
        return None, "Upload or capture an image first.", 0.0, "", None
    try:
        load()
    except FileNotFoundError as e:
        return image, f"**Model missing.** {e}", 0.0, "", None
    r = predict_array(_bgr(image), threshold)
    if r["status"] == "no_hand":
        return image, "**No hand detected.** Show one hand clearly to the camera.", 0.0, "", None
    annotated_rgb = cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB)
    top3 = "\n".join(f"{lab}: {p:.2f}" for lab, p in r["top3"])
    return annotated_rgb, f"<div class='big-letter'>{r['letter']}</div>", r["confidence"], top3, r["letter"]


def add_letter(word: str, letter) -> str:
    if not letter or letter in ("uncertain", "nothing"):
        return word
    if letter == "del":
        return word[:-1]
    return word + (" " if letter == "space" else letter)


def build_app():
    with gr.Blocks(title="SignBridge — ASL Alphabet MVP", css=CSS) as app:
        gr.Markdown("# SignBridge — ASL Alphabet MVP\nStatic letters A–Z (+ space/del/nothing).")
        with gr.Row():
            with gr.Column():
                img = gr.Image(label="Webcam snapshot or upload", sources=["webcam", "upload"], type="numpy")
                threshold = gr.Slider(0.0, 0.95, value=CONFIDENCE_THRESHOLD, step=0.05, label="Confidence threshold")
                btn = gr.Button("Recognize", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="Landmarks (first/most-confident hand used if several)")
                out_letter = gr.Markdown("Press **Recognize**.")
                out_conf = gr.Number(label="Confidence", minimum=0, maximum=1)
                out_top3 = gr.Text(label="Top-3", lines=3)
        state = gr.State(value=None)
        with gr.Row():
            word = gr.Textbox(label="Word builder", placeholder="Recognize, then Add letter…")
            add = gr.Button("Add letter")
            back = gr.Button("Backspace")
            clear = gr.Button("Clear")
        btn.click(classify, [img, threshold], [out_img, out_letter, out_conf, out_top3, state])
        add.click(add_letter, [word, state], word)
        back.click(lambda w: w[:-1], word, word)
        clear.click(lambda: "", None, word)
        gr.Markdown("*Static ASL alphabet demo. Not a full sign language translator.* "
                    f"Model: `{MODEL_PATH.name}` (train: `python -m src.dataset && python -m src.train`).")
    return app


if __name__ == "__main__":
    build_app().launch()
