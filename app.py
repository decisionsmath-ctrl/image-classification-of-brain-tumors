"""Streamlit demo: upload a brain MRI image -> Tumor / No Tumor model score (+ optional Grad-CAM).

    streamlit run app.py

EDUCATIONAL RESEARCH PROTOTYPE. NOT A MEDICAL DEVICE. NOT FOR DIAGNOSIS.

The app only LOADS the saved model (artifacts/models/final_model.keras + app_metadata.json).
It never trains, never runs the optimiser and never downloads the dataset.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from classification.gradcam import GradCAMError, make_gradcam_heatmap, overlay_heatmap
from classification.preprocessing import ImageLoadError, load_image_uint8, preprocess_for_model
from classification.utils import DISCLAIMER, load_config, load_json, path_from_cfg

CFG = load_config()
st.set_page_config(page_title="Brain MRI Tumor / No Tumor (research prototype)", page_icon="🧠", layout="wide")


@st.cache_resource(show_spinner="Loading saved model ...")
def load_model_and_metadata(model_path: str, meta_path: str):
    import keras
    meta = load_json(Path(meta_path))
    model = keras.models.load_model(model_path, compile=False)
    return model, meta


def show_disclaimer() -> None:
    st.error("**Educational research prototype - NOT a medical diagnostic tool.** This model has not been clinically "
             "validated. Its output must never be used to diagnose, rule out, or make decisions about any medical "
             "condition. Consult a qualified healthcare professional for any health concern.", icon="⚠️")


st.title(CFG["app"]["title"])
show_disclaimer()

models_dir = path_from_cfg(CFG, "models_dir")
model_path = models_dir / CFG["app"]["model_file"]
meta_path = models_dir / CFG["app"]["metadata_file"]

missing = [p for p in (model_path, meta_path) if not p.exists()]
if missing:
    st.error("**Model files not found:**\n\n" + "\n".join(f"- `{p}`" for p in missing))
    st.info("Train the model first (see README: *Training commands*), or copy `final_model.keras` and "
            "`app_metadata.json` into `artifacts/models/`. The app never trains or downloads data by itself.")
    st.stop()

try:
    model, meta = load_model_and_metadata(str(model_path), str(meta_path))
except Exception as exc:
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

class_names = meta["class_names"]
positive = meta["positive_class"]
threshold = float(meta["threshold"])
image_cfg = meta["image"]

with st.sidebar:
    st.header("About this prototype")
    st.write(f"**Task:** binary image classification - *{class_names[0]}* vs *{class_names[1]}*.")
    st.write(f"**Model:** custom CNN ({meta['selected_method']} hyper-parameters), trained on {meta['trained_on']}.")
    st.write(f"**Input:** {image_cfg['size']}x{image_cfg['size']} px, "
             f"{'grayscale' if image_cfg['channels'] == 1 else 'RGB'}, pixels / 255.")
    st.write(f"**Decision threshold:** {threshold:.2f} on the model score.")
    show_cam = st.checkbox("Show Grad-CAM heatmap", value=True)
    st.caption("Grad-CAM shows which image regions influenced the model's score. It reflects the model's attention "
               "pattern only and is not proof of clinical reasoning or a tumour location.")
    st.divider()
    st.caption(f"Training data source: {meta.get('dataset', 'n/a')}")

uploaded = st.file_uploader("Upload a brain MRI image (PNG / JPG)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to run the model. Do not upload images that identify a real person.")
    st.stop()

raw = uploaded.getvalue()
try:
    x = preprocess_for_model(io.BytesIO(raw), image_cfg)             # identical to training preprocessing
    shown = load_image_uint8(io.BytesIO(raw), image_cfg["size"], image_cfg["channels"], image_cfg["resample"])
except ImageLoadError as exc:
    st.error(f"This file could not be read as an image: {exc}")
    st.stop()

score = float(model.predict(x, verbose=0)[0, 0])
predicted = class_names[int(score >= threshold)]

left, mid, right = st.columns(3)
with left:
    st.subheader("Uploaded image")
    st.image(raw, width="stretch")
    st.caption(f"Model input after preprocessing: {image_cfg['size']}x{image_cfg['size']}")
with mid:
    st.subheader("Model output")
    st.metric(label="Predicted class (model output, not a diagnosis)", value=predicted)
    st.metric(label=f"Model score for '{positive}'", value=f"{score:.3f}")
    st.progress(min(max(score, 0.0), 1.0))
    st.markdown(
        f"**What the score means:** the network ends in a sigmoid unit, so the score is a number between 0 and 1. "
        f"Scores of **{threshold:.2f} or higher** are labelled *{positive}*, lower scores *{class_names[0]}*. "
        "It is **not** a calibrated probability, not a confidence of being correct, and not a medical result. "
        "Scores close to the threshold mean the model is uncertain.")
    if abs(score - threshold) < 0.15:
        st.warning("The score is close to the decision threshold - the model is uncertain about this image.")
    if abs(score - threshold) >= 0.15 and (score < 0.02 or score > 0.98):
        st.caption("Very extreme scores can also occur for images unlike the training data (for example "
                   "non-brain pictures); the model has no way to say 'this is not a brain MRI'.")
with right:
    st.subheader("Grad-CAM")
    if not show_cam:
        st.info("Grad-CAM is switched off in the sidebar.")
    else:
        try:
            cam = make_gradcam_heatmap(model, x, meta.get("gradcam_layer"))
            if cam.degenerate:
                st.info("Grad-CAM is not informative for this image (gradients vanished, e.g. a saturated score).")
            else:
                overlay = overlay_heatmap(shown, cam.heatmap, float(meta.get("gradcam_alpha", 0.4)))
                st.image(overlay, width="stretch")
                st.caption(f"Layer '{cam.layer_name}'. Warm colours = regions that raised the *{positive}* score. "
                           "Model attention only - not clinical reasoning.")
        except GradCAMError as exc:
            st.info(f"Grad-CAM is not available for this model: {exc}")
        except Exception as exc:  # keep the prediction visible even if the visualisation fails
            st.info(f"Grad-CAM could not be generated: {exc}")

st.divider()
show_disclaimer()
st.caption(DISCLAIMER)
