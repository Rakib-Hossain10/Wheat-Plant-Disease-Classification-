import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from datetime import datetime
import inspect


# ---------------------------------------
# Page config
# ---------------------------------------
st.set_page_config(
    page_title="Wheat Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------
# Blue vivid theme (safe CSS)
# ---------------------------------------
st.markdown("""
<style>
/* App background: vivid blue */
.stApp {
  background:
    radial-gradient(circle at 10% 15%, rgba(56,189,248,0.25), transparent 45%),
    radial-gradient(circle at 90% 10%, rgba(59,130,246,0.22), transparent 40%),
    radial-gradient(circle at 50% 95%, rgba(34,197,94,0.12), transparent 45%),
    linear-gradient(180deg, #081226 0%, #0b1b3a 40%, #0a1630 100%);
}

/* ensure header not clipped */
.block-container {
  padding-top: 3.0rem;
  padding-bottom: 1.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #070f1f 0%, #0b1733 70%, #070f1f 100%) !important;
  border-right: 1px solid rgba(148,163,184,0.18);
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Titles */
h1, h2, h3 { color: #eaf2ff; letter-spacing: -0.3px; }
p, li, span { color: rgba(226,232,240,0.92); }

/* Glass cards */
.card {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(148,163,184,0.22);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.20);
  backdrop-filter: blur(10px);
}

/* Badge */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.2px;
  background: rgba(34,197,94,0.16);
  color: #b7ffd0;
  border: 1px solid rgba(34,197,94,0.35);
}

/* Secondary badge */
.badge2 {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.2px;
  background: rgba(56,189,248,0.14);
  color: #cfefff;
  border: 1px solid rgba(56,189,248,0.35);
}

/* Small text */
.small {
  color: rgba(226,232,240,0.78);
  font-size: 13px;
  line-height: 1.45;
}

/* Hero */
.hero {
  background: linear-gradient(90deg, rgba(59,130,246,0.22), rgba(56,189,248,0.12));
  border: 1px solid rgba(148,163,184,0.22);
  border-radius: 18px;
  padding: 18px 18px;
  box-shadow: 0 18px 40px rgba(0,0,0,0.18);
}
.hero h2 {
  margin: 10px 0 6px 0;
  line-height: 1.15;
}

/* KPI */
.kpi {
  font-size: 34px;
  font-weight: 900;
  color: #eaf2ff;
  line-height: 1.05;
}
.kpi_sub {
  font-size: 13px;
  color: rgba(226,232,240,0.80);
}

/* Buttons */
.stButton > button {
  border-radius: 14px !important;
  padding: 10px 14px !important;
  font-weight: 800 !important;
}

/* Dataframe on dark bg */
[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.06) !important;
  border-radius: 14px !important;
  border: 1px solid rgba(148,163,184,0.18) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------
# Utility: width compatibility (Streamlit deprecation)
# ---------------------------------------
def button_stretch(label, key=None):
    sig = inspect.signature(st.button)
    if "width" in sig.parameters:
        return st.button(label, key=key, width="stretch")
    return st.button(label, key=key, use_container_width=True)

def download_stretch(label, data, file_name, mime, key=None):
    sig = inspect.signature(st.download_button)
    if "width" in sig.parameters:
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, width="stretch")
    return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, use_container_width=True)


# ---------------------------------------
# Load model + metadata
# ---------------------------------------
@st.cache_resource
def load_assets(model_path: str, meta_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    class_names = meta["class_names"]
    image_size = int(meta["IMAGE_SIZE"])
    channels = int(meta.get("CHANNELS", 3))
    return model, class_names, image_size, channels, meta


# ---------------------------------------
# Preprocess: your model includes preprocessing (Sequential + Lambda)
# Keep pixels in 0..255 and only resize.
# ---------------------------------------
def preprocess_image(pil_img: Image.Image, image_size: int) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((image_size, image_size))
    x = np.array(pil_img).astype(np.float32)  # keep 0..255
    x = np.expand_dims(x, axis=0)
    return x


def predict_topk(model, x: np.ndarray, class_names, top_k: int = 3):
    probs = model.predict(x, verbose=0)[0]
    probs = np.array(probs, dtype=np.float32)
    idx = probs.argsort()[::-1][:top_k]
    top = [(class_names[i], float(probs[i])) for i in idx]
    pred_name, pred_prob = top[0]
    return pred_name, pred_prob, top


def make_report(pred_name, pred_prob, top, meta):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": meta.get("PROJECT", "Wheat Disease Detection"),
        "prediction": pred_name,
        "confidence_percent": round(pred_prob * 100.0, 2),
        "top_predictions": [
            {"class": c, "probability_percent": round(p * 100.0, 2)} for c, p in top
        ],
        "class_names": meta.get("class_names", []),
        "merged_classes": meta.get("merged_classes", False),
        "merge_map": meta.get("merge_map", None),
        "prediction_logic": meta.get("prediction_logic", "argmax(softmax)")
    }


# ---------------------------------------
# Sidebar
# ---------------------------------------
st.sidebar.markdown("## Wheat Disease Detection")
st.sidebar.markdown("<div class='small'>ResNet50 • 8 classes</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

with st.sidebar.expander("Model settings", expanded=False):
    model_path = st.text_input("Model file", value="wheat_resnet50.keras")
    meta_path = st.text_input("Metadata file", value="wheat_metadata.json")

top_k = st.sidebar.slider("Top-K predictions", 2, 5, 3)

enable_download = st.sidebar.checkbox("Enable report download", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='small'>Note: This app keeps inputs in 0–255 and only resizes.</div> ",
    unsafe_allow_html=True
)

# Defaults if expander not opened
if "model_path" not in locals():
    model_path = "wheat_resnet50.keras"
if "meta_path" not in locals():
    meta_path = "wheat_metadata.json"


# ---------------------------------------
# Load assets
# ---------------------------------------
try:
    model, class_names, image_size, channels, meta = load_assets(model_path, meta_path)
except Exception as e:
    st.error("Failed to load model/metadata. Ensure files are in the same folder as app.py.")
    st.exception(e)
    st.stop()

if len(class_names) != 8:
    st.warning(f"Metadata contains {len(class_names)} classes (expected 8). Verify `wheat_metadata.json`.")


# ---------------------------------------
# Hero header (no clipping)
# ---------------------------------------
st.markdown(f"""
<div class="hero">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap;">
    <div style="min-width:320px;">
      <div class="badge2">Wheat Plant Disease Detection</div>
      <h2> Fast diagnosis with ResNet50</h2>
      <div class="small">Upload an image or take a photo. The model predicts the 8 classes and shows top-K confidence.</div>
    </div>
    <div style="text-align:right; min-width:220px;">
      <div class="small">Model input</div>
      <div style="font-weight:900; color:#eaf2ff;">{image_size} × {image_size} × {channels}</div>
      <div class="small">Prediction logic: argmax(softmax)</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("")


# ---------------------------------------
# Main layout
# ---------------------------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 1) Provide an image")
    st.markdown('<div class="small">Choose an input method. For best results, focus on the diseased region with clear lighting.</div>', unsafe_allow_html=True)

    tab_upload, tab_camera = st.tabs(["📁 Upload image", "📷 Take photo"])

    pil_img = None

    with tab_upload:
        uploaded = st.file_uploader("Upload (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        if uploaded is not None:
            pil_img = Image.open(uploaded)

    with tab_camera:
        # IMPORTANT: Do NOT mount camera_input automatically.
        if "camera_enabled" not in st.session_state:
            st.session_state.camera_enabled = False

        if not st.session_state.camera_enabled:
            st.markdown("<div class='small'>Camera will only open after you click the button below.</div>", unsafe_allow_html=True)
            if button_stretch("Open camera", key="open_camera_btn"):
                st.session_state.camera_enabled = True
                st.rerun()
        else:
            cam = st.camera_input("Take a photo", key="camera_widget")
            if cam is not None:
                pil_img = Image.open(cam)

            if button_stretch("Close camera", key="close_camera_btn"):
                st.session_state.camera_enabled = False
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 2) Run classification")

    colA, colB = st.columns([1, 1], gap="medium")
    with colA:
        run = button_stretch("Classify", key="classify_btn")
    with colB:
        reset = button_stretch("Reset", key="reset_btn")

    if reset:
        st.session_state.clear()
        st.rerun()

    st.markdown('<div class="small">If confidence is low, try a closer crop and retake the image.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Results")
    st.markdown('<div class="small">Prediction, confidence, and top-K probabilities.</div>', unsafe_allow_html=True)

    if pil_img is None:
        st.info("Provide an image from the left panel to see predictions.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.image(pil_img, caption=f"Preview (resized to {image_size}×{image_size} for inference)", use_container_width=True)

        if run:
            with st.spinner("Running inference..."):
                x = preprocess_image(pil_img, image_size=image_size)
                pred_name, pred_prob, top = predict_topk(model, x, class_names, top_k=top_k)

            st.markdown("---")
            st.markdown('<span class="badge">Prediction</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi">{pred_name}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi_sub">Confidence: {pred_prob*100:.2f}%</div>', unsafe_allow_html=True)
            st.progress(float(np.clip(pred_prob, 0.0, 1.0)))

            # ✅ FIXED: bar chart using a DataFrame with column names
            st.markdown("#### Top-K probabilities")
            df_bar = pd.DataFrame({
                "Class": [c for c, _ in top],
                "Probability (%)": [p * 100 for _, p in top]
            }).sort_values("Probability (%)", ascending=True)

            st.bar_chart(df_bar, x="Class", y="Probability (%)", height=220)

            # Exact table
            df_table = df_bar.sort_values("Probability (%)", ascending=False).reset_index(drop=True)
            st.dataframe(df_table, use_container_width=True, hide_index=True)

            # Download report
            if enable_download:
                report = make_report(pred_name, pred_prob, top, meta)
                download_stretch(
                    "Download result (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name="wheat_prediction_report.json",
                    mime="application/json",
                    key="download_report"
                )

            st.markdown("---")
            st.warning(
                "Educational / research tool only. Field diagnosis can be complex; confirm with agronomic inspection before decisions."
            )

        st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown(
    '<div class="small" style="margin-top:14px;">'
    'Model: ResNet50 • Output: 8-class softmax • Input: 0–255 pixels (model contains preprocessing layers)'
    "</div>",
    unsafe_allow_html=True
)
