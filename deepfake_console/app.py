import streamlit as st
import time
import random
import base64
import os
from model.inference import analyze_video

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Deepfake Detection Console", layout="centered")

# ---------- Helper: Encode image to base64 ----------
def encode_image_to_base64(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------- Load Backgrounds ----------
GREEN_BG = "assets/green_bg.jpg"
RED_BG = "assets/red_bg.jpg"
green_b64 = encode_image_to_base64(GREEN_BG)
red_b64 = encode_image_to_base64(RED_BG)

# ---------- BACKGROUND SETTER ----------
def set_background(image_b64: str):
    """Sets smooth fade transition background using CSS."""
    if not image_b64:
        st.markdown("<style>body { background-color: black; }</style>", unsafe_allow_html=True)
        return
    st.markdown(
        f"""
        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .stApp {{
            background-image: url("data:image/jpg;base64,{image_b64}");
            background-size: cover;
            background-position: center center;
            background-attachment: fixed;
            background-repeat: no-repeat;
            animation: fadeIn 1s ease-in-out;
        }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background: rgba(0,0,0,0);
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------- INITIAL BACKGROUND ----------
set_background(green_b64)

# ---------- STYLES ----------
st.markdown("""
<style>
h1, h3, h4, p {
    color: #00FFDD;
    text-align: center;
    font-family: 'Courier New', monospace;
}
.terminal {
    background-color: rgba(0, 30, 0, 0.85);
    color: #00ff66;
    font-family: 'Courier New', monospace;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 0 20px #00ff66;
    height: 240px;
    overflow-y: auto;
    font-size: 15px;
    margin-top: 25px;
}
.verdict-box {
    text-align: center;
    border-radius: 10px;
    font-family: 'Courier New', monospace;
    font-size: 24px;
    font-weight: bold;
    padding: 25px;
    box-shadow: 0 0 25px currentColor;
    margin-top: 35px;
    animation: fadeIn 1s ease-in-out;
}
.verdict-real {
    color: #00ff88;
    background-color: rgba(0, 30, 0, 0.8);
    border: 2px solid #00ff88;
}
.verdict-fake {
    color: #ff4444;
    background-color: rgba(30, 0, 0, 0.8);
    border: 2px solid #ff4444;
}
@keyframes fadeIn {
    from { opacity: 0; transform: scale(0.95); }
    to { opacity: 1; transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("<h1>🕵️‍♂️ Deepfake Detection Console</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a video for authenticity analysis...</p>", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], label_visibility="hidden")

if uploaded_file is not None:
    set_background(green_b64)  # Stay green until verdict

    result_placeholder = st.empty()
    terminal_output = "> Initializing deepfake detection system...<br>"

    fake_logs = [
        "Loading facial feature encoders...",
        "Extracting 60 keyframes...",
        "Calibrating neural signal integrity...",
        "Reconstructing motion vectors...",
        "Computing biometric authenticity signature...",
        "Cross-referencing deepfake probability layers...",
        "Decrypting latent representation...",
        "Analyzing final confidence metrics...",
        "Finalizing authenticity verdict..."
    ]

    # Typing effect simulation (no partial HTML rendering)
    for log in fake_logs:
        typed_line = "> "
        for ch in log:
            typed_line += ch
            # simulate typing but don't render partial HTML
            time.sleep(0.02)
        terminal_output += typed_line + "<br>"
        result_placeholder.markdown(f"<div class='terminal'>{terminal_output}</div>", unsafe_allow_html=True)
        time.sleep(random.uniform(0.5, 0.9))

    # ---------- RUN INFERENCE ----------
    result = analyze_video(uploaded_file)
    result_placeholder.empty()

    if "error" in result:
        st.error(result["error"])
    else:
        real = result.get("real", 0.0)
        fake = result.get("fake", 0.0)
        verdict_is_real = real > fake

        if verdict_is_real:
            set_background(green_b64)
            verdict_html = f"""
                <div class='verdict-box verdict-real'>
                    🟢 AUTHENTIC MATERIAL VERIFIED<br>
                    Real Probability: {real}%<br>
                    Fake Probability: {fake}%
                </div>
            """
        else:
            set_background(red_b64)
            verdict_html = f"""
                <div class='verdict-box verdict-fake'>
                    🔴 SYNTHETIC SOURCE DETECTED<br>
                    Real Probability: {real}%<br>
                    Fake Probability: {fake}%
                </div>
            """

        result_placeholder.markdown(verdict_html, unsafe_allow_html=True)



