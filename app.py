"""
Image Noise Detection & Denoising Web Application
===================================================
A Streamlit-based web app that:
1. Accepts image upload or camera capture
2. Detects noise type automatically
3. Applies appropriate denoising filter
4. Offers optional enhancement filters
4. Allows download of the final image

Author: Student Project
Tech Stack: Python, OpenCV, NumPy, Streamlit
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


@st.cache_resource
def get_model():
    """
    Load the CNN once per session instead of on every Streamlit rerun.
    Without caching, this would reload the whole model from disk every
    time the user touches a slider or button, since Streamlit re-runs
    the entire script top-to-bottom on every interaction.
    """
    return load_model("cnn_noise_model.h5")


model = get_model()
# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ImageCure · Noise Detector & Denoiser",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* === Global Reset === */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* === App Background === */
.stApp {
    background: #0a0e1a;
    color: #e8eaf0;
}

/* === Main container padding === */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* === Hero Header === */
.hero-header {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2744 50%, #0d1b2a 100%);
    border: 1px solid rgba(100, 180, 255, 0.2);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(64, 140, 255, 0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #64b4ff;
    letter-spacing: -1px;
    margin: 0 0 0.4rem 0;
    line-height: 1.1;
}
.hero-sub {
    font-size: 1rem;
    color: #7a8aaa;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(100, 180, 255, 0.12);
    border: 1px solid rgba(100, 180, 255, 0.3);
    color: #64b4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.1em;
}

/* === Section Cards === */
.section-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #64b4ff;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    background: #64b4ff;
    border-radius: 50%;
}

/* === Noise Badge === */
.noise-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(90deg, rgba(100,180,255,0.15), rgba(100,180,255,0.05));
    border: 1px solid rgba(100, 180, 255, 0.4);
    border-radius: 10px;
    padding: 0.8rem 1.4rem;
    margin: 0.6rem 0;
}
.noise-type-text {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #a8d4ff;
}
.noise-icon {
    font-size: 1.4rem;
}

/* === Confidence Bar === */
.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #1e6fff, #64b4ff);
    transition: width 0.8s ease;
}

/* === Filter Info Box === */
.filter-info {
    background: rgba(30, 111, 255, 0.08);
    border-left: 3px solid #1e6fff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.88rem;
    color: #a0b4cc;
    margin-top: 0.8rem;
}

/* === Metric Tiles === */
.metric-row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 0.8rem;
}
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #64b4ff;
}
.metric-label {
    font-size: 0.72rem;
    color: #5a6a80;
    margin-top: 2px;
    letter-spacing: 0.06em;
}

/* === Sidebar Styling === */
[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Space Mono', monospace;
    color: #64b4ff;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
}

/* === Buttons === */
.stButton > button {
    background: linear-gradient(135deg, #1e4fff, #1e6fff);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2a5bff, #2a7aff);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(30, 111, 255, 0.35);
}

/* === Download button === */
.stDownloadButton > button {
    background: linear-gradient(135deg, #0a3a1a, #0d5a28) !important;
    border: 1px solid rgba(50, 200, 100, 0.3) !important;
    color: #50e090 !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    border-radius: 8px;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #0d5a28, #107030) !important;
    box-shadow: 0 4px 15px rgba(50, 200, 100, 0.2) !important;
}

/* === Divider === */
.custom-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(100,180,255,0.2), transparent);
    margin: 1.5rem 0;
}

/* === Image captions === */
.img-caption {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    color: #4a5a70;
    text-align: center;
    margin-top: 6px;
    text-transform: uppercase;
}

/* === Selectbox, Slider === */
.stSelectbox label, .stSlider label, .stRadio label {
    color: #7a8aaa !important;
    font-size: 0.85rem !important;
}

/* === Success / Info messages === */
.stSuccess {
    background: rgba(30, 180, 80, 0.1) !important;
    border: 1px solid rgba(30, 180, 80, 0.3) !important;
    color: #50e090 !important;
    border-radius: 8px !important;
}
.stInfo {
    background: rgba(30, 111, 255, 0.1) !important;
    border: 1px solid rgba(30, 111, 255, 0.3) !important;
    color: #64b4ff !important;
    border-radius: 8px !important;
}
.stWarning {
    background: rgba(255, 180, 30, 0.08) !important;
    border: 1px solid rgba(255, 180, 30, 0.25) !important;
    border-radius: 8px !important;
}

/* Hide default Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MODULE 1: NOISE DETECTION
# ═══════════════════════════════════════════════════════════════

def compute_image_stats(gray_img):
    """
    Compute statistical features from a grayscale image
    used for noise classification.
    Returns a dictionary of features.
    """
    # Laplacian variance – measures sharpness / high-freq content
    laplacian_var = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    # Mean and std of pixel intensities
    mean_val = float(np.mean(gray_img))
    std_val  = float(np.std(gray_img))

    # Percentage of near-black (≤10) and near-white (≥245) pixels
    total_pixels = gray_img.size
    black_px = np.sum(gray_img <= 10) / total_pixels * 100
    white_px = np.sum(gray_img >= 245) / total_pixels * 100

    # Gradient magnitude via Sobel – captures edge/blur info
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2).mean()

    # Horizontal vs vertical gradient ratio → motion blur hint
    h_grad = np.abs(sobelx).mean()
    v_grad = np.abs(sobely).mean()
    hv_ratio = h_grad / (v_grad + 1e-6)

    # Skewness of pixel distribution
    skewness = float(((gray_img.astype(np.float32) - mean_val) ** 3).mean() / (std_val**3 + 1e-6))

    # Coefficient of Variation of Local Variance (CVLV) – speckle indicator
    kernel = np.ones((5, 5), np.float32) / 25
    local_mean = cv2.filter2D(gray_img.astype(np.float32), -1, kernel)
    local_var  = cv2.filter2D((gray_img.astype(np.float32) - local_mean)**2, -1, kernel)
    cvlv = float(np.std(local_var) / (np.mean(local_var) + 1e-6))

    return {
        "laplacian_var": laplacian_var,
        "mean": mean_val,
        "std": std_val,
        "black_pct": black_px,
        "white_pct": white_px,
        "gradient_mag": gradient_mag,
        "hv_ratio": hv_ratio,
        "skewness": skewness,
        "cvlv": cvlv,
    }


def detect_noise_type(img_bgr):
    # Convert to grayscale (for stats)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ✅ KEEP stats (important for UI)
    stats = compute_image_stats(gray)

    # ✅ CNN input preprocessing
    img_resized = cv2.resize(img_bgr, (128, 128))
    img_resized = img_resized / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # ✅ Predict using CNN
    preds = model.predict(img_input)[0]

    class_names = ["clean", "gaussian", "salt_pepper", "speckle", "motion_blur"]

    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    # Map to UI labels
    label_map = {
        "clean": "Low Noise / Clean",
        "gaussian": "Gaussian",
        "salt_pepper": "Salt & Pepper",
        "speckle": "Speckle",
        "motion_blur": "Motion Blur"
    }

    noise_type = label_map.get(pred_class, "Unknown")

    # All class scores
    scores = dict(zip(class_names, preds))

    # ✅ RETURN EVERYTHING (THIS FIXES YOUR ERROR)
    return noise_type, round(confidence, 2), stats, scores

# ═══════════════════════════════════════════════════════════════
# MODULE 2: DENOISING FILTERS
# ═══════════════════════════════════════════════════════════════

def denoise_salt_and_pepper(img):
    # Convert to grayscale for mask detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect salt & pepper pixels (very dark or very bright)
    mask = (gray <= 10) | (gray >= 245)

    # Apply median filter
    median = cv2.medianBlur(img, 5)

    # Replace ONLY noisy pixels
    result = img.copy()
    result[mask] = median[mask]
    
    return result


def denoise_gaussian(img):
    # Step 1: Non-local means (main denoising)
    if len(img.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            img, None, 10, 10, 7, 21
        )
    else:
        denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

    # Step 2: Edge restoration
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    sharpened = cv2.addWeighted(denoised, 1.4, blurred, -0.4, 0)

    return sharpened


def denoise_speckle(img):
    # Step 1: Bilateral filter (edge preserving)
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Enhance details
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return sharpened


def denoise_motion_blur(img):
    # Step 1: Mild smoothing
    smooth = cv2.GaussianBlur(img, (3, 3), 0)

    # Step 2: Strong unsharp masking
    sharpened = cv2.addWeighted(img, 1.8, smooth, -0.8, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)

def denoise_poisson(img):
    # Step 1: Light Gaussian smoothing
    denoised = cv2.GaussianBlur(img, (3, 3), 0.8)

    # Step 2: Restore contrast
    sharpened = cv2.addWeighted(denoised, 1.3, img, -0.3, 0)

    return sharpened


def denoise_uniform(img):
    # Step 1: Median (small)
    med = cv2.medianBlur(img, 3)

    # Step 2: Bilateral (refinement)
    denoised = cv2.bilateralFilter(med, d=7, sigmaColor=50, sigmaSpace=50)

    # Step 3: Sharpen
    blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
    sharpened = cv2.addWeighted(denoised, 1.4, blurred, -0.4, 0)

    return sharpened


DENOISER_MAP = {
    "Salt & Pepper": denoise_salt_and_pepper,
    "Gaussian":      denoise_gaussian,
    "Speckle":       denoise_speckle,
    "Motion Blur":   denoise_motion_blur,
    "Poisson":       denoise_poisson,
    "Uniform":       denoise_uniform,
}

FILTER_DESCRIPTION = {
    "Salt & Pepper": "Median Filter (5×5) — replaces each pixel with the neighbourhood median, eliminating isolated bright/dark spike pixels.",
    "Gaussian":      "Non-Local Means Denoising — computes a weighted average over non-local similar patches, excellent for random additive noise.",
    "Speckle":       "Bilateral Filter (d=9, σ=75) — edge-preserving smoothing based on spatial and intensity similarity.",
    "Motion Blur":   "Unsharp Masking — amplifies high-frequency details lost to motion blur for perceived sharpness recovery.",
    "Poisson":       "Gaussian Blur (σ=1.0) — mild low-pass filter appropriate for signal-dependent photon shot noise.",
    "Uniform":       "Median + Bilateral combo — median removes outliers, bilateral smooths the residual additive noise.",
}

NOISE_ICONS = {
    "Salt & Pepper": "🧂",
    "Gaussian":      "🌫️",
    "Speckle":       "✨",
    "Motion Blur":   "💨",
    "Poisson":       "⚡",
    "Uniform":       "📊",
}


def auto_denoise(img_bgr, noise_type):

    # 🔥 override if salt-pepper pattern detected
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    extreme_pixels = np.mean((gray <= 10) | (gray >= 245))

    if extreme_pixels > 0.03:   # 3% pixels extreme → salt & pepper
        return denoise_salt_and_pepper(img_bgr)

    if noise_type in ["Low Noise / Clean", "Uncertain"]:
        return img_bgr

    fn = DENOISER_MAP.get(noise_type, denoise_gaussian)
    return fn(img_bgr)


# ═══════════════════════════════════════════════════════════════
# MODULE 3: ENHANCEMENT FILTERS
# ═══════════════════════════════════════════════════════════════

def apply_clahe(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Applies equalisation locally to avoid over-amplifying noise
    in homogeneous regions.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def apply_contrast(img_bgr, alpha=1.3, beta=10):
    """
    Linear contrast and brightness adjustment.
    output = alpha * input + beta
    alpha > 1 increases contrast; beta shifts brightness.
    """
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return adjusted


def apply_gamma(img_bgr, gamma=1.2):
    """
    Gamma correction via a lookup table.
    gamma < 1 brightens; gamma > 1 darkens.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def apply_sharpening(img_bgr, strength=1.0):
    """
    Laplacian-based sharpening.
    Adds scaled high-frequency Laplacian to the image.
    strength controls the sharpening intensity.
    """
    kernel = np.array([[0, -1,  0],
                       [-1, 4+1/strength, -1],
                       [0, -1,  0]], dtype=np.float32) * strength
    sharpened = cv2.filter2D(img_bgr, -1, kernel)
    result = cv2.addWeighted(img_bgr, 1.0, sharpened, 0.6 * strength, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# MODULE 4: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def pil_to_bgr(pil_img):
    """Convert PIL Image (RGB) → OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(bgr_img):
    """Convert OpenCV BGR numpy array → PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def img_to_download_bytes(pil_img, fmt="PNG"):
    """Encode a PIL image to bytes for Streamlit download."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    buf.seek(0)
    return buf.getvalue()


def compute_psnr(original, processed):
    """
    Peak Signal-to-Noise Ratio between two images.
    Higher PSNR → less distortion introduced by processing.
    """
    orig = original.astype(np.float32)
    proc = processed.astype(np.float32)
    mse = np.mean((orig - proc) ** 2)
    if mse == 0:
        return float('inf')
    return round(20 * np.log10(255.0 / np.sqrt(mse)), 2)


def image_entropy(gray):
    """Shannon entropy of an image histogram – measures information / detail."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
    return round(float(entropy), 2)

# ─────────────────────────────────────────────
# IMAGE FORMAT CONVERSIONS
# ─────────────────────────────────────────────

def convert_to_grayscale(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray


def convert_to_binary(img_bgr):
    gray = convert_to_grayscale(img_bgr)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


def convert_to_hsi(img_bgr):
    # Convert BGR → RGB
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    R, G, B = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # Intensity
    I = (R + G + B) / 3.0

    # Saturation
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6)) * min_rgb

    # Hue
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2*np.pi)

    # Stack HSI
    hsi = np.stack((H, S, I), axis=-1)

    # Scale to 0–255 for display
    hsi = (hsi * 255).astype(np.uint8)

    return hsi

# ═══════════════════════════════════════════════════════════════
# MODULE 5: UI RENDERING HELPERS
# ═══════════════════════════════════════════════════════════════

def render_image_card(label, img_bgr, caption=""):
    """Render a styled image panel with label."""
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)
    pil_img = bgr_to_pil(img_bgr)
    st.image(pil_img, use_container_width=True)
    if caption:
        st.markdown(f'<div class="img-caption">{caption}</div>', unsafe_allow_html=True)


def render_noise_result(noise_type, confidence, scores):
    """Display noise detection result with confidence bar."""
    
    icon = NOISE_ICONS.get(noise_type, "🔍")
    st.markdown(f"""
    <div class="noise-badge">
        <span class="noise-icon">{icon}</span>
        <div>
            <div style="font-size:0.72rem;color:#5a6a80;letter-spacing:0.1em;font-family:'Space Mono',monospace;">DETECTED NOISE</div>
            <div class="noise-type-text">{noise_type}</div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.7rem;color:#5a6a80;">Confidence</div>
            <div style="font-family:'Space Mono',monospace;color:#64b4ff;font-size:1rem;">{int(confidence*100)}%</div>
        </div>
    </div>
    <div class="conf-bar-wrap">
        <div class="conf-bar-fill" style="width:{int(confidence*100)}%"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if noise_type == "Clean Image":
        st.success("✅ Image is already clean. No denoising applied.")

    # Show all scores
    with st.expander("🔎 All noise scores", expanded=False):
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for ntype, sc in sorted_scores:
            pct = int(sc * 100)
            ico = NOISE_ICONS.get(ntype, "")
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="width:20px;text-align:center;">{ico}</span>
                <span style="width:120px;font-size:0.82rem;color:#a0b0c0;">{ntype}</span>
                <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:6px;overflow:hidden;">
                    <div style="width:{pct}%;height:100%;background:{'#1e6fff' if ntype==noise_type else '#2a3550'};border-radius:4px;"></div>
                </div>
                <span style="font-family:'Space Mono',monospace;font-size:0.78rem;color:{'#64b4ff' if ntype==noise_type else '#4a5a70'};width:35px;text-align:right;">{pct}%</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="filter-info">
        <b>Applied Filter:</b> {FILTER_DESCRIPTION.get(noise_type, "—")}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Hero Header ─────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">IMAGE PROCESSING · STUDENT PROJECT</div>
        <div class="hero-title">ImageCure</div>
        <div class="hero-sub">Automatic noise detection · intelligent denoising · enhancement studio</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ SETTINGS")
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown("**Input Source**")
        input_mode = st.radio("Select Input", ["📁 Upload Image", "📷 Camera Capture"],
                       label_visibility="collapsed")

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        st.markdown("**Enhancement Filters**")
        st.caption("Applied after denoising (optional)")

        use_clahe = st.checkbox("🌟 CLAHE (Adaptive Contrast)", value=False)
        if use_clahe:
            clahe_clip = st.slider("Clip Limit", 1.0, 8.0, 2.0, 0.5)

        use_contrast = st.checkbox("🎚️ Contrast / Brightness", value=False)
        if use_contrast:
            alpha = st.slider("Alpha (contrast)", 0.5, 3.0, 1.3, 0.05)
            beta  = st.slider("Beta (brightness)", -80, 80, 10, 5)

        use_gamma = st.checkbox("☀️ Gamma Correction", value=False)
        if use_gamma:
            gamma = st.slider("Gamma", 0.3, 3.0, 1.0, 0.05)

        use_sharp = st.checkbox("🔪 Sharpening", value=False)
        if use_sharp:
            sharp_str = st.slider("Sharpening Strength", 0.1, 3.0, 1.0, 0.1)
        st.markdown("**Output Format Conversion**")

        convert_option = st.selectbox(
            "Convert Final Image To:",
            ["None", "Grayscale", "Binary", "HSI"]
        )

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:#3a4a60;line-height:1.6;">
        <b style="color:#4a5a70;">How it works</b><br>
        1. Upload or capture an image<br>
        2. Statistical analysis detects noise<br>
        3. Best-fit filter is applied<br>
        4. Optional manual enhancements<br>
        5. Download your cleaned image
        </div>
        """, unsafe_allow_html=True)

    # ── Image Input ──────────────────────────────────────────
    img_bgr = None

    if "Upload" in input_mode:
        uploaded = st.file_uploader(
            "Drop an image here or click to browse",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            help="Supports JPG, PNG, BMP, TIFF, WebP"
        )
        if uploaded:
            pil_img = Image.open(uploaded)
            img_bgr = pil_to_bgr(pil_img)
    else:
        cam_img = st.camera_input("📷 Take a photo")
        if cam_img:
            pil_img = Image.open(cam_img)
            img_bgr = pil_to_bgr(pil_img)

    # ── Processing Pipeline ──────────────────────────────────
    if img_bgr is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#2a3550;">
            <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.9rem;color:#3a4a60;">
                Upload an image or use your camera to get started
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Resize if too large (for performance)
    h, w = img_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

    # ── Step 1: Show original ────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        render_image_card("📸 ORIGINAL IMAGE", img_bgr,
                          f"{img_bgr.shape[1]} × {img_bgr.shape[0]} px")

    # ── Step 2: Noise Detection ──────────────────────────────
    with st.spinner("🔍 Analysing image for noise patterns…"):
        noise_type, confidence, stats, all_scores = detect_noise_type(img_bgr)
    with st.expander("📊 Debug Stats"):
        st.json(stats)
    with col2:
        st.markdown('<div class="section-label">🧪 NOISE ANALYSIS</div>', unsafe_allow_html=True)
        render_noise_result(noise_type, confidence, all_scores)

        # Image statistics
        gray_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value">{stats['std']:.1f}</div>
                <div class="metric-label">Std Dev</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{stats['laplacian_var']:.0f}</div>
                <div class="metric-label">Laplacian Var</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{image_entropy(gray_orig)}</div>
                <div class="metric-label">Entropy (bits)</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{stats['black_pct']+stats['white_pct']:.1f}%</div>
                <div class="metric-label">Extreme Pixels</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Step 3: Denoising ────────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    with st.spinner(f"✨ Applying {noise_type} denoiser…"):
        if noise_type in ["Low Noise / Clean", "Uncertain"]:
            denoised_bgr = img_bgr.copy()   # 🚀 keep original (no blur)
        else:
           denoised_bgr = auto_denoise(img_bgr, noise_type)

    # ── Step 4: Enhancement ──────────────────────────────────
    enhanced_bgr = denoised_bgr.copy()
    converted_img = None

    if convert_option == "Grayscale":
        converted_img = convert_to_grayscale(enhanced_bgr)

    elif convert_option == "Binary":
        converted_img = convert_to_binary(enhanced_bgr)

    elif convert_option == "HSI":
        converted_img = convert_to_hsi(enhanced_bgr)

    if use_clahe:
        enhanced_bgr = apply_clahe(enhanced_bgr, clip_limit=clahe_clip)
    if use_contrast:
        enhanced_bgr = apply_contrast(enhanced_bgr, alpha=alpha, beta=beta)
    if use_gamma:
        enhanced_bgr = apply_gamma(enhanced_bgr, gamma=gamma)
    if use_sharp and noise_type != "Salt & Pepper":
        enhanced_bgr = apply_sharpening(enhanced_bgr, strength=sharp_str)
    
    converted_img = None

    if convert_option == "Grayscale":
        converted_img = convert_to_grayscale(enhanced_bgr)

    elif convert_option == "Binary":
        converted_img = convert_to_binary(enhanced_bgr)

    elif convert_option == "HSI":
        converted_img = convert_to_hsi(enhanced_bgr)


    # ── Display Results ──────────────────────────────────────
    col3, col4 = st.columns(2, gap="large")

    with col3:
        render_image_card(
            f"✅ DENOISED  ·  {noise_type.upper()} FILTER",
            denoised_bgr,
            "Noise removed with auto-selected filter"
        )
        psnr_val = compute_psnr(img_bgr, denoised_bgr)
        gray_den = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2GRAY)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-tile">
                <div class="metric-value">{psnr_val if psnr_val != float('inf') else '∞'}</div>
                <div class="metric-label">PSNR (dB)</div>
            </div>
            <div class="metric-tile">
                <div class="metric-value">{image_entropy(gray_den)}</div>
                <div class="metric-label">Entropy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        enhancements_applied = [
            e for e, flag in [
                ("CLAHE", use_clahe),
                ("Contrast", use_contrast),
                ("Gamma", use_gamma),
                ("Sharpening", use_sharp),
            ] if flag
        ]
        label = "🎨 FINAL ENHANCED IMAGE" if enhancements_applied else "🎨 FINAL IMAGE (No Extra Enhancement)"
        caption_extra = "Enhanced: " + ", ".join(enhancements_applied) if enhancements_applied else "Enable filters in the sidebar ←"
        render_image_card(label, enhanced_bgr, caption_extra)

        # Download button
        final_pil = bgr_to_pil(enhanced_bgr)
        dl_bytes  = img_to_download_bytes(final_pil, "PNG")
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️  Download Final Image",
            data=dl_bytes,
            file_name=f"imagecure_{noise_type.lower().replace(' ','_')}_denoised.png",
            mime="image/png",
            use_container_width=True,
        )
        
        # ✅ ===== CONVERTED IMAGE DISPLAY =====
        if converted_img is not None:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-label">🔄 CONVERTED IMAGE</div>', unsafe_allow_html=True)

            if convert_option in ["Grayscale", "Binary"]:
                st.image(converted_img, clamp=True, use_container_width=True)
            else:
                st.image(converted_img, use_container_width=True)

            # ✅ ===== DOWNLOAD CONVERTED IMAGE =====
            if len(converted_img.shape) == 2:
                pil_conv = Image.fromarray(converted_img)
            else:
                pil_conv = Image.fromarray(converted_img)

            conv_bytes = img_to_download_bytes(pil_conv, "PNG")

            st.download_button(
                label=f"⬇️ Download {convert_option} Image",
                data=conv_bytes,
                file_name=f"converted_{convert_option.lower()}.png",
                mime="image/png",
                use_container_width=True,
            )


    # ── Histogram Comparison ─────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 PIXEL INTENSITY HISTOGRAMS</div>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5),
                              facecolor="#0a0e1a")
    images_for_hist = [
        (img_bgr,      "Original",         "#4a7fff"),
        (denoised_bgr, "Denoised",          "#50e090"),
        (enhanced_bgr, "Final Enhanced",    "#ff8c50"),
    ]

    for ax, (im, title, colour) in zip(axes, images_for_hist):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ax.hist(gray.ravel(), bins=128, color=colour, alpha=0.85,
                edgecolor="none", linewidth=0)
        ax.set_title(title, color="#a0b4cc", fontsize=9,
                     fontfamily="monospace", pad=6)
        ax.set_facecolor("#0d1220")
        ax.tick_params(colors="#3a4a60", labelsize=7)
        ax.spines[:].set_color("#1a2440")
        ax.set_xlabel("Pixel Value", color="#3a4a60", fontsize=7)
        ax.set_ylabel("Count",       color="#3a4a60", fontsize=7)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Footer ───────────────────────────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;font-size:0.72rem;color:#2a3550;font-family:'Space Mono',monospace;padding:1rem 0 0.5rem;">
        ImageCure · Digital Image Processing · Built with Python, OpenCV & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
