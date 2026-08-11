# 🔬 ImageCure – CNN-Based Noise Detection & Denoising

A smart web application that automatically detects noise in images using a CNN model and applies the best denoising technique, with additional enhancement and format conversion features.

---

## 🚀 Features

### 🧠 Noise Detection (CNN-Based)
- Detects:
  - Gaussian Noise
  - Salt & Pepper Noise
  - Speckle Noise
  - Motion Blur
  - Clean Image
- Shows confidence score

### 🧹 Automatic Denoising
- Applies best filter based on detected noise:
  - Median Filter (Salt & Pepper)
  - Non-Local Means (Gaussian)
  - Bilateral Filter (Speckle)
  - Sharpening (Motion Blur)

### 🎨 Image Enhancement
- CLAHE (Adaptive Contrast)
- Brightness & Contrast adjustment
- Gamma correction
- Sharpening

### 🔄 Image Format Conversion
- Convert final image to:
  - Grayscale
  - Binary
  - HSI

### ⬇️ Download Options
- Download final enhanced image
- Download converted image

---

## 🖥️ Tech Stack

- Python
- OpenCV
- NumPy
- TensorFlow / Keras (CNN Model)
- Streamlit (Web App UI)
- Matplotlib

---

## 📂 Project Structure
image-noise-detector/
│
├── app.py # Streamlit Web App
├── train_model.py # CNN Model Training Script
├── cnn_noise_model.h5 # Trained Model (not included in repo)
├── dataset/ # Training dataset (not included)
├── requirements.txt
└── README.md


---

## ⚙️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

pip install -r requirements.txt

python train_model.py

cnn_noise_model.h5
