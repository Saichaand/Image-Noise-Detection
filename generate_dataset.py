import cv2
import numpy as np
import os

os.makedirs("dataset", exist_ok=True)

classes = ["clean", "gaussian", "salt_pepper", "speckle", "motion_blur"]
for c in classes:
    os.makedirs(f"dataset/{c}", exist_ok=True)


def add_gaussian_noise(img):
    # IMPORTANT: generate noise in a signed dtype and clip AFTER adding.
    # Casting noise to uint8 first (the original bug) makes negative
    # values wrap around (e.g. -5 -> 251), producing garbage instead of
    # real Gaussian noise.
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img):
    out = img.copy()
    prob = 0.08  # fraction of pixels flipped to black/white

    rnd = np.random.rand(*img.shape[:2])
    out[rnd < prob] = 0
    out[rnd > 1 - prob] = 255

    return out


def add_speckle(img):
    noise = np.random.randn(*img.shape) * 0.1
    noisy = img + img * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_motion_blur(img):
    kernel = np.zeros((9, 9))
    kernel[4, :] = np.ones(9) / 9
    return cv2.filter2D(img, -1, kernel)


SOURCE = "clean_images"  # put your source photos here before running

if not os.path.isdir(SOURCE):
    raise FileNotFoundError(
        f"'{SOURCE}/' does not exist. Create it and add source photos "
        f"(any normal, roughly clean images) before running this script."
    )

files = [f for f in os.listdir(SOURCE)
         if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

if not files:
    raise FileNotFoundError(
        f"'{SOURCE}/' has no image files (.png/.jpg/.jpeg/.bmp)."
    )

count = 0
skipped = 0

for file in files:
    path = os.path.join(SOURCE, file)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️  Skipping unreadable file: {file}")
        skipped += 1
        continue

    img = cv2.resize(img, (128, 128))

    cv2.imwrite(f"dataset/clean/{count}.png", img)
    cv2.imwrite(f"dataset/gaussian/{count}.png", add_gaussian_noise(img))
    cv2.imwrite(f"dataset/salt_pepper/{count}.png", add_salt_pepper(img))
    cv2.imwrite(f"dataset/speckle/{count}.png", add_speckle(img))
    cv2.imwrite(f"dataset/motion_blur/{count}.png", add_motion_blur(img))

    count += 1

if count == 0:
    raise RuntimeError(
        "No images were processed successfully — dataset/ is empty. "
        "Check that your source images in clean_images/ are valid."
    )

print(f"✅ Dataset created! {count} source images processed "
      f"({count * len(classes)} total images), {skipped} skipped.")
