from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import rasterio
from scipy.ndimage import rotate
import time


# ===============================
# LOAD PARAMETERS
# ===============================


def get_parameters():
    inputfile = input("Source GeoTIFF Magnetogram Path [Sample.tiff]\n> ") or "Sample.tiff"
    resolution = input("Source GeoTIFF Magnetogram Resolution (cm/pixel) [10]\n> ") or 10
    anomaly_size = input("Diameter of the Dipole Signatures (meters) [2.3]\n> ") or 2.3
    lobe_distance = input("Separation between the two Gaussian lobes (meters) [0.3]\n> ") or 0.3
    sigma = input("Spread or blur of the Gaussian lobes [0.7]\n> ") or 0.7
    angle_step = input("Angular increment between successive template rotations (degrees) [15]\n> ") or 15
    threshold_percentile = input("Percentile threshold used to determine candidate detections [99]\n> ") or 99
    halo_radius = input("Inner radius used for halo validation (meters) [0.7]\n> ") or 0.7
    return inputfile, resolution, anomaly_size, lobe_distance, sigma, angle_step, threshold_percentile, halo_radius


# ===============================
# LOAD DATA
# ===============================


def load_geotiff(filepath):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        nodata = src.nodata

    if nodata is not None:
        mask = (data != nodata) & (~np.isnan(data))
    else:
        mask = ~np.isnan(data)

    data = data.astype(np.float32)
    data = -data
    return data, mask


# ===============================
# ANALYTIC SIGNAL
# ===============================


def compute_analytic_signal(data):
    dx = np.gradient(data, axis=1)
    dy = np.gradient(data, axis=0)
    analytic = np.sqrt(dx**2 + dy**2)
    return analytic


# ===============================
# CREATE DIPOLE TEMPLATE
# ===============================


def create_dipole_template(size, lobe_distance, sigma):
    half = size // 2
    y, x = np.mgrid[-half : half + 1, -half : half + 1]

    # Two Gaussian lobes
    g1 = np.exp(-((x - lobe_distance) ** 2 + y**2) / (2 * sigma**2))
    g2 = np.exp(-((x + lobe_distance) ** 2 + y**2) / (2 * sigma**2))

    dipole = g1 - g2

    # Neutral halo envelope
    r = np.sqrt(x**2 + y**2)
    halo = np.exp(-(r**2) / (2 * (size / 2.2) ** 2))

    template = dipole * halo

    # Zero mean normalization
    template -= np.mean(template)
    template /= np.linalg.norm(template) + 1e-9

    return template


# ===============================
# ROTATION-INVARIANT MATCHED FILTER
# ===============================


def run_rotated_matched_filter(data, template, angle_step):
    max_response = np.zeros_like(data)

    for angle in range(0, 180, angle_step):
        rotated = rotate(template, angle, reshape=False)
        rotated -= np.mean(rotated)
        rotated /= np.linalg.norm(rotated) + 1e-9

        response = cv2.filter2D(data, -1, rotated)

        max_response = np.maximum(max_response, response)

    return max_response


# ===============================
# PEAK DETECTION
# ===============================


def detect_candidates(response, artifact_size, threshold_percentile):
    thresh = np.percentile(response, threshold_percentile)
    candidate_mask = response > thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (artifact_size, artifact_size))

    dilated = cv2.dilate(response, kernel)
    peaks = (response == dilated) & candidate_mask

    y, x = np.where(peaks)
    return list(zip(x, y))


# ===============================
# HALO CHECK (Gray Surrounding)
# ===============================


def halo_check(window, anomaly_size, halo_radius):
    h, w = window.shape
    cy, cx = h // 2, w // 2

    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inner = window[r < halo_radius]
    outer = window[(r >= halo_radius) & (r <= anomaly_size // 2)]

    if len(outer) < 5:
        return False

    outer_std = np.std(outer)
    inner_std = np.std(inner)

    return outer_std < 0.5 * inner_std


def dipole_balance(window):
    pos = window[window > 0]
    neg = window[window < 0]

    if len(pos) < 5 or len(neg) < 5:
        return False

    pos_mean = np.mean(pos)
    neg_mean = np.mean(np.abs(neg))

    ratio = pos_mean / (neg_mean + 1e-6)

    return 0.6 < ratio < 1.6


# ===============================
# VISUALIZATION
# ===============================


def visualize(data, resolution, detections, artifact_size):
    norm = np.clip(data, -7, 7)
    norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    img = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    for x, y in detections:
        cv2.circle(img, (x, y), artifact_size // 2, (0, 255, 0), 2)

    cv2.imwrite(Path("results/detected_dipoles.png"), img)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlabel("meters")
    im = ax.imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        extent=[0, (img.shape[1] / resolution), 0, (img.shape[0] / resolution)],
    )
    cursor = mplcursors.cursor(im, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        x_val, y_val = sel.target
        sel.annotation.set_text(f"Value: {data[int(y_val), int(x_val)]:.1f} nT")

    plt.title(f"Detected Dipoles: {len(detections)}")
    plt.show()


# ===============================
# MAIN
# ===============================

inputfile, resolution, anomaly_size, lobe_distance, sigma, angle_step, threshold_percentile, halo_radius = (
    get_parameters()
)
inputfile = Path(inputfile)
anomaly_size = int(float(anomaly_size) * float(resolution))
lobe_distance = float(lobe_distance) * float(resolution)
sigma = float(sigma)
angle_step = int(angle_step)
threshold_percentile = float(threshold_percentile)
halo_radius = float(halo_radius) * float(resolution)

start_time = time.perf_counter()

print("Loading GeoTIFF…")
data, mask = load_geotiff(inputfile)

print("Computing analytic signal…")
analytic = compute_analytic_signal(data)

print("Creating dipole template…")
template = create_dipole_template(anomaly_size, lobe_distance, sigma)

print("Running rotation-invariant matched filter…")
response = run_rotated_matched_filter(analytic, template, angle_step)

print("Detecting candidates…")
candidates = detect_candidates(response, anomaly_size, threshold_percentile)

print(f"Found {len(candidates)} candidates.")

print("Applying halo validation…")
detected = []

half = anomaly_size // 2

for cx, cy in candidates:
    x1, x2 = cx - half, cx + half
    y1, y2 = cy - half, cy + half
    if x1 < 0 or y1 < 0 or x2 >= data.shape[1] or y2 >= data.shape[0]:
        continue
    window = data[y1:y2, x1:x2]
    if halo_check(window, anomaly_size, halo_radius) and dipole_balance(window):
        detected.append((cx, cy))

print(f"Detected {len(detected)} dipoles.")
print(f"Time for detection: {(time.perf_counter() - start_time):.3f} seconds")
visualize(data, resolution, detected, anomaly_size)
