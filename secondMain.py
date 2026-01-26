import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- HELPER FUNCTIONS ---

def create_dipole_template(size, angle_deg):
    pad = int(size * 0.5)
    canvas_size = size + 2 * pad
    template = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    center = (canvas_size // 2, canvas_size // 2)
    radius = size // 2
    
    mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.circle(mask, center, radius, 1, -1)
    
    half_mask = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    cv2.rectangle(half_mask, (0, 0), (center[0], canvas_size), 1, -1)
    
    template[mask == 1] = 1.0
    template[np.logical_and(mask == 1, half_mask == 1)] = -1.0
    
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(template, M, (canvas_size, canvas_size))
    
    y1, y2 = pad, pad + size
    x1, x2 = pad, pad + size
    return rotated[y1:y2, x1:x2]

def rotational_matched_filter(image, dipole_size, angular_step=15):
    # Preprocessing
    valid_mask = image > 10 
    mean_val = np.mean(image[valid_mask])
    img_float = (image.astype(np.float32) - mean_val).astype(np.float32)
    img_float[~valid_mask] = 0
    
    h, w = image.shape
    max_response_map = np.full((h, w), -np.inf, dtype=np.float32)
    best_angle_map = np.zeros((h, w), dtype=np.int16)
    
    template_dummy = create_dipole_template(dipole_size, 0)
    th, tw = template_dummy.shape
    pad_h = (h - (h - th + 1)) // 2 
    pad_w = (w - (w - tw + 1)) // 2
    
    print(f"Scanning 0 to 360 deg in steps of {angular_step}...")
    
    for angle in np.arange(0, 360, angular_step):
        kernel = create_dipole_template(dipole_size, angle).astype(np.float32)
        res = cv2.matchTemplate(img_float, kernel, cv2.TM_CCOEFF)
        res_h, res_w = res.shape
        full_res = cv2.copyMakeBorder(res, pad_h, h - res_h - pad_h, pad_w, w - res_w - pad_w, cv2.BORDER_CONSTANT, value=-np.inf)
        
        update_mask = full_res > max_response_map
        max_response_map[update_mask] = full_res[update_mask]
        best_angle_map[update_mask] = angle

    max_response_map[~valid_mask] = -np.inf
    return max_response_map, best_angle_map

def vectorized_peak_detection(heatmap, percentile_threshold, min_distance):
    valid_scores = heatmap[heatmap > -np.inf]
    if len(valid_scores) == 0: return [], heatmap
    
    thresh_val = np.percentile(valid_scores, percentile_threshold)
    print(f"Auto-calculated Threshold: {thresh_val:.2f} (Top {100-percentile_threshold:.2f}%)")
    
    mask = heatmap > thresh_val
    kernel_size = min_distance | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    safe_heatmap = heatmap.copy()
    safe_heatmap[safe_heatmap == -np.inf] = np.min(valid_scores)
    dilated = cv2.dilate(safe_heatmap, kernel)
    
    is_peak = (safe_heatmap == dilated) & mask
    y_coords, x_coords = np.where(is_peak)
    return list(zip(x_coords, y_coords)), safe_heatmap

# --- IMPROVED SHAPE VERIFICATION ---

def verify_shape_properties(image, candidates, size):
    """
    Filters out:
    1. Linear features (Roads/Ditches) - using Aspect Ratio
    2. Monopoles (Speckle noise) - using Dynamic Range Balance
    """
    confirmed = []
    rejected_linear = []
    rejected_balance = []
    rejected_weak = []  # New category for things that are too faint
    
    h_img, w_img = image.shape
    rad = int(size * 0.6) 

    # We need a minimum contrast to consider something a dipole.
    # If the difference between Darkest and Brightest pixel is < 60, it's probably just texture.
    MIN_DYNAMIC_RANGE = 60 

    print(f"Verifying {len(candidates)} candidates...")

    for (cx, cy) in candidates:
        x1, x2 = max(0, cx - rad), min(w_img, cx + rad)
        y1, y2 = max(0, cy - rad), min(h_img, cy + rad)
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0: continue

        # --- CHECK 1: INTENSITY & BALANCE (The Fix) ---
        
        # 1. Check Contrast (Is it loud enough?)
        min_val, max_val, _, _ = cv2.minMaxLoc(roi)
        dynamic_range = max_val - min_val
        
        if dynamic_range < MIN_DYNAMIC_RANGE:
            rejected_weak.append((cx, cy))
            continue
            
        # 2. Check Pole Balance (Trimodal Logic)
        # We define "Dark Pole" as the bottom 33% of the signal range
        # We define "Light Pole" as the top 33% of the signal range
        dark_thresh = min_val + (dynamic_range * 0.35)
        light_thresh = max_val - (dynamic_range * 0.35)
        
        dark_pixels = np.sum(roi < dark_thresh)
        light_pixels = np.sum(roi > light_thresh)
        total_pixels = roi.size
        
        # A valid dipole needs significant mass in BOTH extremes.
        # We require at least 5% of the area to be "Very Dark" and 5% to be "Very Light"
        if dark_pixels < (0.05 * total_pixels) or light_pixels < (0.05 * total_pixels):
            rejected_balance.append((cx, cy))
            continue

        # --- CHECK 2: GEOMETRY (Is it Round?) ---
        dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(dx, dy)
        
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        c = max(contours, key=cv2.contourArea)
        
        if len(c) >= 5:
            (x,y), (ma, MA), angle = cv2.fitEllipse(c)
        else:
            rect = cv2.minAreaRect(c)
            (x,y), (ma, MA), angle = rect
            
        if MA == 0: MA = 0.01
        
        axis_lengths = sorted([ma, MA])
        aspect_ratio = axis_lengths[0] / axis_lengths[1]
        
        # Reject lines (Roads usually have ratio < 0.4)
        if aspect_ratio < 0.5:
            rejected_linear.append((cx, cy))
            continue
            
        confirmed.append((cx, cy))
            
    print(f"  -> Kept: {len(confirmed)}")
    print(f"  -> Rejected (Linear/Roads): {len(rejected_linear)}")
    print(f"  -> Rejected (Weak/Faint): {len(rejected_weak)}")
    print(f"  -> Rejected (Monopoles/Imbalanced): {len(rejected_balance)}")
    
    # Return all rejection lists so we can visualize them if needed
    return confirmed, rejected_linear, rejected_weak, rejected_balance

# --- MAIN EXECUTION ---

# 1. Load Image
image_path = './renders/whole.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Could not load {image_path}.")
    exit()

# 2. Parameters
# Small artifact size usually requires careful tuning.
ARTIFACT_SIZE = 14      
ANGULAR_STEP = 15       

# LOWER PERCENTILE to catch faint artifacts (we rely on verification to remove the extra noise)
PERCENTILE = 99      

# 3. Process
print("Step 1: Matched Filter...")
response_map, angle_map = rotational_matched_filter(image, ARTIFACT_SIZE, ANGULAR_STEP)

print("Step 2: Peak Detection...")
# min_distance ensures we don't get overlapping circles
candidates, final_heatmap = vectorized_peak_detection(response_map, PERCENTILE, min_distance=int(ARTIFACT_SIZE*0.8))

print("Step 3: Verification...")
# Step 3: Verification...
confirmed_detections, linear_rejects, weak_rejects, balance_rejects = verify_shape_properties(image, candidates, ARTIFACT_SIZE)

# Visualization
output_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Draw Confirmed (RED)
for (x, y) in confirmed_detections:
    cv2.circle(output_img, (x, y), ARTIFACT_SIZE//2 + 2, (0, 0, 255), 2)

# Optional: Visualize what was rejected to tune parameters
# Yellow = Roads
# Blue   = Monopoles (speckles) or Faint Noise
for (x, y) in linear_rejects:
    cv2.circle(output_img, (x, y), 2, (0, 255, 255), -1) 
for (x, y) in balance_rejects:
    cv2.circle(output_img, (x, y), 2, (255, 0, 0), -1)
for (x, y) in weak_rejects:
    cv2.circle(output_img, (x, y), 2, (255, 0, 0), -1)

# Plotting
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
ax.set_title(f"Red = Confirmed ({len(confirmed_detections)}) | Yellow points = Rejected Linear Noise")
ax.axis('off')

plt.tight_layout()
plt.show()