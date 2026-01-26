import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

# --- HELPER FUNCTIONS ---

def load_geotiff(filepath):
    """
    Loads a GeoTIFF and handles NoData/NaN values.
    Returns:
        data_float (np.array): Raw magnetic data (Float32)
        valid_mask (np.array): Boolean mask (True where data exists)
        profile (dict): Metadata to save GeoTIFFs later if needed
    """
    with rasterio.open(filepath) as src:
        # Read the first band
        data = src.read(1)
        profile = src.profile
        nodata_val = src.nodata

    # Create a mask of valid pixels
    # Valid if NOT nodata AND NOT NaN (Not a Number)
    if nodata_val is not None:
        valid_mask = (data != nodata_val) & (~np.isnan(data))
    else:
        valid_mask = ~np.isnan(data)

    # Set invalid areas to 0 (or mean) to prevent math errors, 
    # but we will rely on valid_mask for processing.
    data_float = data.copy()
    data_float[~valid_mask] = np.nanmean(data[valid_mask]) # Fill voids with mean for stability
    
    return data_float, valid_mask, profile

def normalize_for_visualization(data, valid_mask):
    """
    Converts raw nT data to 0-255 uint8 for Shape Verification and PNG export.
    Uses percentile clipping to handle extreme magnetic outliers.
    """
    # Get valid data only
    valid_data = data[valid_mask]
    
    # Clip extremes (e.g., bottom 1% and top 1%) to improve contrast
    vmin, vmax = np.percentile(valid_data, [1, 99])
    clipped = np.clip(data, vmin, vmax)
    
    # Normalize to 0-255
    norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)
    
    # Black out the invalid regions
    norm[~valid_mask] = 0
    
    return norm

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

def rotational_matched_filter(data_float, valid_mask, dipole_size, angular_step=15):
    # Prepare Data: Subtract mean of VALID area only
    # Note: We use the Raw Float data here for physics-accurate matching
    mean_val = np.mean(data_float[valid_mask])
    img_centered = data_float - mean_val
    img_centered[~valid_mask] = 0 # Zero out noise in void areas
    
    h, w = data_float.shape
    max_response_map = np.full((h, w), -np.inf, dtype=np.float32)
    best_angle_map = np.zeros((h, w), dtype=np.int16)
    
    template_dummy = create_dipole_template(dipole_size, 0)
    th, tw = template_dummy.shape
    pad_h = (h - (h - th + 1)) // 2 
    pad_w = (w - (w - tw + 1)) // 2
    
    print(f"Scanning 0 to 360 deg in steps of {angular_step}...")
    
    for angle in np.arange(0, 360, angular_step):
        kernel = create_dipole_template(dipole_size, angle).astype(np.float32)
        
        # Match Template on Raw Float Data
        res = cv2.matchTemplate(img_centered.astype(np.float32), kernel, cv2.TM_CCOEFF)
        
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

def verify_shape_properties(image_uint8, candidates, size):
    """
    Uses the 0-255 version of the image to check shapes.
    (Contours work best on standard integer ranges).
    """
    confirmed = []
    rejected_linear = []
    rejected_balance = []
    rejected_weak = []
    
    h_img, w_img = image_uint8.shape
    rad = int(size * 0.6) 

    # Tuned for 0-255 images
    MIN_DYNAMIC_RANGE = 40 

    print(f"Verifying {len(candidates)} candidates...")

    for (cx, cy) in candidates:
        x1, x2 = max(0, cx - rad), min(w_img, cx + rad)
        y1, y2 = max(0, cy - rad), min(h_img, cy + rad)
        roi = image_uint8[y1:y2, x1:x2]
        
        if roi.size == 0: continue

        # --- CHECK 1: DYNAMIC RANGE ---
        min_val, max_val, _, _ = cv2.minMaxLoc(roi)
        dynamic_range = max_val - min_val
        
        if dynamic_range < MIN_DYNAMIC_RANGE:
            rejected_weak.append((cx, cy))
            continue
            
        # --- CHECK 2: BALANCE ---
        dark_thresh = min_val + (dynamic_range * 0.35)
        light_thresh = max_val - (dynamic_range * 0.35)
        
        dark_pixels = np.sum(roi < dark_thresh)
        light_pixels = np.sum(roi > light_thresh)
        total_pixels = roi.size
        
        if dark_pixels < (0.05 * total_pixels) or light_pixels < (0.05 * total_pixels):
            rejected_balance.append((cx, cy))
            continue

        # --- CHECK 3: GEOMETRY ---
        dx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(dx, dy)
        
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, thresh = cv2.threshold(mag_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
            
        c = max(contours, key=cv2.contourArea)
        
        if len(c) >= 5:
            (x,y), (ma, MA), angle = cv2.fitEllipse(c)
        else:
            rect = cv2.minAreaRect(c)
            (x,y), (ma, MA), angle = rect
            
        if MA == 0: MA = 0.01
        
        axis_lengths = sorted([ma, MA])
        aspect_ratio = axis_lengths[0] / axis_lengths[1]
        
        if aspect_ratio < 0.5:
            rejected_linear.append((cx, cy))
            continue
            
        confirmed.append((cx, cy))
            
    print(f"  -> Kept: {len(confirmed)}")
    return confirmed, rejected_linear, rejected_weak, rejected_balance

# --- MAIN EXECUTION ---

# 1. Load GeoTIFF
file_path = 'your_magnetometry_data.tif'  # UPDATE THIS
print(f"Loading GeoTIFF: {file_path}")

try:
    raw_data, valid_mask, profile = load_geotiff(file_path)
except Exception as e:
    print(f"Error loading GeoTIFF: {e}")
    exit()

print(f"Data Loaded. Shape: {raw_data.shape}")

# 2. Parameters
ARTIFACT_SIZE = 14      
ANGULAR_STEP = 15       
PERCENTILE = 98.5      

# 3. Process (Uses RAW Float Data)
print("Step 1: Matched Filter (Physics Based)...")
response_map, angle_map = rotational_matched_filter(raw_data, valid_mask, ARTIFACT_SIZE, ANGULAR_STEP)

print("Step 2: Peak Detection...")
candidates, final_heatmap = vectorized_peak_detection(response_map, PERCENTILE, min_distance=int(ARTIFACT_SIZE*0.8))

# 4. Prepare Visualization Image (Float -> 0-255)
# We need this 0-255 version specifically for the Shape Verification algorithms 
# (which rely on standard image histograms) and for the final PNG output.
print("Step 3: Creating Visualization Image...")
vis_image = normalize_for_visualization(raw_data, valid_mask)

print("Step 4: Verification...")
confirmed_detections, linear_rejects, weak_rejects, balance_rejects = verify_shape_properties(vis_image, candidates, ARTIFACT_SIZE)

# 5. Output PNG
output_img = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

# Draw Confirmed (RED)
for (x, y) in confirmed_detections:
    cv2.circle(output_img, (x, y), ARTIFACT_SIZE//2 + 2, (0, 0, 255), 1)

# Save results
output_filename = "magnetometry_results.png"
cv2.imwrite(output_filename, output_img)
print(f"Saved results to {output_filename}")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
ax.set_title(f"Final Analysis on GeoTIFF Data ({len(confirmed_detections)} found)")
ax.axis('off')
plt.show()