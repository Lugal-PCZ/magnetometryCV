# Magnetometry Dipole Detection

## Overview

This program is a computer vision-based application for magnetic dipole detection using magnetometry data. It is written in Python and uses the NumPy and SciPy libraries, along with other image-processing libraries, to detect candidate dipoles using a rotation-invariant matched filtering approach. A secondary filtering algorithm based on the shape and contrast of the dipole poles is then used to reduce false detections.

## Features

- Loads magnetometry data from a GeoTIFF file.
- Computes the analytic signal using spatial gradients.
- Generates a Gaussian-based dipole template.
- Applies a rotation-invariant matched filter to detect dipole-like features at multiple orientations.
- Detects local response peaks using percentile thresholding and morphological dilation.
- Validates candidate detections using halo and dipole-balance checks.
- Generates a visualization of confirmed detections.
- Supports interactive inspection of detection coordinates and data values.

## Code Overview

The program follows the detection pipeline outlined below.

### 1. Load Data

Loads magnetometry data from a GeoTIFF file using Rasterio.

### 2. Compute the Analytic Signal

Calculates the analytic signal using the spatial gradients of the data.

### 3. Create the Dipole Template

Generates a dipole template using Gaussian lobes.

### 4. Rotation-Invariant Matched Filtering

The dipole template is rotated through multiple orientations.

For each orientation, the program applies a matched filter. The maximum response across all orientations is stored, allowing the detection process to identify dipolar artifacts with different orientations.

### 5. Candidate Detection

Candidate detections are identified by:

- Applying a percentile threshold to the matched-filter response.
- Using morphological dilation to identify local response peaks.
- Recording the pixel coordinates of the detected peaks.

### 6. Validation

Each candidate is evaluated using two validation checks.

**Halo Check**

The surrounding region must have sufficiently low variation compared to the inner region of the candidate.

**Dipole Balance**

Evaluates the contrast ratio between the positive and negative poles of the dipole.

If the ratio falls within a specified range and the candidate passes the halo check, it is accepted as a confirmed detection.

### 7. Visualization

Generates a Matplotlib visualization of the magnetometry data, with confirmed dipoles circled in green.

The visualization is saved as:

`results/detected_dipoles.png`

## Requirements

The following Python libraries are required:

- NumPy
- OpenCV (`cv2`)
- Matplotlib
- mplcursors
- Rasterio
- SciPy

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Input Data

The input file should be a magnetogram in GeoTIFF format.

## Running the Program

Run the Python script:

```bash
python main.py
```

## Parameters

The following parameters control the behavior of the detection algorithm.

| Parameter | Description |
|---|---|
| `inputfile` | Path of the source GeoTIFF magnetogram (defaults to Sample.tiff). |
| `resolution` | Resolution of the source GeoTIFF magnetogram (defaults to 10cm/pixel). |
| `anomaly_size` | Diameter, in meters, of the anomaly detection window (defaults to 2.3m). |
| `lobe_distance` | Separation, in meters, between the two Gaussian lobes (defaults to 0.3m). |
| `sigma` | Spread or blur of the Gaussian lobes (defaults to 0.7). |
| `angle_step` | Angular increment, in degrees, between successive template rotations (defaults to 15°). |
| `threshold_percentile` | Percentile threshold used to determine candidate detections (defaults to 99%). |
| `halo_radius` | Inner radius, in meters, used for halo validation (defaults to 0.7m). |

The total number of template orientations evaluated depends on the angular step. In the current implementation, the template is rotated from 0 to 180 degrees, and the number of orientations evaluated is approximately 180/angle_step

## Output

### `results/detected_dipoles.png`

A visualization of the magnetometry data with dipole detections marked by green circles.

### Console Output

The program reports:

- The number of candidate dipoles.
- The number of detected dipoles.
- The elapsed detection time.

## Limitations

- Detection results depend on the selected template parameters and threshold values.
- The validation checks are heuristic and do not guarantee that every confirmed detection represents a real archaeological or geological feature.
- Candidates near the edges of the input data are excluded if their validation window extends outside the image.
- The current implementation processes a single GeoTIFF band.
- The visualization uses pixel coordinates (expressed in meters from an origin in the southwest corner) rather than geographic coordinates.
