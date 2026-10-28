# Magnetometry Dipole Detection

## Overview

This program is a computer vision-based application for magnetic dipole detection using magnetometry data. It is Python-based and uses the NumPy and SciPy libraries, along with other image-processing libraries, to detect candidate dipoles using a rotation-invariant matched filtering approach. A secondary filtering algorithm based on the shape and contrast of the dipole poles is then used to reduce false detections.

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

`magnetometry_results.png`

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

The program currently uses the following input file:

`Lagash_Area_G_Magnetogram.tif`

The input file should be a GeoTIFF containing magnetometry data.

Place the file in the same directory as the Python script, or update the `file_path` variable to point to its location.

## Running the Program

Run the Python script:

```bash
python dipole_detection.py
```

## Parameters

The following parameters control the behavior of the detection algorithm.

| Parameter | Description |
|---|---|
| `ARTIFACT_SIZE` | Diameter of the artifact detection window, in pixels. |
| `lobe_distance` | Separation between the two Gaussian lobes, in pixels. |
| `sigma` | Controls the spread or blur of the Gaussian lobes. |
| `angle_step` | Angular increment between successive template rotations. |
| `threshold_percentile` | Percentile threshold used to determine candidate detections. |
| `inner_radius` | Inner radius used for halo validation. |
| `outer_radius` | Outer radius used for halo validation. |

The total number of template orientations evaluated depends on the angular step. In the current implementation, the template is rotated from 0 to 180 degrees.

The number of orientations evaluated is approximately:

\[
\frac{180}{\text{angle\_step}}
\]

## Output

### `magnetometry_results.png`

A visualization of the magnetometry data with confirmed dipole detections marked by green circles.

### Console Output

The program reports:

- The number of detected candidates.
- The number of confirmed dipoles.
- The elapsed detection time.

## Limitations

- Detection results depend on the selected template parameters and threshold values.
- The validation checks are heuristic and do not guarantee that every confirmed detection represents a real archaeological or geological feature.
- Candidates near the edges of the input data are excluded if their validation window extends outside the image.
- The current implementation processes a single GeoTIFF band.
- The visualization uses pixel coordinates rather than geographic coordinates.
