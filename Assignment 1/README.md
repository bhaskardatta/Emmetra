# ISP (Image Signal Processing) Pipeline Tool

This application provides a graphical interface for processing RAW image files through various image processing steps including demosaicing, white balance, denoising, gamma correction, and sharpening.

## Features

- RAW image loading (12-bit)
- Interactive GUI for parameter tuning
- Real-time image preview
- Multiple processing steps:
  - Demosaicing (5x5 edge-based)
  - White Balance (Gray World method)
  - Denoising
  - Gamma Correction
  - Image Sharpening
- Automatic processing with predefined combinations
- Save processed images in various formats

## Prerequisites

- Python 3.6 or higher

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv isp-env
source isp-env/bin/activate  # On Windows use: isp-env\Scripts\activate
```

2. Install the required packages:
```bash
pip install numpy opencv-python matplotlib pillow tkinter
```

## Usage

1. Run the application:
```bash
python emmetra1.py
```

2. Using the GUI:
   - Click "Load RAW Image" to select your RAW image file
   - Use checkboxes to enable/disable processing steps
   - Adjust parameters using the sliders:
     - Denoise Sigma (0.1 - 5.0)
     - Gamma (0.1 - 5.0)
     - Sharpen Amount (0.1 - 5.0)
   - Click "Process Image" to apply changes
   - Click "Save Processed Image" to export the result

3. Automatic Processing:
   - Click "Run Automatic Processing"
   - Select a directory to save the processed images
   - The tool will generate four different combinations:
     - Demosaic + Gamma
     - Demosaic + White Balance + Gamma
     - Demosaic + White Balance + Denoise + Gamma
     - Demosaic + White Balance + Denoise + Gamma + Sharpen

## Input Requirements

- RAW image files (12-bit)
- Expected resolution: 1920x1280 pixels
- Bayer pattern: GRBG

## Output

- Processed images can be saved in PNG or JPEG format
- When using automatic processing, all versions are saved with descriptive names

## Notes

- The application assumes a GRBG Bayer pattern for demosaicing
- The preview window automatically scales large images to fit the display
- All processing is done in memory, so ensure sufficient RAM for large images
