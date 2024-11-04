# Enhanced RAW Image Processing Tool

An advanced Python application for processing RAW image files using deep learning and traditional image processing techniques. This tool provides comprehensive image enhancement capabilities including denoising, contrast enhancement, and color correction.

## Features

- Deep Learning based denoising using DnCNN
- Multiple enhancement techniques:
  - Adaptive bilateral filtering
  - Multi-scale denoising
  - Local contrast enhancement
  - Enhanced Laplacian sharpening
  - Color enhancement
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Automated metrics calculation (PSNR, SNR)
- Interactive file selection
- Comprehensive result visualization
- Automatic results saving with comparison plots

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (optional, for faster processing)
- pip package manager

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv img-proc-env

# Activate on Windows
img-proc-env\Scripts\activate

# Activate on Linux/Mac
source img-proc-env/bin/activate
```

2. Install required packages:

```bash
pip install numpy
pip install opencv-python
pip install matplotlib
pip install scikit-image
pip install rawpy
pip install torch torchvision
pip install scipy
```

## Usage

1. Run the program:
```bash
python emmetra2.py
```

2. Through the GUI:
   - Select your RAW image file when prompted
   - The program will automatically:
     - Process the image using multiple enhancement techniques
     - Generate enhanced versions
     - Create comparison visualizations
     - Calculate quality metrics
     - Save all results

3. Output:
   - All results are saved in a new directory named `[input_filename]_enhanced_results`
   - The output includes:
     - Individual enhanced images (PNG format)
     - Comparison plot showing all versions
     - Metrics text file with PSNR and SNR values

## Supported File Formats

- RAW (.raw, .RAW)
- Digital Negative (.dng, .DNG)

## Default Parameters

- Image dimensions: 1920x1280
- Bit depth: 14-bit
- Color pattern: Bayer

## Output Structure

```
input_file_enhanced_results/
├── Original.png
├── Bilateral.png
├── Advanced_Denoised.png
├── DL_Denoised.png
├── Enhanced_Laplacian.png
├── Local_Contrast.png
├── Combined_Enhancement.png
├── comparison.png
└── metrics.txt
```

## Performance Notes

- GPU acceleration is automatically used if available
- Processing time varies based on:
  - Image size
  - Available computing resources
  - Selected enhancement techniques

## Troubleshooting

Common issues and solutions:

1. CUDA/GPU errors:
   - Ensure CUDA toolkit is installed
   - Update GPU drivers
   - The program will automatically fall back to CPU if GPU is unavailable

2. Memory errors:
   - Close other applications
   - Process smaller images
   - Ensure sufficient RAM (8GB minimum recommended)

3. RAW file reading errors:
   - Verify file format compatibility
   - Check file permissions
   - Ensure file isn't corrupted

## Acknowledgments

- DnCNN architecture based on the paper "Beyond a Gaussian Denoiser"
- Uses various open-source computer vision libraries
