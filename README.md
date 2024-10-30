# Emmetra Assignments

Welcome to the Emmetra assignments repository! This repository contains implementations for three assignments focused on Image Signal Processing (ISP), Denoise and Sharpness Techniques, and HDR Imaging. Each assignment is organized in its own folder with source code, test data, and documentation.

## Repository Structure
- **Assignment 1**: Basic ISP Implementation
- **Assignment 2**: Denoise and Sharpness Techniques
- **Assignment 3**: HDR Imaging

Each assignment folder contains:
- **Source Code**: All code implementations for the tasks.
- **Test Data**: Example input files and any other required data.
- **Documentation**: Report and README files explaining the approach, design, and usage instructions.

## Timeline
- **Due Date**: November 5th, 2024

Upon completion, submit the assignment via the provided form for eligibility for certificates. Top 3 submissions will receive special rewards!

---

## Assignment - 1: Basic ISP Implementation

### Objective
Implement a basic Image Signal Processing (ISP) pipeline for sensor raw images. This includes:
- **Demosaicing**: Edge-based interpolation (5x5).
- **White Balance**: Simple gray world algorithm.
- **Denoising**: Gaussian filter (5x5).
- **Gamma Correction**: sRGB gamma conversion (12-bit to 8-bit).
- **Sharpening**: Unsharp mask filter.

#### Bonus Task
Create a UI tool to control parameters and visualize results for each processing block.

### Input and Output
- **Input**: 12-bit Bayer raw image in GRBG format, 1920x1280 resolution.
- **Output**: RGB channel with 24 bits per pixel (8 bits for each channel).

### Tools
- **Tools to view RAW images**: PixelViewer, IrfanView with RAW plugin.
- **Configuration for input**: Bayer - 12 bits, GRBG, 1920x1280.

### Tasks
Implement the following routine combinations and document observations:
- Demosaic + Gamma
- Demosaic + White Balance + Gamma
- Demosaic + White Balance + Denoise + Gamma
- Demosaic + White Balance + Denoise + Gamma + Sharpen

---

## Assignment - 2: Denoise and Sharpness Techniques

### Objective
Explore and compare different denoise and sharpness techniques to assess image quality.

### Tasks
#### Denoising
1. **Implement** median and bilateral filters. Compare these with the Gaussian filter from Assignment 1.
2. **AI-based Denoising**: Implement or use a pre-trained AI model (e.g., U-net, FFDNet) for image denoising.
3. **Signal-to-Noise Ratio**: Calculate spatial SNR for three gray tones as per [Imatest](https://www.imatest.com/imaging/noise/).

#### Edge Enhancement
1. Implement a Laplacian filter for edge enhancement.
2. Compute edge strength based on gradient-based methods for each technique.

### Input and Output
- **Input**: 12-bit Bayer raw image in GRBG format, 1920x1280 resolution.
- **Output**: RGB channel with 24 bits per pixel.

### AI Model
- **Framework**: TensorFlow for a denoise CNN model (use pre-trained models if preferred).

---

## Assignment - 3: HDR Imaging

### Objective
Implement an HDR imaging algorithm to merge and tone-map images of different exposures.

### Tasks
1. **Capture**: Take three differently exposed images of a high-contrast scene (outdoors or indoors with bright, shadow, and low-light areas).
2. **HDR Merge**: Implement merging of the images and tone mapping to an 8-bit format for display.

### Input and Output
- **Input**: 3 differently exposed images.
- **Output**: Merged and tone-mapped HDR image.

---

## Submission Guidelines
1. Organize your code, test data, and documentation in separate folders for each assignment.
2. Submit the repository link via the provided form to complete your submission.

All participants who successfully submit are eligible for certificates, and top 3 winners will receive rewards. Good luck!
