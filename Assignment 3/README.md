# HDR Image Processing Tool

A Python application for creating High Dynamic Range (HDR) images from multiple exposure brackets. This tool provides automated image alignment, HDR merging, tone mapping, and enhancement capabilities.

## Features

- Automatic image alignment
- HDR image creation from multiple exposures
- Tone mapping using Drago's method
- Image enhancement:
  - Contrast adjustment
  - Saturation enhancement
  - Gamma correction
- Automatic image resizing for consistency
- Support for various image formats (JPG, PNG, etc.)

## Prerequisites

- Python 3.6 or higher
- pip package manager

## Installation

1. Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv hdr-env

# Activate on Windows
hdr-env\Scripts\activate

# Activate on Linux/Mac
source hdr-env/bin/activate
```

2. Install required packages:

```bash
pip install numpy
pip install opencv-python
```

## Usage

1. Prepare your images:
   - Take 3 bracketed exposures of the same scene
   - Images should include:
     - Underexposed image (to capture highlight details)
     - Normal exposed image
     - Overexposed image (to capture shadow details)

2. Find the exposure times of your images:
   - Check the EXIF data of your images using:
     - Your camera's image info display
     - Image editing software
     - Online EXIF viewers
   - Note down the exposure time (shutter speed) for each image

3. Modify the script:
   - Open `emmetra3.py`
   - Locate the `load_exposure_images` method
   - Update the exposure_times list with your actual values:
   ```python
   # Example: Replace these values with your actual exposure times in seconds
   exposure_times = [
       1/100,  # Your underexposed image's exposure time
       1/30,   # Your normal exposure image's exposure time
       1/8     # Your overexposed image's exposure time
   ]
   ```
   - Update the image paths in the `main()` function:
   ```python
   image_paths = [
       "path/to/underexposed.jpg",
       "path/to/normal.jpg",
       "path/to/overexposed.jpg"
   ]
   output_path = "path/to/output_hdr.jpg"
   ```

4. Run the program:
```bash
python emmetra3.py
```

## Processing Steps

1. Image Loading
   - Images are loaded and automatically resized if larger than 1500 pixels
   - Consistent dimensions are enforced across all images

2. Image Alignment
   - Automatic alignment using ECC algorithm
   - Middle exposure used as reference
   - Handles minor camera movement between shots

3. HDR Creation
   - Merges aligned images using Debevec's method
   - Combines multiple exposures into a single HDR image

4. Tone Mapping
   - Applies Drago tone mapping operator
   - Converts HDR to displayable 8-bit image
   - Preserves details in highlights and shadows

5. Enhancement
   - Applies final adjustments to:
     - Contrast (default: 1.1)
     - Saturation (default: 1.2)
     - Gamma (default: 1.0)

## Customization

You can adjust processing parameters by modifying the class variables:

```python
processor = HDRProcessor()
processor.gamma = 1.0      # Adjust gamma (0.5 to 2.0)
processor.saturation = 1.2 # Adjust saturation (0.0 to 2.0)
processor.contrast = 1.1   # Adjust contrast (0.5 to 2.0)
```

## Output

- Outputs a single JPEG image
- Recommended for:
  - High contrast scenes
  - Interior shots with bright windows
  - Sunset/sunrise photography
  - Architecture photography

## Best Practices

1. Image Capture:
   - Use a tripod (essential for sharp HDR)
   - Avoid moving subjects
   - Use bracketing mode on your camera if available
   - Keep ISO constant across all shots
   - Use aperture priority mode (Av) to maintain consistent depth of field
   - Ensure no clipping in highlights or shadows across the series

2. Camera Settings:
   - Turn off Auto White Balance (use manual WB)
   - Disable HDR mode if your camera has it
   - Turn off any in-camera noise reduction
   - Use RAW format if possible

3. Processing:
   - Use RAW images when possible
   - Ensure images are properly aligned
   - Start with default enhancement values
   - Adjust parameters based on scene
   - Monitor for halos or artifacts

## Troubleshooting

Common issues and solutions:

1. Alignment errors:
   - Ensure minimal movement between shots
   - Check if images have sufficient detail for alignment
   - Try using RAW format images
   - Reduce time between shots
   - Use timer or remote trigger

2. Memory errors:
   - Reduce input image size
   - Close other applications
   - Ensure sufficient RAM (4GB minimum recommended)
   - Process images in smaller batches

3. Output quality issues:
   - Adjust gamma, saturation, and contrast values
   - Ensure proper exposure bracketing
   - Check input image quality
   - Verify exposure times are correctly entered
   - Look for motion blur or camera shake

4. Ghost artifacts:
   - Reshoot if possible without moving elements
   - Try shooting at different times
   - Use faster shutter speeds if possible

## Limitations

- Requires static scenes
- Best results with 3 exposure brackets
- Maximum input image dimension: 1500 pixels
- May struggle with extreme movement between frames
- Processing time depends on image size and computer specifications

## Tips for Specific Scenarios

1. Interior Shots:
   - Include at least one exposure for window details
   - Ensure deepest shadows are captured
   - Consider using more than 3 brackets for extreme contrast

2. Landscape Photography:
   - Use early morning or late afternoon light
   - Include exposure for bright sky
   - Consider using filters to reduce contrast

3. Architecture:
   - Shoot during blue/golden hour
   - Account for artificial lighting
   - Consider vertical orientation for tall buildings

## Acknowledgments

- Uses OpenCV's HDR implementation
- Tone mapping based on Drago's operator
- Inspired by professional HDR photography techniques

## Version History

- 1.0: Initial release
- 1.1: Added automatic image alignment
- 1.2: Enhanced tone mapping options
