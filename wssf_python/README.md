# WSSF Python Implementation

This is a Python implementation of the WSSF (Weighted Scale Space Feature) algorithm for image feature detection, description, and matching.

## Project Structure

```
wssf_python/
├── core/                  # Core algorithm components
│   ├── descriptors.py     # GLOH descriptor implementation
│   ├── feature_detection.py # Feature detection algorithms
│   ├── feature_extraction.py # Feature extraction pipeline
│   ├── gradient_feature.py # Gradient feature computation
│   ├── image_space.py     # Image space creation
│   ├── matching.py        # Feature matching and FSC
│   └── nms.py             # Non-maximum suppression
├── utils/                 # Utility functions
│   └── image_utils.py     # Image processing utilities
├── visualization/         # Visualization tools
│   └── visualization.py   # Functions for visualizing results
└── wssf_demo.py           # Main demo script
```

## Requirements

- Python 3.6+
- NumPy
- OpenCV
- Matplotlib
- scikit-image

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the demo script:

```
python wssf_demo.py
```

The script will:
1. Load and preprocess input images
2. Create image spaces
3. Extract gradient features
4. Detect and extract WSSF features
5. Compute GLOH descriptors
6. Match features between images
7. Apply FSC for robust matching
8. Visualize matches and create image fusion

## Implementation Notes

This Python implementation follows the original MATLAB implementation with some adaptations for Python's ecosystem. The core algorithms remain the same, but some implementation details may differ due to differences between MATLAB and Python libraries.

## References

- Original MATLAB implementation by [Original Authors]
- [Paper references if available]