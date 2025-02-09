# Weber-Inspect

Weber-Inspect is a Python-based image processing pipeline for precision edge detection and geometric consolidation. The project detects outer and inner boundaries in an image—such as those from a whiteboard or technical drawing—and exports the consolidated boundaries as a DXF file suitable for CAD measurement and analysis.

## Features

- **Precision Edge Detection:**  
  Uses OpenCV to perform robust image preprocessing, including whiteboard calibration, component isolation, enhanced processing, and dual thresholding.

- **Geometric Consolidation:**  
  Utilizes Shapely to merge nearby features into continuous boundaries while preserving inner details (e.g., holes and grooves).

- **DXF Export:**  
  Exports detected boundaries to DXF format with ezdxf, enabling easy integration with CAD software for measurements and further analysis.

- **Debug Mode:**  
  Optionally displays intermediate results (e.g., whiteboard calibration, thresholding, contour extraction) to help with parameter tuning and debugging.

- **Automated Logging and Output Management:**  
  Logs processing steps to the terminal and saves output images and DXF files into organized subdirectories.

## Requirements

- Python 3.6+
- [OpenCV](https://opencv.org/) (`opencv-python`)
- [Shapely](https://pypi.org/project/Shapely/)
- [ezdxf](https://pypi.org/project/ezdxf/)

Install the required packages using pip:

```bash
pip install opencv-python shapely ezdxf
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Weber-Inspect.git
   cd Weber-Inspect
   ```

2. **(Optional) Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *(If a `requirements.txt` file is not provided, install the packages manually as shown in the Requirements section.)*

## Usage

Run the script by providing the source image. The processing pipeline will generate:
- A consolidated edge detection image (PNG) saved in `outputs/png`
- A DXF file containing the detected boundaries saved in `outputs/dxf`

You can enable intermediate image display (debug mode) by setting the `image_debug` flag to `True`.

For example, if your script file is named `precision_edge_detection.py`, run:

```bash
python precision_edge_detection.py
```

In the code, update the source image filename (e.g., `"oip.jpg"`) if needed.

## Project Structure

```
Weber-Inspect/
├── outputs/
│   ├── dxf/         # DXF files for CAD measurement
│   └── png/         # Processed output images (consolidated boundaries)
├── src/             # Source code (if applicable)
├── .gitignore       # Git ignore file to exclude src and outputs content
└── README.md        # This file
```

## .gitignore

The repository is set to ignore the contents of the `src` and `outputs` folders. Refer to the `.gitignore` file for details.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [Shapely](https://pypi.org/project/Shapely/)
- [ezdxf](https://pypi.org/project/ezdxf/)
```

---

Feel free to modify the text (such as the GitHub repository URL, license details, or additional sections) to suit your project needs.