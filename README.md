# face-recognition

# Face Recognition System

A robust face recognition system built with Python that can detect, recognize, and verify human faces in real-time using computer vision and deep learning techniques.

## Features

- **Real-time Face Detection**: Detect faces in live video streams or static images
- **Face Recognition**: Identify known individuals from a pre-trained database
- **Face Verification**: Verify if two faces belong to the same person
- **Multiple Face Support**: Handle multiple faces in a single frame
- **High Accuracy**: Uses state-of-the-art deep learning models
- **Easy Integration**: Simple API for integration into other applications

## Technologies Used

- **Python 3.8+**
- **OpenCV**: Computer vision library for image processing
- **dlib**: Machine learning library for face detection and recognition
- **face_recognition**: Python library built on dlib
- **NumPy**: Numerical computing
- **PIL/Pillow**: Image processing
- **Flask/FastAPI**: Web framework (if applicable)

## Installation

### Prerequisites

Make sure you have Python 3.8 or higher installed on your system.

### Clone the Repository

```bash
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Alternative Installation with conda

```bash
conda create -n face_recognition python=3.8
conda activate face_recognition
pip install -r requirements.txt
```

## Dependencies

Create a `requirements.txt` file with the following:

```
opencv-python==4.8.1.78
face-recognition==1.3.0
dlib==19.24.2
numpy==1.24.3
Pillow==10.0.0
click==8.1.7
```

## Usage

### Basic Face Recognition

```python
import face_recognition
import cv2

# Load and encode known faces
known_image = face_recognition.load_image_file("path/to/known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Load unknown image
unknown_image = face_recognition.load_image_file("path/to/unknown_person.jpg")
unknown_encodings = face_recognition.face_encodings(unknown_image)

# Compare faces
results = face_recognition.compare_faces([known_encoding], unknown_encodings[0])
print(f"Match found: {results[0]}")
```

### Real-time Face Recognition

```bash
python real_time_recognition.py
```

### Command Line Interface

```bash
# Recognize faces in an image
python face_recognition_cli.py --input path/to/image.jpg --output path/to/output.jpg

# Train the model with new faces
python train_model.py --dataset path/to/training/images

# Start webcam recognition
python webcam_recognition.py
```

## Project Structure

```
face-recognition/
│
├── src/
│   ├── face_detector.py      # Face detection utilities
│   ├── face_recognizer.py    # Face recognition core logic
│   ├── utils.py              # Helper functions
│   └── config.py             # Configuration settings
│
├── models/
│   ├── encodings.pkl         # Pre-computed face encodings
│   └── classifier.pkl        # Trained classifier (if using)
│
├── data/
│   ├── known_faces/          # Training images
│   ├── unknown_faces/        # Test images
│   └── output/               # Processed results
│
├── tests/
│   ├── test_detection.py     # Unit tests for detection
│   └── test_recognition.py   # Unit tests for recognition
│
├── examples/
│   ├── basic_example.py      # Simple usage example
│   └── advanced_example.py   # Advanced features demo
│
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
├── config.yaml               # Configuration file
└── README.md                 # This file
```

## Configuration

Edit `config.yaml` to customize the system:

```yaml
face_recognition:
  tolerance: 0.6              # Recognition tolerance (lower = stricter)
  model: "hog"               # Detection model: "hog" or "cnn"
  max_faces: 10              # Maximum faces to process per frame

camera:
  source: 0                  # Camera source (0 for default webcam)
  width: 640                 # Frame width
  height: 480                # Frame height

output:
  save_results: true         # Save recognition results
  output_dir: "data/output"  # Output directory
```

## API Documentation

### Core Functions

#### `detect_faces(image_path)`
Detects faces in the given image.

**Parameters:**
- `image_path` (str): Path to the input image

**Returns:**
- List of face locations as tuples (top, right, bottom, left)

#### `recognize_faces(image_path, known_encodings, known_names)`
Recognizes faces in the image against known faces.

**Parameters:**
- `image_path` (str): Path to the input image
- `known_encodings` (list): List of known face encodings
- `known_names` (list): List of corresponding names

**Returns:**
- Dictionary with recognized names and confidence scores

## Performance

- **Detection Speed**: ~30 FPS on CPU, ~100 FPS on GPU
- **Recognition Accuracy**: 99.38% on LFW dataset
- **Memory Usage**: ~200MB for basic operations
- **Supported Formats**: JPG, PNG, BMP, TIFF

## Troubleshooting

### Common Issues

1. **dlib installation fails**
   ```bash
   # On Red Hat / Fedora
    sudo dnf groupinstall "Development Tools"
    sudo dnf install cmake
    sudo dnf install openblas-devel lapack-devel

   # On Ubuntu/Debian
   sudo apt-get install build-essential cmake
   sudo apt-get install libopenblas-dev liblapack-dev

   # On macOS
   brew install cmake
   ```

2. **OpenCV import error**
   ```bash
   pip uninstall opencv-python
   pip install opencv-python-headless
   ```

3. **Low accuracy**
   - Ensure good lighting conditions
   - Use high-resolution images for training
   - Adjust tolerance settings in config

### Performance Optimization

- Use GPU acceleration with dlib compiled with CUDA support
- Resize images to smaller dimensions for faster processing
- Use the "hog" model for CPU, "cnn" for GPU

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=src tests/

# Run specific test
python -m pytest tests/test_recognition.py::test_face_matching
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library by Adam Geitgey
- [dlib](http://dlib.net/) library by Davis King
- OpenCV community for computer vision tools
- Contributors and the open-source community

## Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

## Changelog

### v1.0.0 (2024-01-15)
- Initial release
- Basic face detection and recognition
- Real-time webcam support
- Command-line interface

### v0.2.0 (2023-12-20)
- Added batch processing
- Improved accuracy with better preprocessing
- Added configuration file support

### v0.1.0 (2023-11-30)
- Beta release
- Core face recognition functionality

---

**Note**: This project is for educational and research purposes. Ensure compliance with privacy laws and regulations when using face recognition technology in production environments.
