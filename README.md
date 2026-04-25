# moonshinekedit-web

## Overview

MoonShineKedit-Web is a web-based image processing application that leverages the power of Python (Flask) and OpenCV.js to provide real-time image manipulation capabilities directly in your browser. This project allows users to apply a variety of image processing filters and algorithms to live camera feeds or uploaded images. The application is designed to be intuitive, offering a range of tools for tasks such as thresholding, blurring, edge detection, and object detection.

The core functionality is driven by a Flask backend that serves the web interface and handles image processing requests. The frontend, built with HTML, CSS, and JavaScript, utilizes OpenCV.js to perform client-side image manipulations, ensuring a responsive and interactive user experience. The application also supports saving and loading processing parameters, allowing users to easily recall and reuse their preferred filter configurations.

## Features

*   **Real-time Camera Feed Processing:** Apply image filters and algorithms to a live webcam stream.
*   **Image Upload and Processing:** Load static images and apply various processing techniques.
*   **Extensive Image Filtering Options:**
    *   **Thresholding:** Binary, Otsu, and Adaptive Thresholding.
    *   **Blurring:** Box, Gaussian, Normal, Bilateral, and Median filters.
    *   **Gamma Correction:** Adjust image brightness and contrast.
*   **Advanced Image Analysis:**
    *   **Edge Detection:** Sobel, Laplacian, Canny, Deriche, and Harris corner detection.
    *   **Object Detection:** Viola-Jones Haar Cascade for face detection, contour detection, and Watershed segmentation.
*   **Parameter Management:** Save and load processing parameters for reproducible results.
*   **User-Friendly Interface:** An intuitive web interface built with Bootstrap for a clean and responsive design.

## Project Structure

```
├── LICENSE.txt
├── app.py
├── models/
│   └── haarcascade_frontalface_default.xml
├── static/
│   └── script.js
└── templates/
    └── index.html
└── uploads/
```

## Getting Started

To run this application, you will need Python and Flask installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd moonshinekedit-web
    ```

2.  **Install dependencies:**
    ```bash
    pip install Flask Flask-SocketIO opencv-python numpy
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

The application will be accessible at `http://127.0.0.1:5000/`.

## License

This project is licensed under the GNU General Public License v3.0. See the `LICENSE.txt` file for more details.