# 2112721017    Ilkay onay

from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import os
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def apply_thresholding(image, threshold_value, block_size, c, otsu):
    if otsu:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        if block_size and c:
            image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return image

def apply_blur(image, kernel_size, blur_type):
    if blur_type == 'Box':
        return cv2.boxFilter(image, -1, (kernel_size, kernel_size))
    elif blur_type == 'Gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif blur_type == 'Normal':
        return cv2.blur(image, (kernel_size, kernel_size))
    elif blur_type == 'Bilateral':
        return cv2.bilateralFilter(image, kernel_size, 75, 75)
    elif blur_type == 'Median':
        return cv2.medianBlur(image, kernel_size)
    return image

def apply_gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_sobel_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_normalized = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return sobel_normalized

def apply_canny_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, 100, 200)

def apply_deriche_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    alpha, kernel_size = 0.5, 3
    kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
    deriche_kernel_x, deriche_kernel_y = alpha * kx, alpha * ky
    deriche_x = cv2.filter2D(gray_image, -1, deriche_kernel_x)
    deriche_y = cv2.filter2D(gray_image, -1, deriche_kernel_y)
    edges = np.sqrt(np.square(deriche_x) + np.square(deriche_y))
    edges = (edges / np.max(edges)) * 255
    return edges.astype(np.uint8)

def apply_harris_corners(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float32(gray_image)
    harris_corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    image[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]
    return image

def apply_laplacian_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    return laplacian.astype(np.uint8)

def apply_watershed(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.medianBlur(gray_image, 3)
    _, img_thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    img_open = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=7)
    sure_bg = cv2.dilate(img_open, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(img_open, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg, labels=5)
    markers = markers + 1
    markers[unknown == 255] = 0
    image_8uc3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    markers_32s = np.int32(markers)
    cv2.watershed(image_8uc3, markers_32s)
    contours, hierarchy = cv2.findContours(markers_32s, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (255, 0, 0), 5)
    return image

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Error: Unable to load the Haar Cascade classifier.")
        return image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=6)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 10)
    return image

def detect_contours(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'success': True, 'filename': filename})
    return jsonify({'success': False})

@app.route('/download_image/<filename>')
def download_image(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

@app.route('/save_parameters', methods=['POST'])
def save_parameters():
    parameters = request.json
    filename = 'parameters.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(parameters, file)
    return jsonify({'success': True})

@app.route('/load_parameters', methods=['GET'])
def load_parameters():
    filename = 'parameters.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            parameters = pickle.load(file)
        return jsonify(parameters)
    return jsonify({'success': False})

@app.route('/generate_image', methods=['POST'])
def generate_image():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if data['thr_otsu']:
        frame = apply_thresholding(frame, 0, 0, 0, True)
    elif data['thr_value'] and data['thr_block_size'] and data['thr_c']:
        frame = apply_thresholding(frame, int(data['thr_value']), int(data['thr_block_size']), int(data['thr_c']), False)

    if data['bl_kernelsize'] and data['bl_filter']:
        frame = apply_blur(frame, int(data['bl_kernelsize']), 'Box')
    elif data['bl_kernelsize'] and data['bl_gaussian']:
        frame = apply_blur(frame, int(data['bl_kernelsize']), 'Gaussian')
    elif data['bl_kernelsize'] and data['bl_normal']:
        frame = apply_blur(frame, int(data['bl_kernelsize']), 'Normal')
    elif data['bl_kernelsize'] and data['bl_bilateral']:
        frame = apply_blur(frame, int(data['bl_kernelsize']), 'Bilateral')
    elif data['bl_kernelsize'] and data['bl_median']:
        frame = apply_blur(frame, int(data['bl_kernelsize']), 'Median')

    if data['gmm_value']:
        frame = apply_gamma_correction(frame, float(data['gmm_value']))

    if data['bda_sobel']:
        frame = apply_sobel_filter(frame)
    elif data['bda_laplacian']:
        frame = apply_laplacian_filter(frame)
    elif data['bda_canny']:
        frame = apply_canny_edge_detection(frame)
    elif data['bda_deriche']:
        frame = apply_deriche_filter(frame)
    elif data['bda_harris']:
        frame = apply_harris_corners(frame)

    if data['dt_viola']:
        frame = detect_faces(frame)
    elif data['dt_kontur']:
        frame = detect_contours(frame)
    elif data['dt_watershed']:
        frame = apply_watershed(frame)

    filename = secure_filename('processed_image.png')
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(file_path, frame)

    return jsonify({'success': True, 'filename': filename})

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    filename = secure_filename('saved_image.png')
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    cv2.imwrite(file_path, frame)

    return jsonify({'success': True})

@app.route('/exit', methods=['POST'])
def exit_app():
    return jsonify({'success': True})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True)
