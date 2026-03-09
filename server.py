from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tempfile
import urllib.request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max upload

# Model downloads automatically on first run
MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'webm', 'm4v'}

LEFT_IRIS = [474, 475, 476, 477]

_detector = None


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading face landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model downloaded!")


def get_detector():
    global _detector
    if _detector is None:
        ensure_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_iris_center(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def analyze_video(path):
    logger.info(f"Analyzing video at: {path}")
    logger.info(f"File size: {os.path.getsize(path)} bytes")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "Could not open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None, "Could not determine video frame rate"
    logger.info(f"FPS: {fps}")

    detector = get_detector()

    x_positions = []
    y_positions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            cx, cy = get_iris_center(lm, LEFT_IRIS, w, h)
            x_positions.append(cx)
            y_positions.append(cy)
        else:
            if x_positions:
                x_positions.append(x_positions[-1])
                y_positions.append(y_positions[-1])

    cap.release()
    logger.info(f"Tracked {len(x_positions)} frames")

    if len(x_positions) < 10:
        return None, "Not enough eye tracking data"

    x = np.array(x_positions)
    y = np.array(y_positions)
    x -= np.mean(x)
    y -= np.mean(y)

    peaks, _   = find_peaks(x, height=np.std(x) * 0.5, distance=fps * 0.1)
    troughs, _ = find_peaks(-x, height=np.std(x) * 0.5, distance=fps * 0.1)
    all_extremes = np.sort(np.concatenate([peaks, troughs]))

    if len(all_extremes) < 2:
        peaks, _   = find_peaks(y, height=np.std(y) * 0.5, distance=fps * 0.1)
        troughs, _ = find_peaks(-y, height=np.std(y) * 0.5, distance=fps * 0.1)
        all_extremes = np.sort(np.concatenate([peaks, troughs]))

    if len(all_extremes) < 2:
        return None, "No clear oscillation detected"

    intervals = np.diff(all_extremes) / fps
    hz = 1.0 / (np.mean(intervals) * 2)
    return round(hz, 2), None


@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Nystagmus Tracker server is running!'})


@app.route('/analyze', methods=['POST'])
def analyze():
    logger.info(f"Files received: {list(request.files.keys())}")

    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided. Send a file with the key "video".'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(video_file.filename):
        return jsonify({'error': f'Unsupported file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

    tmp_path = os.path.join(tempfile.gettempdir(), 'eye_scan_temp.mp4')
    video_file.save(tmp_path)
    logger.info(f"Video saved to: {tmp_path}")

    hz, error = analyze_video(tmp_path)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    if error:
        logger.warning(f"Analysis error: {error}")
        return jsonify({'error': error}), 422

    logger.info(f"Result: {hz} Hz")
    return jsonify({'hz': hz})


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 100 MB.'}), 413


# Pre-load model at startup
with app.app_context():
    logger.info("Pre-loading face landmark model...")
    try:
        get_detector()
        logger.info("Model ready.")
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
