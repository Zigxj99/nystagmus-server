from flask import Flask, request, jsonify
import cv2
import numpy as np
from scipy.signal import find_peaks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tempfile
import urllib.request

app = Flask(__name__)

MODEL_PATH = "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

LEFT_IRIS = [474, 475, 476, 477]

# Nystagmus is small, repetitive oscillations — 1 Hz and above
MIN_HZ = 1.0

# Max amplitude in pixels — big jumps are normal saccades, not nystagmus
MAX_AMPLITUDE_PX = 20.0

# Minimum amplitude — eye must actually be moving
MIN_AMPLITUDE_PX = 3.0

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!")

def get_iris_center(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy

def analyze_video(path):
    print(f"Analyzing video at: {path}")
    print(f"File exists: {os.path.exists(path)}")
    print(f"File size: {os.path.getsize(path)} bytes")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "Could not open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"FPS: {fps}")

    ensure_model()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    x_positions = []
    y_positions = []
    frame_idx = 0

    # Skip the first second to let the eye settle
    frames_to_skip = int(fps * 1.0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx <= frames_to_skip:
            continue

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
    print(f"Tracked {len(x_positions)} frames after skipping first second")

    if len(x_positions) < 10:
        return None, "Not enough eye tracking data"

    x = np.array(x_positions)
    y = np.array(y_positions)
    x -= np.mean(x)
    y -= np.mean(y)

    x_amplitude = np.max(x) - np.min(x)
    y_amplitude = np.max(y) - np.min(y)
    print(f"X amplitude: {x_amplitude:.2f}px, Y amplitude: {y_amplitude:.2f}px")

    # Pick the axis with more movement
    if x_amplitude >= y_amplitude:
        signal = x
        amplitude = x_amplitude
    else:
        signal = y
        amplitude = y_amplitude

    # Too still — eye not moving enough
    if amplitude < MIN_AMPLITUDE_PX:
        return None, "No significant nystagmus detected in this scan."

    # Too much movement — big saccades, not nystagmus
    if amplitude > MAX_AMPLITUDE_PX:
        return None, "No significant nystagmus detected in this scan."

    # Detect peaks and troughs with raised threshold
    peaks,   _ = find_peaks(signal, height=np.std(signal) * 1.5, distance=fps * 0.1)
    troughs, _ = find_peaks(-signal, height=np.std(signal) * 1.5, distance=fps * 0.1)
    all_extremes = np.sort(np.concatenate([peaks, troughs]))

    if len(all_extremes) < 2:
        return None, "No clear oscillation detected in this scan."

    intervals = np.diff(all_extremes) / fps
    hz = 1.0 / (np.mean(intervals) * 2)
    hz = round(hz, 2)
    print(f"Raw Hz: {hz}")

    # Below 1 Hz is too slow to be nystagmus
    if hz < MIN_HZ:
        return None, "No significant nystagmus detected in this scan."

    return hz, None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Nystagmus Tracker server is running!'})

@app.route('/analyze', methods=['POST'])
def analyze():
    print("Files received:", request.files)
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    video_file = request.files['video']
    tmp_path = os.path.join(tempfile.gettempdir(), 'eye_scan_temp.mp4')
    video_file.save(tmp_path)
    print(f"Video saved to: {tmp_path}")

    hz, error = analyze_video(tmp_path)

    try:
        os.unlink(tmp_path)
    except:
        pass

    if error:
        print(f"Result: {error}")
        return jsonify({'error': error}), 500

    print(f"Result: {hz} Hz")
    return jsonify({'hz': hz})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
