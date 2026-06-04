import cv2
import numpy as np
import base64
import json
import time
import os
import math
import torch
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# Patch torch.load to allow unsafe loading (required for ultralytics weights)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from collections import defaultdict
import threading

# MediaPipe imports (Tasks API for 0.10.30+)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Project root (where yolov8n.pt lives)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

app = Flask(__name__)

# Configure CORS for production - allow requests from Vercel and local development
cors_origin = os.environ.get('CORS_ORIGIN', '*')
CORS(app, resources={
    r"/*": {
        "origins": cors_origin if cors_origin != '*' else '*',
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})


class HeadPoseEstimator:
    """Uses MediaPipe FaceLandmarker (Tasks API) to estimate head pose (yaw, pitch, roll)."""

    _MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

    def __init__(self):
        model_path = self._get_model()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # 3D model points for head pose estimation (generic face model)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Nose tip (1)
            (0.0, -330.0, -65.0),    # Chin (152)
            (-225.0, 170.0, -135.0), # Left eye left corner (263)
            (225.0, 170.0, -135.0),  # Right eye right corner (33)
            (-150.0, -150.0, -125.0),# Left mouth corner (287)
            (150.0, -150.0, -125.0)  # Right mouth corner (57)
        ], dtype=np.float64)

        # Landmark indices for pose estimation
        self.pose_landmark_indices = [1, 152, 263, 33, 287, 57]

    def _get_model(self):
        model_dir = os.path.join(PROJECT_ROOT, 'models')
        model_path = os.path.join(model_dir, 'face_landmarker.task')
        if not os.path.exists(model_path):
            os.makedirs(model_dir, exist_ok=True)
            print(f"Downloading MediaPipe face landmark model...")
            import requests as req
            r = req.get(self._MODEL_URL, timeout=60)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                f.write(r.content)
            print(f"Model saved to {model_path}")
        return model_path

    def estimate(self, frame):
        """Estimate head poses for all faces in the frame.
        Returns list of dicts with yaw, pitch, roll and face bounding box."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        poses = []
        if not detection_result.face_landmarks:
            return poses

        h, w, _ = frame.shape

        # Camera internals
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        for face_landmarks in detection_result.face_landmarks:
            # Extract 2D image points
            image_points = np.array([
                (face_landmarks[idx].x * w,
                 face_landmarks[idx].y * h)
                for idx in self.pose_landmark_indices
            ], dtype=np.float64)

            # SolvePnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            # Convert rotation vector to euler angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            proj_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
            )

            pitch = euler_angles[0][0]
            yaw = euler_angles[1][0]
            roll = euler_angles[2][0]

            # Compute face bounding box from all landmarks
            x_coords = [lm.x * w for lm in face_landmarks]
            y_coords = [lm.y * h for lm in face_landmarks]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            pose_data = {
                'yaw': float(yaw),
                'pitch': float(pitch),
                'roll': float(roll),
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'looking_away': abs(yaw) > 30,
                'looking_down': pitch < -15,
                'head_turn': abs(yaw) > 45,
            }
            poses.append(pose_data)

        return poses

    def draw_poses(self, frame, poses):
        """Draw head pose indicators on frame."""
        for pose in poses:
            x, y, w_box, h_box = pose['bbox']

            # Color based on attention
            if pose['head_turn']:
                color = (0, 0, 255)   # Red for head turn
                label = f"HEAD TURN (yaw:{pose['yaw']:.0f})"
            elif pose['looking_down']:
                color = (0, 165, 255) # Orange for looking down
                label = f"LOOKING DOWN (pitch:{pose['pitch']:.0f})"
            elif pose['looking_away']:
                color = (0, 255, 255) # Yellow for looking away
                label = f"LOOKING AWAY (yaw:{pose['yaw']:.0f})"
            else:
                color = (0, 255, 0)   # Green for attentive
                label = "ATTENTIVE"

            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame


# ─── COCO class IDs for reference ─────────────────────────────────
# YOLOv8 uses COCO 80 classes. Key ones for exam proctoring:
# 0: person, 24: backpack, 25: umbrella, 26: handbag, 27: tie,
# 28: suitcase, 39: bottle, 41: cup, 56: chair, 57: couch,
# 58: potted plant, 62: tv, 63: laptop, 64: mouse, 65: remote,
# 66: keyboard, 67: cell phone, 73: book, 74: clock, 75: vase,
# 76: scissors, 77: teddy bear

# Mapping from COCO class names to exam violation types
COCO_TO_EXAM_TYPE = {
    # Direct mappings (high confidence)
    'cell phone': 'PHONE',
    'laptop': 'DEVICE',
    'remote': 'DEVICE',
    'keyboard': 'DEVICE',
    'mouse': 'DEVICE',
    'tv': 'DEVICE',
    'book': 'TEXTBOOK',
    
    # Contextual mappings (need spatial/size analysis)  
    'clock': 'WATCH_CANDIDATE',       # Small clock near wrist = watch
    'handbag': 'EARPHONE_CANDIDATE',  # Small near head = earphone case
    'tie': 'EARPHONE_CANDIDATE',      # Cord-like near head = earphone wire
    'scissors': 'CHIT_CANDIDATE',     # Small object near hands
    'bottle': 'DEVICE',               # Can hide phone/chit
    'backpack': 'BAG',                # Bags shouldn't be at desk
}

# Severity scores for each violation type
VIOLATION_SCORES = {
    'PHONE': 30,
    'EARPHONE': 25,
    'WATCH': 20,
    'CHIT': 25,
    'TEXTBOOK': 30,
    'NOTEBOOK': 25,
    'DEVICE': 20,
    'HEAD_TURN': 15,
    'LEANING': 10,
    'MULTIPLE_PEOPLE': 20,
    'NO_PERSON': 15,
}


class DetectionEngine:
    def __init__(self):
        print("Loading YOLOv8 model...")

        model_path = os.path.join(PROJECT_ROOT, 'yolov8n.pt')
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}, downloading...")
            model_path = 'yolov8n.pt'  # Let ultralytics download it

        try:
            self.yolo_model = YOLO(model_path)
            print("YOLOv8 model loaded successfully!")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # Initialize MediaPipe head pose estimator
        try:
            self.head_pose = HeadPoseEstimator()
            print("MediaPipe Face Mesh loaded successfully!")
        except Exception as e:
            print(f"Error loading MediaPipe: {e}")
            self.head_pose = None

        self.person_tracks = defaultdict(lambda: {
            'frames': [],
            'head_poses': [],
            'last_seen': 0,
            'violations': []
        })
        self.next_track_id = 1
        self.frame_count = 0

        # ALL potentially relevant COCO classes for exam proctoring
        self.detection_classes = [
            'person', 'cell phone', 'book', 'laptop', 'remote',
            'keyboard', 'mouse', 'tv', 'clock', 'scissors',
            'handbag', 'tie', 'backpack', 'bottle', 'cup',
            'umbrella', 'suitcase', 'vase'
        ]

        self.class_colors = {
            'PHONE': (0, 0, 255),       # Red
            'EARPHONE': (255, 0, 255),   # Magenta
            'WATCH': (0, 215, 255),      # Gold
            'CHIT': (0, 255, 0),         # Green
            'TEXTBOOK': (0, 165, 255),   # Orange
            'NOTEBOOK': (0, 165, 255),   # Orange
            'DEVICE': (128, 0, 128),     # Purple
            'person': (0, 255, 0),       # Green
            'default': (255, 255, 255)   # White
        }

        # Detection history for temporal smoothing (reduces false positives)
        self.detection_history = defaultdict(lambda: {
            'count': 0,
            'last_seen': 0,
            'confidences': []
        })
        self.HISTORY_WINDOW = 5  # Frames to track
        self.MIN_CONSISTENCY = 2  # Min detections in window to confirm

    def _classify_detection(self, det, persons, frame_h, frame_w):
        """Classify a COCO detection into an exam-specific violation type.
        Uses spatial analysis relative to detected persons."""
        coco_class = det['class']
        bbox = det['bbox']  # [x, y, w, h]
        conf = det['confidence']
        x, y, w, h = bbox
        
        # Direct high-confidence mappings
        if coco_class == 'cell phone':
            return 'PHONE', conf * 1.1  # Boost confidence
        
        if coco_class == 'book':
            # Large book = textbook, small = notebook/chit
            area = w * h
            frame_area = frame_h * frame_w
            relative_size = area / frame_area if frame_area > 0 else 0
            
            if relative_size > 0.03:
                return 'TEXTBOOK', conf
            elif relative_size > 0.008:
                return 'NOTEBOOK', conf * 0.95
            else:
                return 'CHIT', conf * 0.85
        
        if coco_class in ['laptop', 'keyboard', 'mouse', 'remote', 'tv']:
            return 'DEVICE', conf
        
        if coco_class == 'clock':
            # Small clock detection = likely a watch (wristwatch)
            area = w * h
            frame_area = frame_h * frame_w
            relative_size = area / frame_area if frame_area > 0 else 0
            aspect = w / max(h, 1)
            
            # Watch characteristics: small, roughly square
            if relative_size < 0.02 and 0.5 < aspect < 2.0:
                # Check if near a person's wrist area (lower half of person bbox)
                is_near_person = False
                for p in persons:
                    px, py, pw, ph = p['bbox']
                    # Wrist area: bottom 40% of person, sides
                    wrist_y_start = py + int(ph * 0.6)
                    if y > wrist_y_start and x > px - 30 and x < px + pw + 30:
                        is_near_person = True
                        break
                
                if is_near_person:
                    return None, 0  # Skip watch detection (removed)
                else:
                    return None, 0  # Skip watch detection (removed)
            else:
                return 'DEVICE', conf * 0.6  # Wall clock or similar
        
        if coco_class in ['handbag', 'tie']:
            # Earphone detection removed — skip handbag/tie detections
            return None, 0
        
        if coco_class == 'scissors':
            # Small object = possibly chit/note
            return 'CHIT', conf * 0.65
        
        if coco_class == 'bottle':
            # Bottle can hide things, low priority alert
            return None, 0  # Don't alert on bottles
        
        return None, 0

    def detect_objects(self, frame):
        if self.yolo_model is None:
            print("YOLO model not loaded, cannot detect objects")
            return []

        try:
            # Run YOLO with lower confidence for catching more items
            results = self.yolo_model(
                frame, 
                verbose=False, 
                conf=0.15,    # Lower threshold to catch more items
                iou=0.4       # NMS IoU for exam scenarios
            )[0]
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

        h, w, _ = frame.shape
        raw_detections = []
        persons = []

        # First pass: collect all raw COCO detections
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]

            if class_name in self.detection_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                det = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2 - x1, y2 - y1]
                }
                raw_detections.append(det)
                
                if class_name == 'person':
                    persons.append(det)

        # Second pass: classify non-person detections into exam types
        exam_detections = []
        
        for det in raw_detections:
            if det['class'] == 'person':
                exam_detections.append(det)
                continue
            
            exam_type, adjusted_conf = self._classify_detection(det, persons, h, w)
            
            if exam_type and adjusted_conf > 0.12:
                # Apply temporal smoothing
                det_key = f"{exam_type}_{det['bbox'][0]//50}_{det['bbox'][1]//50}"
                history = self.detection_history[det_key]
                history['count'] = min(history['count'] + 1, self.HISTORY_WINDOW)
                history['last_seen'] = self.frame_count
                history['confidences'].append(adjusted_conf)
                if len(history['confidences']) > self.HISTORY_WINDOW:
                    history['confidences'] = history['confidences'][-self.HISTORY_WINDOW:]
                
                # Use smoothed confidence
                avg_conf = sum(history['confidences']) / len(history['confidences'])
                final_conf = min(avg_conf * 1.05, 0.99)  # Slight boost for consistency
                
                exam_detections.append({
                    'class': det['class'],
                    'exam_type': exam_type,
                    'confidence': final_conf,
                    'raw_confidence': det['confidence'],
                    'bbox': det['bbox']
                })

        # Decay old detection history
        stale_keys = []
        for key, hist in self.detection_history.items():
            if self.frame_count - hist['last_seen'] > self.HISTORY_WINDOW * 2:
                stale_keys.append(key)
        for key in stale_keys:
            del self.detection_history[key]

        return exam_detections

    def estimate_head_poses(self, frame):
        """Run MediaPipe face mesh head pose estimation."""
        if self.head_pose is None:
            return []
        try:
            return self.head_pose.estimate(frame)
        except Exception as e:
            print(f"Head pose estimation error: {e}")
            return []

    def analyze_behavior(self, frame, detections, head_poses=None):
        persons = [d for d in detections if d['class'] == 'person']

        behavior_events = []

        if len(persons) >= 2:
            for i, p1 in enumerate(persons):
                for p2 in persons[i+1:]:
                    dist = self._calculate_distance(p1['bbox'], p2['bbox'])
                    if dist < 100:
                        behavior_events.append({
                            'type': 'PROXIMITY_ALERT',
                            'distance': int(dist),
                            'confidence': 0.75
                        })

        # Add head pose behavior events from MediaPipe
        if head_poses:
            for pose in head_poses:
                if pose['head_turn']:
                    behavior_events.append({
                        'type': 'HEAD_TURN',
                        'yaw': float(pose['yaw']),
                        'pitch': float(pose['pitch']),
                        'confidence': 0.85
                    })
                elif pose['looking_down']:
                    behavior_events.append({
                        'type': 'LOOKING_DOWN',
                        'yaw': float(pose['yaw']),
                        'pitch': float(pose['pitch']),
                        'confidence': 0.80
                    })
                elif pose['looking_away']:
                    behavior_events.append({
                        'type': 'LOOKING_AWAY',
                        'yaw': float(pose['yaw']),
                        'pitch': float(pose['pitch']),
                        'confidence': 0.70
                    })

        return behavior_events

    def _calculate_distance(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        c1 = (x1 + w1//2, y1 + h1//2)
        c2 = (x2 + w2//2, y2 + h2//2)

        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

    def draw_detections(self, frame, detections, head_poses=None):
        for det in detections:
            x, y, w, h = det['bbox']
            class_name = det.get('exam_type', det['class'])
            conf = det['confidence']

            color = self.class_colors.get(class_name, self.class_colors['default'])

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Show exam type label instead of COCO class
            display_name = class_name
            if det.get('exam_type'):
                display_name = det['exam_type']
            
            label = f"{display_name}: {conf:.2f}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw head pose overlays
        if head_poses and self.head_pose:
            frame = self.head_pose.draw_poses(frame, head_poses)

        return frame

    def process_frame(self, frame, session_id=None):
        self.frame_count += 1

        detections = self.detect_objects(frame)
        head_poses = self.estimate_head_poses(frame)
        behaviors = self.analyze_behavior(frame, detections, head_poses)

        person_count = len([d for d in detections if d['class'] == 'person'])

        # Prohibited items: anything with an exam_type
        prohibited_items = [d for d in detections if d.get('exam_type')]

        result = {
            'frame_number': self.frame_count,
            'timestamp': time.time(),
            'detections': detections,
            'person_count': person_count,
            'prohibited_items': prohibited_items,
            'behaviors': behaviors,
            'head_poses': [
                {
                    'yaw': float(p['yaw']),
                    'pitch': float(p['pitch']),
                    'roll': float(p['roll']),
                    'looking_away': bool(p['looking_away']),
                    'looking_down': bool(p['looking_down']),
                    'head_turn': bool(p['head_turn']),
                    'bbox': [int(x) for x in p['bbox']]
                }
                for p in head_poses
            ]
        }

        return result

    def process_video(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 5 == 0:
                result = self.process_frame(frame)
                results.append(result)

                if progress_callback:
                    progress = int((frame_idx / total_frames) * 100)
                    progress_callback(frame_idx, total_frames, progress, len(results))

            frame_idx += 1

        cap.release()
        return results

    def process_stream(self, stream_url, duration=60):
        cap = cv2.VideoCapture(stream_url)

        if not cap.isOpened():
            raise ValueError(f"Cannot open stream: {stream_url}")

        start_time = time.time()
        results = []

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            if self.frame_count % 5 == 0:
                result = self.process_frame(frame)
                results.append(result)

            self.frame_count += 1

            time.sleep(0.1)

        cap.release()
        return results

detection_engine = DetectionEngine()

current_stream_url = None
resolved_stream_url = None
stream_thread = None
stream_active = False
latest_frame = None
latest_result = None

# Webcam globals
webcam_cap = None
webcam_thread = None
webcam_active = False
webcam_latest_frame = None
webcam_latest_result = None


def try_open_stream(url: str):
    """Try to open a stream using OpenCV. Returns (cap, resolved_url) or (None, None)."""
    candidates = []
    if url:
        candidates.append(url)

        if url.endswith('/vid'):
            candidates.append(url[:-4] + '/video')
            candidates.append(url[:-4] + '/mjpeg')
            candidates.append(url[:-4] + '/mjpegfeed')
            candidates.append(url[:-4] + '/videofeed')
        elif url.endswith('/'):
            candidates.append(url + 'video')
            candidates.append(url + 'mjpeg')
            candidates.append(url + 'mjpegfeed')
            candidates.append(url + 'videofeed')
        else:
            candidates.append(url + '/video')
            candidates.append(url + '/mjpeg')
            candidates.append(url + '/mjpegfeed')
            candidates.append(url + '/videofeed')

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            cap = cv2.VideoCapture(candidate)
            if cap.isOpened():
                return cap, candidate
            cap.release()
        except Exception:
            try:
                cap.release()
            except Exception:
                pass

    return None, None


def stream_capture_loop():
    global current_stream_url, resolved_stream_url, stream_active, latest_frame, latest_result

    cap = None

    while stream_active:
        if resolved_stream_url:
            if cap is None or not cap.isOpened():
                print(f"Opening stream: {resolved_stream_url}")
                cap = cv2.VideoCapture(resolved_stream_url)
                if not cap.isOpened():
                    print(f"Failed to open stream: {resolved_stream_url}")
                    time.sleep(2)
                    continue

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame, retrying...")
                cap.release()
                cap = None
                time.sleep(1)
                continue

            latest_frame = frame.copy()

            if detection_engine.frame_count % 2 == 0:
                result = detection_engine.process_frame(frame)
                latest_result = result

            time.sleep(0.2)
        else:
            time.sleep(1.0)

    if cap:
        cap.release()


def webcam_capture_loop(camera_index=0):
    """Capture frames from the laptop webcam via OpenCV."""
    global webcam_active, webcam_latest_frame, webcam_latest_result, webcam_cap

    webcam_cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not webcam_cap.isOpened():
        # Fallback without DSHOW
        webcam_cap = cv2.VideoCapture(camera_index)

    if not webcam_cap.isOpened():
        print(f"Failed to open webcam index {camera_index}")
        webcam_active = False
        return

    webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Webcam opened successfully (index {camera_index})")

    frame_counter = 0
    while webcam_active:
        ret, frame = webcam_cap.read()
        if not ret:
            print("Webcam read failed, retrying...")
            time.sleep(0.5)
            continue

        webcam_latest_frame = frame.copy()
        frame_counter += 1

        # Process every 3rd frame for performance
        if frame_counter % 3 == 0:
            result = detection_engine.process_frame(frame)
            webcam_latest_result = result

        time.sleep(0.033)  # ~30 FPS capture

    webcam_cap.release()
    webcam_cap = None
    print("Webcam released")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': bool(detection_engine.yolo_model is not None),
        'mediapipe_loaded': bool(detection_engine.head_pose is not None),
        'stream_active': bool(stream_active),
        'webcam_active': bool(webcam_active),
        'current_stream': current_stream_url
    })


@app.route('/cameras', methods=['GET'])
def list_cameras():
    """List available camera devices."""
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append({
                'index': i,
                'name': f'Camera {i}',
                'resolution': f'{w}x{h}'
            })
            cap.release()
    return jsonify({'cameras': available})


# ── Webcam (laptop camera) endpoints ─────────────────────────────────

@app.route('/webcam/start', methods=['POST'])
def start_webcam():
    global webcam_thread, webcam_active, webcam_latest_frame, webcam_latest_result

    if webcam_active:
        return jsonify({'status': 'already_running', 'message': 'Webcam is already active'})

    data = request.get_json() or {}
    camera_index = data.get('camera_index', 0)

    webcam_latest_frame = None
    webcam_latest_result = None
    webcam_active = True

    webcam_thread = threading.Thread(target=webcam_capture_loop, args=(camera_index,), daemon=True)
    webcam_thread.start()

    # Wait briefly for camera to initialize
    time.sleep(1.5)

    if not webcam_active:
        return jsonify({'error': 'Failed to open webcam'}), 400

    return jsonify({
        'status': 'started',
        'camera_index': camera_index,
        'message': 'Webcam capture started. Use /webcam/frame for detections.'
    })


@app.route('/webcam/stop', methods=['POST'])
def stop_webcam():
    global webcam_active, webcam_latest_frame, webcam_latest_result

    webcam_active = False
    webcam_latest_frame = None
    webcam_latest_result = None

    return jsonify({'status': 'stopped'})


@app.route('/webcam/frame', methods=['GET'])
def get_webcam_frame():
    """Get the latest webcam frame with YOLO + MediaPipe detections."""
    global webcam_latest_frame, webcam_latest_result

    if webcam_latest_frame is None:
        return jsonify({'error': 'No frame available', 'waiting': True}), 404

    # Draw detections on frame
    frame_with_boxes = detection_engine.draw_detections(
        webcam_latest_frame.copy(),
        webcam_latest_result.get('detections', []) if webcam_latest_result else [],
        webcam_latest_result.get('head_poses', []) if webcam_latest_result else []
    )

    _, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_base64,
        'result': webcam_latest_result,
        'timestamp': time.time()
    })


@app.route('/webcam/mjpeg', methods=['GET'])
def get_webcam_mjpeg():
    """Get an MJPEG video stream from the webcam with overlays."""
    global webcam_latest_frame, webcam_latest_result, webcam_active

    def generate():
        while webcam_active and webcam_latest_frame is not None:
            try:
                frame_with_boxes = detection_engine.draw_detections(
                    webcam_latest_frame.copy(),
                    webcam_latest_result.get('detections', []) if webcam_latest_result else [],
                    webcam_latest_result.get('head_poses', []) if webcam_latest_result else []
                )

                _, buffer = cv2.imencode('.jpg', frame_with_boxes)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.05)
            except Exception as e:
                print(f"Webcam MJPEG error: {e}")
                break

    return Response(stream_with_context(generate()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── IP Camera / DroidCam stream endpoints ─────────────────────────────

@app.route('/stream/start', methods=['POST'])
def start_stream():
    global current_stream_url, resolved_stream_url, stream_thread, stream_active, latest_frame, latest_result

    data = request.get_json()
    stream_url = data.get('stream_url')

    if not stream_url:
        return jsonify({'error': 'No stream_url provided'}), 400

    current_stream_url = stream_url
    resolved_stream_url = None
    latest_frame = None
    latest_result = None
    stream_active = True

    cap, resolved = try_open_stream(stream_url)
    if cap is None:
        stream_active = False
        return jsonify({
            'error': 'Cannot open stream_url with OpenCV',
            'stream_url': stream_url,
            'hint': 'For DroidCam, use the exact HTTP MJPEG URL shown in the app (e.g. /video or /mjpegfeed).',
            'tried': 'original plus common variants: /video, /mjpeg, /mjpegfeed, /videofeed'
        }), 400
    cap.release()
    resolved_stream_url = resolved

    if stream_thread is None or not stream_thread.is_alive():
        stream_thread = threading.Thread(target=stream_capture_loop, daemon=True)
        stream_thread.start()

    return jsonify({
        'status': 'started',
        'stream_url': stream_url,
        'resolved_stream_url': resolved_stream_url,
        'message': 'Stream capture started. Use /stream/frame to get frames.'
    })


@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    global stream_active, current_stream_url, resolved_stream_url

    stream_active = False
    current_stream_url = None
    resolved_stream_url = None

    return jsonify({'status': 'stopped'})


@app.route('/stream/frame', methods=['GET'])
def get_stream_frame():
    global latest_frame, latest_result

    if latest_frame is None:
        return jsonify({'error': 'No frame available', 'waiting': True}), 404

    frame_with_boxes = detection_engine.draw_detections(
        latest_frame.copy(),
        latest_result.get('detections', []) if latest_result else [],
        latest_result.get('head_poses', []) if latest_result else []
    )

    _, buffer = cv2.imencode('.jpg', frame_with_boxes, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_base64,
        'result': latest_result,
        'timestamp': time.time()
    })


@app.route('/stream/mjpeg', methods=['GET'])
def get_mjpeg_stream():
    global latest_frame, latest_result, stream_active

    def generate():
        while stream_active and latest_frame is not None:
            try:
                frame_with_boxes = detection_engine.draw_detections(
                    latest_frame.copy(),
                    latest_result.get('detections', []) if latest_result else [],
                    latest_result.get('head_poses', []) if latest_result else []
                )

                _, buffer = cv2.imencode('.jpg', frame_with_boxes)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.05)
            except Exception as e:
                print(f"Stream error: {e}")
                break

    return Response(stream_with_context(generate()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Single frame analysis endpoints ───────────────────────────────────

@app.route('/detect/frame', methods=['POST'])
def detect_frame():
    data = request.get_json()

    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = data['image']
        # Strip data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        decoded = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        result = detection_engine.process_frame(frame)

        # Also return annotated frame
        annotated = detection_engine.draw_detections(
            frame.copy(),
            result.get('detections', []),
            result.get('head_poses', [])
        )
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buf).decode('utf-8')

        result['annotated_frame'] = annotated_b64

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/detect/video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    # Use temp directory that works on Windows
    import tempfile
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, file.filename)
    file.save(video_path)

    results = detection_engine.process_video(video_path)

    # Clean up
    try:
        os.remove(video_path)
    except:
        pass

    return jsonify({
        'total_frames': len(results),
        'results': results[-10:]
    })


@app.route('/detect/stream', methods=['POST'])
def detect_stream():
    data = request.get_json()
    stream_url = data.get('stream_url')

    if not stream_url:
        return jsonify({'error': 'No stream_url provided'}), 400

    return jsonify({
        'message': 'Use /stream/start to begin capture, then /stream/frame to get results'
    })


if __name__ == '__main__':
    print("="*60)
    print("  AI Detection Server (YOLO + MediaPipe)")
    print("  Enhanced Exam Proctoring Detection")
    print("="*60)
    print("Detectable Items:")
    print("  📱 Phone (cell phone)")
    print("  🎧 Earphone (spatial analysis near head)")
    print("  ⌚ Watch (small clock-like objects near wrist)")
    print("  📝 Chit (small paper/objects)")
    print("  📖 Textbook (large books)")
    print("  📓 Notebook (medium books)")
    print("  💻 Devices (laptop, keyboard, remote, etc.)")
    print("  🔄 Head Turn (MediaPipe face mesh)")
    print("")
    print("Endpoints:")
    print("  GET  /health                     - Server health check")
    print("  GET  /cameras                    - List available cameras")
    print("")
    print("  Webcam (Laptop Camera):")
    print("  POST /webcam/start               - Start laptop webcam capture")
    print("  POST /webcam/stop                - Stop webcam capture")
    print("  GET  /webcam/frame               - Get latest webcam frame (JSON)")
    print("  GET  /webcam/mjpeg               - Get webcam MJPEG stream")
    print("")
    print("  IP Camera / DroidCam:")
    print("  POST /stream/start               - Start capturing from URL")
    print("  POST /stream/stop                - Stop capturing")
    print("  GET  /stream/frame               - Get latest frame (JSON)")
    print("  GET  /stream/mjpeg               - Get MJPEG video stream")
    print("")
    print("  Analysis:")
    print("  POST /detect/frame               - Analyze single image")
    print("  POST /detect/video               - Upload & analyze video")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
