# ─────────────────────────────────────────────
#  face_gestures.py  –  Person 1's module
#  Compatible with mediapipe >= 0.10
#  Uses the new FaceLandmarker Task API
# ─────────────────────────────────────────────

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import urllib.request
import os
import config

# ── Download the required model file if not present
MODEL_PATH = "face_landmarker.task"


def _download_model():
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmark model... (one-time, ~30MB)")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Model downloaded.")


_download_model()

# ── Initialize FaceLandmarker
_options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
_detector = FaceLandmarker.create_from_options(_options)

# ── State
_prev_nose = None
_accum_x = 0     # accumulated horizontal movement
_accum_y = 0     # accumulated vertical movement
# True while head is mid-gesture (hasn't returned to center yet)
_gesture_active = False

# ── Face mesh connections for manual drawing
_FACE_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332,
                                        284), (284, 251), (251, 389), (389, 356),
    (356, 454), (454, 323), (323, 361), (361,
                                         288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377,
                                         152), (152, 148), (148, 176), (176, 149),
    (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234),
    (234, 127), (127, 162), (162, 21), (21,
                                        54), (54, 103), (103, 67), (67, 109), (109, 10),
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155),
    (155, 133), (33, 246), (246, 161), (161, 160), (160,
                                                    159), (159, 158), (158, 157), (157, 173), (173, 133),
    (362, 382), (382, 381), (381, 380), (380,
                                         374), (374, 373), (373, 390), (390, 249),
    (249, 263), (362, 398), (398, 384), (384, 385), (385,
                                                     386), (386, 387), (387, 388), (388, 466), (466, 263),
    (1, 2), (2, 98), (98, 97), (1, 168), (168, 5), (5, 4),
    (61, 185), (185, 40), (40, 39), (39, 37), (37,
                                               0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
]


def get_nose_position(frame):
    """
    Returns the nose tip (x, y) pixel coords, or None if no face detected.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_im)

    if not result.face_landmarks:
        return None

    nose = result.face_landmarks[0][config.NOSE_TIP]
    return (int(nose.x * w), int(nose.y * h))


def detect_head_gesture(frame):
    """
    Accumulate nose movement across frames.
    Only triggers once the total displacement exceeds HEAD_THRESHOLD.
    Resets accumulator after firing so return-to-center is ignored.
    Returns: "head_left" | "head_right" | "head_up" | "head_down" | None
    """
    global _prev_nose, _accum_x, _accum_y, _gesture_active

    nose = get_nose_position(frame)

    if nose is None or _prev_nose is None:
        _prev_nose = nose
        return None

    dx = nose[0] - _prev_nose[0]
    dy = nose[1] - _prev_nose[1]
    _prev_nose = nose

    # Ignore tiny micro-movements (dead zone) — these are return-to-center noise
    if abs(dx) < config.HEAD_DEAD_ZONE:
        dx = 0
    if abs(dy) < config.HEAD_DEAD_ZONE:
        dy = 0

    # Accumulate movement in the dominant axis only
    # (prevents diagonal movement from double-triggering)
    if abs(dx) >= abs(dy):
        _accum_x += dx
        _accum_y = 0
    else:
        _accum_y += dy
        _accum_x = 0

    gesture = None

    if _accum_x < -config.HEAD_THRESHOLD:
        gesture = "head_left"
    elif _accum_x > config.HEAD_THRESHOLD:
        gesture = "head_right"
    elif _accum_y < -config.HEAD_THRESHOLD:
        gesture = "head_up"
    elif _accum_y > config.HEAD_THRESHOLD:
        gesture = "head_down"

    if gesture:
        # Reset accumulator immediately — return movement won't re-trigger
        _accum_x = 0
        _accum_y = 0

    return gesture


def draw_landmarks(frame):
    """
    Draw face mesh on the frame. Returns the annotated frame.
    """
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_im)

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]

        for (a, b) in _FACE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (0, 220, 100), 1)

        if config.NOSE_TIP < len(pts):
            cv2.circle(frame, pts[config.NOSE_TIP], 4, (0, 255, 200), -1)

    return frame


def reset_state():
    """Reset the previous nose position. Call when gesture mode is toggled OFF."""
    global _prev_nose
    _prev_nose = None

def get_face_center(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_im)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    cx = int(sum(xs) / len(xs) * w)
    cy = int(sum(ys) / len(ys) * h)

    return (cx, cy)    