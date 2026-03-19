import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os
import config
import pyautogui

# ── Download model
MODEL_PATH = "hand_landmarker.task"

def _download_model():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model...")
        urllib.request.urlretrieve(url, MODEL_PATH)

_download_model()

# ── Init detector
_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
)
_detector = HandLandmarker.create_from_options(_options)

# ── Landmarks
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 2
INDEX_TIP = 8
INDEX_MCP = 5

FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [2, 5, 9, 13, 17]

# ── Settings
HOLD_REQUIRED = 8
SCROLL_INTERVAL = 4
DEAD_ZONE = 5

SCROLL_UP_ZONE = 0.30
SCROLL_DOWN_ZONE = 0.75

# ── State
_prev_wrist = None
_accum_x = 0
_last_pose = None
_hold_frames = 0
_scroll_frames = 0
_scrolling = None

# ── Screen size
screen_w, screen_h = pyautogui.size()

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _get_landmarks(frame):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _detector.detect(mp_im)

    if not result.hand_landmarks:
        return None, h, w

    lm = result.hand_landmarks[0]
    return [(int(p.x * w), int(p.y * h)) for p in lm], h, w


def _is_finger_extended(pts, tip_idx, mcp_idx):
    return pts[tip_idx][1] < pts[mcp_idx][1] - 10


# ── NEW: index only detection
def _is_index_only(pts):
    index_up = pts[INDEX_TIP][1] < pts[INDEX_MCP][1] - 15

    others_down = all(
        pts[FINGERTIPS[i]][1] > pts[MCP_JOINTS[i]][1]
        for i in range(2, 5)
    )

    thumb_down = pts[THUMB_TIP][1] > pts[THUMB_MCP][1] - 5

    return index_up and others_down and thumb_down


def _detect_thumb_up(pts):
    thumb_raised = pts[THUMB_TIP][1] < pts[WRIST][1] - 40
    others_curled = all(
        pts[FINGERTIPS[i]][1] > pts[MCP_JOINTS[i]][1]
        for i in range(1, 5)
    )
    return thumb_raised and others_curled


def _classify_pose(pts):
    if _detect_thumb_up(pts):
        return "thumb_up"

    extended = sum(
        _is_finger_extended(pts, FINGERTIPS[i], MCP_JOINTS[i])
        for i in range(1, 5)
    )

    if extended >= 3:
        return "open_palm"
    elif extended == 0:
        if pts[THUMB_TIP][1] > pts[THUMB_MCP][1] - 5:
            return "fist"

    return None


# ─────────────────────────────────────────────
#  MAIN DETECTION
# ─────────────────────────────────────────────

def detect_hand_gesture(frame):
    global _prev_wrist, _accum_x, _last_pose, _hold_frames
    global _scroll_frames, _scrolling

    pts, frame_h, frame_w = _get_landmarks(frame)

    if pts is None:
        _prev_wrist = None
        _accum_x = 0
        _last_pose = None
        _hold_frames = 0
        _scroll_frames = 0
        _scrolling = None
        return None

    # ══════════════════════════════════════════
    #  🎯 INDEX MOUSE CONTROL (NEW FEATURE)
    # ══════════════════════════════════════════
    if _is_index_only(pts):
        ix, iy = pts[INDEX_TIP]

        x_norm = ix / frame_w
        y_norm = iy / frame_h

        screen_x = int(x_norm * screen_w)
        screen_y = int(y_norm * screen_h)

        pyautogui.moveTo(screen_x, screen_y, duration=0.01)

        return "index_mouse"

    wrist = pts[WRIST]
    wrist_y = wrist[1] / frame_h

    # ── SCROLL ZONE
    if wrist_y < SCROLL_UP_ZONE:
        current_scroll = "scroll_up"
    elif wrist_y > SCROLL_DOWN_ZONE:
        current_scroll = "scroll_down"
    else:
        current_scroll = None

    if current_scroll:
        _scroll_frames += 1

        if _scrolling != current_scroll:
            _scrolling = current_scroll
            _scroll_frames = 1

        if _scroll_frames % SCROLL_INTERVAL == 0:
            return current_scroll

        _prev_wrist = wrist
        return None
    else:
        _scrolling = None

    # ── SWIPE
    swipe_gesture = None
    if _prev_wrist is not None:
        dx = wrist[0] - _prev_wrist[0]
        if abs(dx) < DEAD_ZONE:
            dx = 0

        _accum_x += dx

        if _accum_x > config.SWIPE_THRESHOLD:
            swipe_gesture = "swipe_right"
            _accum_x = 0
        elif _accum_x < -config.SWIPE_THRESHOLD:
            swipe_gesture = "swipe_left"
            _accum_x = 0

    _prev_wrist = wrist

    if swipe_gesture:
        return swipe_gesture

    # ── STATIC POSES
    current_pose = _classify_pose(pts)

    if current_pose == _last_pose:
        _hold_frames += 1
    else:
        _last_pose = current_pose
        _hold_frames = 0

    if _hold_frames == HOLD_REQUIRED and current_pose is not None:
        return current_pose

    return None


def draw_hand_landmarks(frame):
    pts, frame_h, frame_w = _get_landmarks(frame)

    if pts:
        for pt in pts:
            cv2.circle(frame, pt, 3, (255, 100, 0), -1)

        # show index mouse mode
        if _is_index_only(pts):
            cv2.putText(frame, ">>> INDEX MOUSE <<<",
                        (pts[INDEX_TIP][0], pts[INDEX_TIP][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def reset_state():
    global _prev_wrist, _accum_x, _last_pose, _hold_frames, _scroll_frames, _scrolling
    _prev_wrist = None
    _accum_x = 0
    _last_pose = None
    _hold_frames = 0
    _scroll_frames = 0
    _scrolling = None