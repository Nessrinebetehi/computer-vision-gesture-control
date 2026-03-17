# ─────────────────────────────────────────────
#  hand_gestures.py
#
#  Static poses (hold ~0.25s to fire once):
#    "open_palm"      — 4+ fingers extended        → pause/play
#    "thumb_up"       — only thumb up              → show desktop
#    "fist"           — all fingers closed         → screenshot
#
#  Continuous scroll (fires every SCROLL_INTERVAL frames while held):
#    "scroll_up"      — hand raised above head zone → scroll up
#    "scroll_down"    — hand lowered below waist zone → scroll down
#
#  Swipes (fast wrist movement, fires once):
#    "swipe_right"    — wrist moves right           → next tab
#    "swipe_left"     — wrist moves left            → prev tab
# ─────────────────────────────────────────────

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os
import config

# ── Download model if not present
MODEL_PATH = "hand_landmarker.task"


def _download_model():
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model... (one-time, ~30MB)")
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Hand model downloaded.")


_download_model()

# ── Initialize HandLandmarker
_options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
)
_detector = HandLandmarker.create_from_options(_options)

# ── Landmark indices
WRIST = 0
THUMB_TIP = 4
THUMB_MCP = 2
FINGERTIPS = [4,  8,  12, 16, 20]
MCP_JOINTS = [2,  5,   9, 13, 17]

# ── Hand connections for drawing
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ── Tuning constants
HOLD_REQUIRED = 8    # frames to hold a static pose before it fires (~0.25s)
SCROLL_INTERVAL = 4    # fire a scroll tick every N frames while hand is in scroll zone
DEAD_ZONE = 5    # pixels — ignore micro wrist movements

# ── Scroll zone thresholds (fraction of frame height)
# Hand wrist y < SCROLL_UP_ZONE   → scroll up   (hand raised high)
# Hand wrist y > SCROLL_DOWN_ZONE → scroll down (hand lowered low)
SCROLL_UP_ZONE = 0.30   # top 30% of frame
SCROLL_DOWN_ZONE = 0.75   # bottom 25% of frame

# ── State
_prev_wrist = None
_accum_x = 0
_last_pose = None
_hold_frames = 0
_scroll_frames = 0    # counts frames spent in scroll zone
_scrolling = None  # "scroll_up" | "scroll_down" | None


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


def detect_hand_gesture(frame):
    """
    Returns a gesture string or None each frame.

    Scroll gestures return continuously every SCROLL_INTERVAL frames
    while the wrist stays in the scroll zone — no need to re-swipe.
    Withdrawing the hand (or moving wrist back to neutral) stops scrolling.
    """
    global _prev_wrist, _accum_x, _last_pose, _hold_frames
    global _scroll_frames, _scrolling

    pts, frame_h, frame_w = _get_landmarks(frame)

    # ── Hand left frame — stop everything
    if pts is None:
        _prev_wrist = None
        _accum_x = 0
        _last_pose = None
        _hold_frames = 0
        _scroll_frames = 0
        if _scrolling is not None:
            print(f"[HAND] scroll stopped (hand withdrawn)")
            _scrolling = None
        return None

    wrist = pts[WRIST]
    wrist_y = wrist[1] / frame_h   # normalize to 0.0 – 1.0

    # ══════════════════════════════════════════
    #  CONTINUOUS SCROLL ZONE
    # ══════════════════════════════════════════
    if wrist_y < SCROLL_UP_ZONE:
        current_scroll = "scroll_up"
    elif wrist_y > SCROLL_DOWN_ZONE:
        current_scroll = "scroll_down"
    else:
        current_scroll = None

    if current_scroll:
        _scroll_frames += 1

        # First entry into zone — announce it
        if _scrolling != current_scroll:
            _scrolling = current_scroll
            _scroll_frames = 1
            print(
                f"[HAND] {current_scroll} started — keep hand here to continue")

        # Fire a scroll tick every SCROLL_INTERVAL frames
        if _scroll_frames % SCROLL_INTERVAL == 0:
            return current_scroll

        # Reset other state while scrolling
        _prev_wrist = wrist
        _last_pose = None
        _hold_frames = 0
        return None

    else:
        # Wrist is back in neutral zone — stop scrolling
        if _scrolling is not None:
            print(f"[HAND] scroll stopped (hand back to neutral)")
            _scrolling = None
            _scroll_frames = 0

    # ══════════════════════════════════════════
    #  HORIZONTAL SWIPE (left / right only)
    # ══════════════════════════════════════════
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
        print(f"[HAND DETECTED] {swipe_gesture}")
        _last_pose = None
        _hold_frames = 0
        return swipe_gesture

    # ══════════════════════════════════════════
    #  STATIC POSES (open_palm / fist / thumb_up)
    # ══════════════════════════════════════════
    current_pose = _classify_pose(pts)

    if current_pose == _last_pose:
        _hold_frames += 1
    else:
        _last_pose = current_pose
        _hold_frames = 0

    if _hold_frames == HOLD_REQUIRED and current_pose is not None:
        print(f"[HAND DETECTED] {current_pose}")
        return current_pose

    if _hold_frames > 0 and _hold_frames % 3 == 0:
        print(
            f"[HAND] holding '{current_pose}' — frame {_hold_frames}/{HOLD_REQUIRED}")

    return None


def draw_hand_landmarks(frame):
    """Draw hand skeleton, pose label, and scroll zone guides."""
    pts, frame_h, frame_w = _get_landmarks(frame)

    # ── Draw scroll zone lines on the frame
    up_y = int(SCROLL_UP_ZONE * frame_h)
    down_y = int(SCROLL_DOWN_ZONE * frame_h)
    cv2.line(frame, (0, up_y),   (frame_w, up_y),   (0, 200, 255), 1)
    cv2.line(frame, (0, down_y), (frame_w, down_y), (0, 200, 255), 1)
    cv2.putText(frame, "^ scroll up zone",   (8, up_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    cv2.putText(frame, "v scroll down zone", (8, down_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    if pts:
        for (a, b) in _HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
        for pt in pts:
            cv2.circle(frame, pt, 3, (255, 100, 0), -1)
        for tip in FINGERTIPS:
            if tip < len(pts):
                cv2.circle(frame, pts[tip], 6, (0, 200, 255), -1)

        # Show scroll state or pose label
        if _scrolling:
            label = f">>> {_scrolling} <<<"
            color = (0, 255, 80)
        else:
            pose = _classify_pose(pts)
            label = pose if pose else "..."
            color = (0, 255, 120) if pose else (160, 160, 160)

        cv2.putText(frame, label,
                    (pts[WRIST][0] + 10, pts[WRIST][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def reset_state():
    global _prev_wrist, _accum_x, _last_pose, _hold_frames, _scroll_frames, _scrolling
    _prev_wrist = None
    _accum_x = 0
    _last_pose = None
    _hold_frames = 0
    _scroll_frames = 0
    _scrolling = None