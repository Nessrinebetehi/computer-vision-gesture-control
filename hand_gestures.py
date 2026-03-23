import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pyautogui
import os
import urllib.request
import time

# ───── INIT ─────
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ───── STATE ─────
prev_x, prev_y = 0, 0
smooth = 0.25
gesture_counter = {}
HOLD_FRAMES = 5
last_pts = None

# ───── LANDMARKS ─────
def get_landmarks(frame):
    global last_pts
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp)
    if not result.hand_landmarks:
        last_pts = None
        return None, w, h
    pts = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
    last_pts = pts
    return pts, w, h

# ───── FINGERS ─────
def fingers_up(pts):
    fingers = []
    fingers.append(pts[4][0] < pts[3][0])  # pouce
    for tip in [8, 12, 16, 20]:
        fingers.append(pts[tip][1] < pts[tip-2][1])
    return fingers

# ───── STABILITY ─────
def stable_gesture(name):
    global gesture_counter
    gesture_counter[name] = gesture_counter.get(name, 0) + 1
    for k in list(gesture_counter.keys()):
        if k != name:
            gesture_counter[k] = 0
    if gesture_counter[name] >= HOLD_FRAMES:
        gesture_counter[name] = 0
        return True
    return False

# ───── MAIN ─────
def detect_hand_gesture(frame):
    global prev_x, prev_y
    pts, w, h = get_landmarks(frame)
    if pts is None:
        return None

    fingers = fingers_up(pts)

    # 🔊 VOLUME CONTROL PRIORITAIRE (corrigé, sans conflit)
    # Volume Up : index + majeur + annulaire levés
    if fingers[1] and fingers[2] and fingers[3] and stable_gesture("volume_up"):
        pyautogui.press("volumeup")
        return "volume_up"

    # Volume Down : pouce + index levés seulement
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and stable_gesture("volume_down"):
        pyautogui.press("volumedown")
        return "volume_down"

    # 🎯 MOUSE CONTROL (index only)
    if fingers == [0,1,0,0,0]:
        margin = 0.15
        x_norm = (pts[8][0] - margin) / (1 - 2*margin)
        y_norm = (pts[8][1] - margin) / (1 - 2*margin)
        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))
        x = int(prev_x + (x_norm * screen_w - prev_x) * smooth)
        y = int(prev_y + (y_norm * screen_h - prev_y) * smooth)
        pyautogui.moveTo(x, y)
        prev_x, prev_y = x, y
        return "mouse"

    # 🔼 SCROLL UP (V sign)
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        pyautogui.scroll(30)
        return "scroll_up"

    # 🔽 SCROLL DOWN (thumb only)
    if fingers[0] and not any(fingers[1:]):
        pyautogui.scroll(-30)
        return "scroll_down"

    # 🖱️ LEFT CLICK (open hand)
    if fingers.count(True) >= 4 and stable_gesture("click"):
        pyautogui.click()
        return "click"

    # 🖱️ RIGHT CLICK (fist)
    if fingers.count(True) == 0 and stable_gesture("right"):
        pyautogui.rightClick()
        return "right_click"

    return None

# ───── DRAW ─────
def draw_hand_landmarks(frame):
    global last_pts
    if last_pts:
        h, w, _ = frame.shape
        for p in last_pts:
            cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 3, (0,255,0), -1)
    return frame

# ───── RESET ─────
def reset_state():
    global prev_x, prev_y, gesture_counter
    prev_x, prev_y = 0, 0
    gesture_counter = {}