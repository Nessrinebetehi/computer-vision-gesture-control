import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pyautogui
import os
import urllib.request
import time

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
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# ───── STATE ─────
prev_x, prev_y = 0, 0
smooth = 0.15   # 🔥 أسرع
last_action = 0
delay = 0.35

# ───── LANDMARKS ─────
def get_landmarks(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, w, h

    pts = [(lm.x, lm.y) for lm in result.hand_landmarks[0]]
    return pts, w, h

# ───── FINGERS ─────
def fingers_up(pts):
    fingers = []

    fingers.append(pts[4][0] < pts[3][0])  # thumb

    for tip in [8, 12, 16, 20]:
        fingers.append(pts[tip][1] < pts[tip - 2][1])

    return fingers

# ───── MAIN ─────
def detect_hand_gesture(frame):
    global prev_x, prev_y, last_action

    pts, w, h = get_landmarks(frame)
    if pts is None:
        return None

    fingers = fingers_up(pts)

    # 🎯 MOUSE (محسن)
    if fingers == [0,1,0,0,0]:
        # 🟢 scaling correct
        x = int(pts[8][0] * screen_w * 1.2)
        y = int(pts[8][1] * screen_h * 1.2)

        # 🟢 limit
        x = max(0, min(screen_w-1, x))
        y = max(0, min(screen_h-1, y))

        # 🟢 smoothing
        x = int(prev_x + (x - prev_x) * smooth)
        y = int(prev_y + (y - prev_y) * smooth)

        pyautogui.moveTo(x, y)

        prev_x, prev_y = x, y
        return "mouse"

    # 🖱️ LEFT CLICK (hand open - FIXED)
    if fingers.count(True) >= 4:
        if time.time() - last_action > delay:
            pyautogui.click()
            last_action = time.time()
        return "click"

    # 🖱️ RIGHT CLICK (fist)
    if fingers.count(True) == 0:
        if time.time() - last_action > delay:
            pyautogui.rightClick()
            last_action = time.time()
        return "right_click"

    # 🔼 SCROLL UP (✌️ V sign)
    if fingers == [0,1,1,0,0]:
        pyautogui.scroll(80)
        return "scroll_up"

    # 🔽 SCROLL DOWN (thumb only)
    if fingers == [1,0,0,0,0]:
        pyautogui.scroll(-80)
        return "scroll_down"

    return None

# ───── DRAW ─────
def draw_hand_landmarks(frame):
    pts, w, h = get_landmarks(frame)
    if pts:
        for p in pts:
            cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 3, (0,255,0), -1)
    return frame

# ───── RESET ─────
def reset_state():
    global prev_x, prev_y
    prev_x, prev_y = 0, 0