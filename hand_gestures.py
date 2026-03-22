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

# تحميل الموديل
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
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

# ───────── STATE ─────────
prev_x, prev_y = 0, 0
smooth_factor = 0.25
last_click_time = 0
click_delay = 0.4

# ───────── LANDMARKS (مرة واحدة فقط) ─────────
def get_landmarks(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, w, h

    pts = []
    for lm in result.hand_landmarks[0]:
        pts.append((lm.x, lm.y))

    return pts, w, h


# ───────── FINGERS ─────────
def fingers_up(pts):
    fingers = []

    # thumb
    fingers.append(pts[4][0] < pts[3][0])

    # fingers
    for tip in [8, 12, 16, 20]:
        fingers.append(pts[tip][1] < pts[tip - 2][1])

    return fingers


# ───────── MAIN ─────────
def detect_hand_gesture(frame):
    global prev_x, prev_y, last_click_time

    pts, w, h = get_landmarks(frame)
    if pts is None:
        return None

    fingers = fingers_up(pts)

    # 🎯 INDEX MOUSE (FIXED)
    if fingers == [0,1,0,0,0]:
        x = int(pts[8][0] * screen_w)
        y = int(pts[8][1] * screen_h)

        # smoothing
        x = int(prev_x + (x - prev_x) * smooth_factor)
        y = int(prev_y + (y - prev_y) * smooth_factor)

        pyautogui.moveTo(x, y)

        prev_x, prev_y = x, y
        return "index_mouse"

    # 🖱️ LEFT CLICK
    if fingers == [0,1,1,1,1]:
        if time.time() - last_click_time > click_delay:
            pyautogui.click()
            last_click_time = time.time()
        return "click"

    # 🖱️ RIGHT CLICK
    if fingers == [0,0,0,0,0]:
        if time.time() - last_click_time > click_delay:
            pyautogui.rightClick()
            last_click_time = time.time()
        return "right_click"

    # 🔼 SCROLL UP
    if pts[0][1] < 0.3:
        pyautogui.scroll(50)
        return "scroll_up"

    # 🔽 SCROLL DOWN
    if pts[0][1] > 0.7:
        pyautogui.scroll(-50)
        return "scroll_down"

    return None


# ───────── DRAW ─────────
def draw_hand_landmarks(frame):
    pts, w, h = get_landmarks(frame)
    if pts:
        for p in pts:
            cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 3, (0,255,0), -1)
    return frame


# ───────── RESET ─────────
def reset_state():
    global prev_x, prev_y
    prev_x, prev_y = 0, 0