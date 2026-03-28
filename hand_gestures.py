import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import pyautogui
import os
import urllib.request
import time
import math

# ───── INIT ─────
pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Téléchargement du modèle MediaPipe...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Modèle téléchargé.")

BaseOptions           = python.BaseOptions
HandLandmarker        = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode           = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2
)
detector = HandLandmarker.create_from_options(options)

# ───── STATE ─────
prev_x, prev_y  = 0, 0
smooth          = 0.25
gesture_counter = {}
HOLD_FRAMES     = 5
last_pts        = None
ZOOM_COOLDOWN   = 0.4
last_zoom_time  = 0

# ───── LANDMARKS ─────
def get_landmarks(frame):
    global last_pts
    h, w, _   = frame.shape
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp  = int(time.time() * 1000)
    result     = detector.detect_for_video(mp_image, timestamp)

    if not result.hand_landmarks:
        last_pts = None
        return None, w, h

    pts_list = [[(lm.x, lm.y) for lm in hand] for hand in result.hand_landmarks]
    last_pts = pts_list
    return pts_list, w, h

# ───── FINGERS ─────
def fingers_up(pts):
    fingers = []
    fingers.append(pts[4][0] < pts[3][0])  # pouce
    for tip in [8, 12, 16, 20]:
        fingers.append(pts[tip][1] < pts[tip - 2][1])
    return fingers

# ✅ NOUVELLE FONCTION (FIX ZOOM OUT)
def is_fist(pts):
    return (
        pts[8][1] > pts[6][1] and
        pts[12][1] > pts[10][1] and
        pts[16][1] > pts[14][1] and
        pts[20][1] > pts[18][1]
    )

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

# ───── DETECT GESTURE ─────
def detect_hand_gesture(frame):
    global prev_x, prev_y, last_zoom_time

    pts_list, w, h = get_landmarks(frame)

    if pts_list is None:
        return None

    # ── 2 MAINS : ZOOM ──
    if len(pts_list) == 2:
        fingers1 = fingers_up(pts_list[0])
        fingers2 = fingers_up(pts_list[1])
        now = time.time()

        # 🖐️🖐️ Zoom IN
        if fingers1.count(True) >= 4 and fingers2.count(True) >= 4:
            if (now - last_zoom_time) > ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '=')
                last_zoom_time = now
                return "zoom_in"

        # ✊✊ Zoom OUT (FIXED)
        if is_fist(pts_list[0]) and is_fist(pts_list[1]):
            if (now - last_zoom_time) > ZOOM_COOLDOWN:
                pyautogui.hotkey('ctrl', '-')
                last_zoom_time = now
                return "zoom_out"

        return None

    # ── 1 MAIN : GESTES ──
    if len(pts_list) < 1:
        return None

    fingers = fingers_up(pts_list[0])

    # 🔊 Volume +
    if fingers[1] and fingers[2] and fingers[3] and stable_gesture("volume_up"):
        pyautogui.press("volumeup")
        return "volume_up"

    # 🔉 Volume -
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and stable_gesture("volume_down"):
        pyautogui.press("volumedown")
        return "volume_down"

    # 🖱️ Souris
    if fingers == [False, True, False, False, False]:
        margin = 0.15
        x_norm = (pts_list[0][8][0] - margin) / (1 - 2 * margin)
        y_norm = (pts_list[0][8][1] - margin) / (1 - 2 * margin)

        x_norm = max(0, min(1, x_norm))
        y_norm = max(0, min(1, y_norm))

        x = int(prev_x + (x_norm * screen_w - prev_x) * smooth)
        y = int(prev_y + (y_norm * screen_h - prev_y) * smooth)

        pyautogui.moveTo(x, y)
        prev_x, prev_y = x, y
        return "mouse"

    # 🔼 Scroll up
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        pyautogui.scroll(30)
        return "scroll_up"

    # 🔽 Scroll down
    if fingers[0] and not any(fingers[1:]):
        pyautogui.scroll(-30)
        return "scroll_down"

    # 🖱️ Clic gauche
    if fingers.count(True) >= 4 and stable_gesture("click"):
        pyautogui.click()
        return "click"

    # 🖱️ Clic droit
    if fingers.count(True) == 0 and stable_gesture("right"):
        pyautogui.rightClick()
        return "right_click"

    return None

# ───── DRAW ─────
def draw_hand_landmarks(frame):
    global last_pts
    if last_pts:
        h, w, _ = frame.shape
        for hand in last_pts:
            for p in hand:
                cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, (0, 255, 0), -1)
    return frame

# ───── MAIN ─────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    print("Contrôle actif — q pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        action = detect_hand_gesture(frame)
        frame  = draw_hand_landmarks(frame)

        if action:
            cv2.putText(frame, action, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.imshow("Hand Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

def get_hand_center():
    global last_pts

    if not last_pts:
        return None

    hand = last_pts[0]  # première main

    xs = [p[0] for p in hand]
    ys = [p[1] for p in hand]

    cx = int(sum(xs) / len(xs))
    cy = int(sum(ys) / len(ys))

    return (cx, cy)    