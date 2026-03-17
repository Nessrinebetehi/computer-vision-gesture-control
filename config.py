# ─────────────────────────────────────────────
#  config.py  –  all tunable values live here
#  Change these numbers to tune sensitivity.
#  Do NOT hardcode any of these in other files.
# ─────────────────────────────────────────────

# ── Face landmark indices (MediaPipe FaceMesh)
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# ── Head gesture sensitivity
# Pixels of nose movement (per frame) needed to trigger a gesture.
# Lower  → more sensitive (triggers on small movements)
# Higher → less sensitive (requires bigger head tilts)
HEAD_THRESHOLD = 14   # pixels to trigger a gesture (the intentional tilt)
# pixels — movements smaller than this are ignored (return-to-center noise)
HEAD_DEAD_ZONE = 6

# ── Cooldown between any two gesture triggers (seconds)
# Prevents one head tilt from firing the same command 20× across 20 frames.
# seconds to ignore ALL gestures after one fires (time to return to center)
COOLDOWN_SECONDS = 1.2

# ── Swipe sensitivity (for hand_gestures.py)
SWIPE_THRESHOLD = 40  # pixels of wrist movement to trigger a swipe

# ── MediaPipe detection confidence
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# ── Webcam
CAMERA_INDEX = 0          # try 1 if your webcam doesn't open
FLIP_FRAME = True       # True = mirror image (more natural)

# ── Activation key
TOGGLE_KEY = ord('g')     # press G to turn gesture mode ON / OFF
QUIT_KEY = ord('q')     # press Q to exit the program

# ── Gesture → OS command mapping
# Keys must match the strings returned by detect_head_gesture()
# and detect_hand_gesture() exactly.
GESTURE_COMMANDS = {
    # ── Head gestures
    # next window (Alt+Tab)
    "head_right":  ("alt_tab", "right"),
    # prev window (Alt+Shift+Tab)
    "head_left":   ("alt_tab", "left"),
    # open Win+Tab task view
    "head_up":     ("task_view",),
    "head_down":   ("scroll", 0, -3),                       # scroll down

    # ── Hand static poses
    # pause / play media
    "open_palm":   ("hotkey", "space"),
    # screenshot (Win+Shift+S)
    "fist":        ("screenshot",),
    # show desktop (Win+D)
    "thumb_up":    ("hotkey", "win", "d"),

    # ── Hand swipes
    "swipe_right": ("hotkey", "ctrl", "tab"),              # next tab
    "swipe_left":  ("hotkey", "ctrl", "shift", "tab"),   # prev tab
    # continuous scroll up
    "scroll_up":   ("scroll_continuous", 0, 5),
    # continuous scroll down
    "scroll_down": ("scroll_continuous", 0, -5),
}
