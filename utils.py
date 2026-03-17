# ─────────────────────────────────────────────
#  utils.py
#  Supports two modes:
#    NORMAL MODE  — head left/right = Alt+Tab switch
#                   head up         = open Task View (Win+Tab)
#                   head down       = scroll down
#    TASK VIEW MODE (after Win+Tab opens) —
#                   head left/right = navigate thumbnails (arrow keys)
#                   head down       = confirm selection (Enter)
#                   head up         = close Task View (Escape)
# ─────────────────────────────────────────────

import time
import cv2
import pyautogui
import config

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

try:
    import win32gui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    print("[WARN] pywin32 not installed — pip install pywin32")

# ── State
_last_trigger_time = 0
_target_hwnd = None
_task_view_open = False   # True while Win+Tab overlay is visible


def set_target_window():
    global _target_hwnd
    if WIN32_AVAILABLE:
        _target_hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(_target_hwnd)
        print(f"[TARGET] Commands will go to: '{title}'")
    else:
        print("[TARGET] pywin32 not available — click your target window manually.")


def _refocus_target():
    if WIN32_AVAILABLE and _target_hwnd and not _task_view_open:
        try:
            win32gui.SetForegroundWindow(_target_hwnd)
            time.sleep(0.05)
        except Exception:
            pass


def _should_trigger():
    global _last_trigger_time
    now = time.time()
    if now - _last_trigger_time >= config.COOLDOWN_SECONDS:
        _last_trigger_time = now
        return True
    return False


def execute_command(gesture):
    global _task_view_open

    if gesture is None:
        return

    # scroll_continuous bypasses the cooldown — it's meant to fire every few frames
    command = config.GESTURE_COMMANDS.get(gesture)
    if command and command[0] == "scroll_continuous":
        try:
            pyautogui.scroll(command[2])
        except Exception as e:
            print(f"[ERROR] scroll failed: {e}")
        return

    if not _should_trigger():
        return

    print(
        f"[GESTURE] {gesture}  (task_view={'ON' if _task_view_open else 'OFF'})")

    # ══════════════════════════════════════════
    #  TASK VIEW MODE — Win+Tab overlay is open
    # ══════════════════════════════════════════
    if _task_view_open:
        if gesture == "head_right":
            # Move right through thumbnails
            pyautogui.press("right")
            print("  → thumbnail: right")

        elif gesture == "head_left":
            # Move left through thumbnails
            pyautogui.press("left")
            print("  → thumbnail: left")

        elif gesture == "head_down":
            # Confirm — press Enter to switch to selected window
            pyautogui.press("enter")
            _task_view_open = False
            print("  → selected window")

        elif gesture == "head_up":
            # Cancel — close Task View without switching
            pyautogui.press("escape")
            _task_view_open = False
            print("  → task view closed")

        return   # don't fall through to normal mode

    # ══════════════════════════════════════════
    #  NORMAL MODE
    # ══════════════════════════════════════════
    command = config.GESTURE_COMMANDS.get(gesture)
    if command is None:
        print(f"[WARN] No command mapped for: '{gesture}'")
        return

    _refocus_target()

    action = command[0]
    try:
        if action == "hotkey":
            pyautogui.hotkey(*command[1:])

        elif action == "scroll":
            pyautogui.scroll(command[2])

        elif action == "press":
            pyautogui.press(command[1])

        elif action == "alt_tab":
            if command[1] == "right":
                pyautogui.hotkey("alt", "tab")
            else:
                pyautogui.hotkey("alt", "shift", "tab")

        elif action == "screenshot":
            # Win+Shift+S  opens the Snipping Tool overlay on Windows 10/11
            pyautogui.hotkey("win", "shift", "s")
            print("  → screenshot snip tool opened")

        elif action == "task_view":
            # Open Win+Tab — Task View overlay
            pyautogui.hotkey("win", "tab")
            time.sleep(0.4)   # wait for overlay animation to finish
            _task_view_open = True
            print(
                "  → task view opened — tilt left/right to browse, down to confirm, up to cancel")

    except Exception as e:
        print(f"[ERROR] PyAutoGUI failed for '{gesture}': {e}")


def draw_overlay(frame, gesture_mode, current_gesture):
    global _task_view_open
    h, w, _ = frame.shape

    # Status bar
    bar_color = (0, 200, 80) if gesture_mode else (120, 120, 120)
    status_text = "GESTURE MODE: ON  [G=off | Q=quit]" if gesture_mode \
                  else "GESTURE MODE: OFF [G=on  | Q=quit]"
    cv2.rectangle(frame, (8, 8), (w - 8, 44), (0, 0, 0), -1)
    cv2.rectangle(frame, (8, 8), (w - 8, 44), bar_color, 1)
    cv2.putText(frame, status_text, (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 2)

    # Task View mode indicator
    if _task_view_open:
        cv2.rectangle(frame, (8, 50), (w - 8, 86), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 50), (w - 8, 86), (0, 180, 255), 1)
        cv2.putText(frame, "TASK VIEW: left/right=browse  down=select  up=cancel",
                    (16, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    # Last gesture label
    if current_gesture:
        label = f"Gesture: {current_gesture}"
        cv2.rectangle(frame, (8, h - 48), (340, h - 8), (0, 0, 0), -1)
        cv2.putText(frame, label, (16, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)

    return frame