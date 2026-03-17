# ─────────────────────────────────────────────
#  main.py  –  entry point
# ─────────────────────────────────────────────

import cv2
import time
import config
import face_gestures
import hand_gestures
from utils import draw_overlay, execute_command, set_target_window


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {config.CAMERA_INDEX}.")
        return

    gesture_mode = False
    current_gesture = None

    print("=" * 56)
    print("  Gesture Control System — ready")
    print()
    print("  HEAD GESTURES:")
    print("    Tilt right  → next window (Alt+Tab)")
    print("    Tilt left   → prev window (Alt+Shift+Tab)")
    print("    Tilt up     → open Task View (Win+Tab)")
    print("    Tilt down   → scroll down")
    print()
    print("  HAND GESTURES (static poses — hold ~0.25s):")
    print("    Open palm   → pause / play (Space)")
    print("    Fist        → screenshot  (Win+Shift+S)")
    print("    Thumb up    → show desktop (Win+D)")
    print()
    print("  HAND SWIPES (fast movement):")
    print("    Swipe right → next tab (Ctrl+Tab)")
    print("    Swipe left  → prev tab (Ctrl+Shift+Tab)")
    print("    Hand raised high  → scroll up   (continuous while held)")
    print("    Hand lowered low  → scroll down (continuous while held)")
    print()
    print("  G = toggle gesture mode ON/OFF")
    print("  Q = quit")
    print("=" * 56)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        if config.FLIP_FRAME:
            frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF

        if key == config.QUIT_KEY:
            break

        if key == config.TOGGLE_KEY:
            gesture_mode = not gesture_mode
            current_gesture = None
            face_gestures.reset_state()
            hand_gestures.reset_state()

            if gesture_mode:
                print("\n>>> Activating in 3 seconds — click your target window NOW!")
                for i in [3, 2, 1]:
                    print(f"    {i}...")
                    time.sleep(1)
                set_target_window()
                print(">>> Gesture mode: ON\n")
            else:
                print(">>> Gesture mode: OFF")

        # ── Always draw landmarks (looks great for the demo)
        frame = face_gestures.draw_landmarks(frame)
        frame = hand_gestures.draw_hand_landmarks(frame)

        # ── Gesture detection — face and hand are INDEPENDENT
        # Face gestures and hand gestures no longer block each other.
        # Both are checked every frame. First one to return a result wins,
        # but hand is NOT skipped just because face returned something.
        if gesture_mode:
            face_gesture = face_gestures.detect_head_gesture(frame)
            hand_gesture = hand_gestures.detect_hand_gesture(frame)

            if face_gesture is not None:
                print(f"[FACE] detected: {face_gesture}")
                current_gesture = face_gesture
                execute_command(face_gesture)
            elif hand_gesture is not None:
                print(f"[HAND] detected: {hand_gesture}")
                current_gesture = hand_gesture
                execute_command(hand_gesture)

        frame = draw_overlay(frame, gesture_mode, current_gesture)
        cv2.imshow("Gesture Control System", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")


if __name__ == "__main__":
    main()
