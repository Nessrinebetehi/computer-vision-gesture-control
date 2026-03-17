# Gesture-Based Computer Control System

Control your computer using head and hand gestures via webcam — no mouse needed.

## How it works

```
Webcam → MediaPipe landmarks → Movement delta → Gesture classifier → OS command
```

| Gesture     | Action                            |
| ----------- | --------------------------------- |
| Head left   | Previous window (Alt + ←)         |
| Head right  | Next window (Alt + →)             |
| Head up     | Scroll up                         |
| Head down   | Scroll down                       |
| Open palm   | Pause / play (Space)              |
| Swipe right | Next tab (Ctrl + Tab)             |
| Swipe left  | Previous tab (Ctrl + Shift + Tab) |

## Setup

```bash
pip install opencv-python mediapipe pyautogui numpy
python main.py
```

## Controls

| Key | Action                       |
| --- | ---------------------------- |
| `G` | Toggle gesture mode ON / OFF |
| `Q` | Quit                         |

## Project structure

```
gesture-control/
├── main.py            # entry point — wires everything together
├── face_gestures.py   # MediaPipe FaceMesh, head tilt detection
├── hand_gestures.py   # MediaPipe Hands, palm + swipe detection
├── config.py          # all tunable thresholds and constants
├── utils.py           # overlay drawing + OS command execution
└── README.md
```

## Tuning

All sensitivity values are in `config.py`:

- `HEAD_THRESHOLD` — pixels of nose movement to trigger a gesture (default 15)
- `COOLDOWN_SECONDS` — minimum seconds between two triggers (default 0.6)
- `MIN_DETECTION_CONFIDENCE` — how sure MediaPipe must be before tracking (default 0.6)

## Built with

- [MediaPipe](https://developers.google.com/mediapipe) — real-time landmark detection
- [OpenCV](https://opencv.org/) — webcam capture and frame rendering
- [PyAutoGUI](https://pyautogui.readthedocs.io/) — OS keyboard/mouse simulation
