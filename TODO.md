# Gesture-Based Mouse Control with Index Finger TODO

## Approved Plan Summary

- Detect when **only index finger raised** (others curled, hold ~0.25s = 8 frames).
- Then **continuously move cursor** to track index fingertip position (smoothed, normalized to screen).
- Use `pyautogui.moveTo(x, y)` directly in `hand_gestures.py` during continuous mode.
- Minimal API changes; integrate into existing detection flow after other gestures.
- Visual feedback: Highlight index tip, show status label.

## Step-by-Step Implementation

### Step 1: [PENDING] Add constants and state to hand_gestures.py

- Add `INDEX_TIP = 8`, `INDEX_MCP = 5`
- Add state: `_mouse_active = False`, `_mouse_hold_frames = 0`, `_smooth_tip = None`
- Import `pyautogui`, get `screen_w, screen_h = pyautogui.size()`

### Step 2: [PENDING] Add \_detect_index_mouse function

- Check: index extended, thumb/others curled (`tip_y > mcp_y`).
- Hold logic like static poses.

### Step 3: [PENDING] Integrate into detect_hand_gesture

- After scroll/swipes/static checks.
- Continuous move + return "index_mouse" during active.

### Step 4: [PENDING] Update draw_hand_landmarks

- Draw smoothed tip circle, "MOUSE: ON" label.

### Step 5: [PENDING] Add to config.py

- `"index_mouse": ("noop",)` (since handled directly)

### Step 6: [PENDING] Update reset_state

### Step 7: [DONE] Test & Tune
