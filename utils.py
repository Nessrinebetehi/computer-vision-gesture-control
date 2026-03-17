import os
import pyautogui
import time
import config

def execute_command(gesture):
    global _task_view_open
    if gesture is None: return

    command = config.GESTURE_COMMANDS.get(gesture)
    if not command: return

    # --- Action continue (Scroll) ---
    if command[0] == "scroll_continuous":
        pyautogui.scroll(command[2])
        return

    # --- Actions avec Cooldown (une seule fois) ---
    if not _should_trigger(): return

    action = command[0]
    try:
        if action == "hotkey":
            pyautogui.hotkey(*command[1:])
        
        elif action == "type_text":
            pyautogui.write(command[1], interval=0.1)
            print(f"  [Action] Écriture : {command[1]}")

        elif action == "open_folder":
            os.startfile("explorer.exe")
            print("  [Action] Ouverture de l'Explorateur")

        elif action == "alt_tab":
            if command[1] == "right": pyautogui.hotkey("alt", "tab")
            else: pyautogui.hotkey("alt", "shift", "tab")

        # ... (Garder le reste de ta fonction execute_command originale ici)
    except Exception as e:
        print(f"Erreur : {e}")