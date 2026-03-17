# --- Dans config.py ---
GESTURE_COMMANDS = {
    # -- Visage (on ne change rien)
    "head_right":  ("alt_tab", "right"),
    "head_left":   ("alt_tab", "left"),
    "head_up":     ("task_view",),
    "head_down":   ("scroll", 0, -3),

    # -- Main (Amélioré)
    "open_palm":   ("hotkey", "space"),          # Play/Pause
    "fist":        ("hotkey", "alt", "f4"),      # FERMER le fichier/fenêtre
    "thumb_up":    ("open_folder", "explorer"),  # OUVRIR l'Explorateur
    "peace_sign":  ("type_text", "Bonjour "),   # ECRIRE du texte
    "pinky_up":    ("hotkey", "win", "d"),       # MONTRER LE BUREAU (réduire tout)

    "swipe_right": ("hotkey", "ctrl", "tab"),
    "swipe_left":  ("hotkey", "ctrl", "shift", "tab"),
    "scroll_up":   ("scroll_continuous", 0, 8),
    "scroll_down": ("scroll_continuous", 0, -8),
}