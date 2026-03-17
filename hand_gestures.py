import cv2
import config

# Indices des points de la main
FINGERTIPS = [4, 8, 12, 16, 20]
MCP_JOINTS = [2, 5, 9, 13, 17]

def _classify_pose(pts):
    if not pts: return None
    
    # Détection des doigts levés (True/False)
    # Pouce : on compare l'axe X (car il se plie horizontalement)
    thumb_open = pts[4][0] < pts[2][0] - 15 
    # Autres doigts : on compare l'axe Y
    index_open = pts[8][1] < pts[6][1] - 15
    middle_open = pts[12][1] < pts[10][1] - 15
    ring_open = pts[16][1] < pts[14][1] - 15
    pinky_open = pts[20][1] < pts[18][1] - 15

    # --- RÈGLES DE GESTES ---
    # 1. PEINTURE / PEACE (Index + Majeur) -> ECRIRE
    if index_open and middle_open and not ring_open:
        return "peace_sign"
    
    # 2. POUCE LEVÉ -> OUVRIR DOSSIER
    if thumb_open and not index_open and not pinky_open:
        return "thumb_up"
    
    # 3. AURICULAIRE (Petit doigt) -> BUREAU
    if pinky_open and not index_open and not middle_open:
        return "pinky_up"

    # 4. POING -> FERMER (Alt+F4)
    if not any([index_open, middle_open, ring_open, pinky_open]):
        return "fist"

    # 5. MAIN ENTIÈRE -> PLAY/PAUSE
    if all([index_open, middle_open, ring_open, pinky_open]):
        return "open_palm"

    return None