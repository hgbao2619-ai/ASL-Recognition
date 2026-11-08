"""
ASL Recognition - Real-time American Sign Language Recognition Pipeline

Architecture:
  [Webcam Live]
    └──> [MediaPipe Hand Tracking] (21 Landmarks with 3D context & Handedness)
          └──> [Geometric Rule-based Classifier] (21 Core Signs: A, B, C, D, F, G, H, I, K, L, M, N, O, R, S, T, U, V, W, X, Y)
                └──> [Debounce Buffer] (12 frames >= 80% confidence + 1.0s Cooldown + 1.8s Space)
                      └──> [AI Sentence Builder] (Bilingual EN/VI Natural Sentence Synthesis with Offline Fallbacks)
                            └──> [Thread-safe Non-blocking TTS] (Dedicated worker thread with queue)
                                  └──> [OpenCV Dark HUD] (Minimalist HUD with Unicode Vietnamese text support)

Installation:
  pip install -r requirements.txt
Run application:
  python app.py
"""

import os
import sys
import time
import math
import queue
import threading
import unicodedata
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Union

# ---------------------------------------------------------
# DEPENDENCY CHECKS & SAFE IMPORTS
# ---------------------------------------------------------
try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False

try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# ---------------------------------------------------------
# SYSTEM CONFIGURATION (21 CORE SIGNS)
# ---------------------------------------------------------
CORE_SIGNS = [
    'A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
]
MIN_CONFIRM_FRAMES = 12             # Consecutive frames required for confirmation
MIN_CONFIDENCE_THRESHOLD = 0.80     # Confidence threshold >= 80%
COOLDOWN_SECONDS = 1.0              # Cooldown period after confirmed sign
SPACE_HOLD_SECONDS = 1.8            # Hold 5-finger open hand 1.8s to insert space
GEMINI_MODEL = "gemini-2.5-flash"


# ---------------------------------------------------------
# UNICODE TEXT DRAWING UTILITIES (VIETNAMESE SUPPORT)
# ---------------------------------------------------------
def strip_vietnamese_accents(text: str) -> str:
    """
    Converts Vietnamese accented characters to plain ASCII for safe rendering with cv2.putText.
    e.g., 'Tôi yêu bạn' -> 'Toi yeu ban', 'Chào bạn' -> 'Chao ban'
    """
    text = text.replace('đ', 'd').replace('Đ', 'D')
    normalized = unicodedata.normalize('NFD', text)
    stripped = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return unicodedata.normalize('NFC', stripped)


_FONT_CACHE: Dict[int, Any] = {}

def get_pil_font(size: int = 20):
    if not HAS_PIL:
        return None
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]

    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\segoeui.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf"
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue

    try:
        font = ImageFont.load_default()
        _FONT_CACHE[size] = font
        return font
    except Exception:
        return None


def draw_text_unicode(
    frame: Any,
    text: str,
    pos: Tuple[int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int],
    thickness: int = 1,
    is_unicode: bool = False
) -> Any:
    """
    Draws text onto frame. If text contains non-ASCII characters (e.g. Vietnamese):
    - Uses PIL if available for crisp full Unicode rendering.
    - Otherwise cleanly strips accents with unicodedata and uses cv2.putText to guarantee no junk symbols.
    """
    if cv2 is None or frame is None:
        return frame

    is_plain_ascii = all(ord(c) < 128 for c in text)
    if is_plain_ascii and not is_unicode:
        scale = font_size / 30.0
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)
        return frame

    font = get_pil_font(font_size) if HAS_PIL else None
    if HAS_PIL and font is not None:
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            pil_pos = (pos[0], max(0, pos[1] - int(font_size * 0.85)))
            draw.text(pil_pos, text, font=font, fill=color_rgb)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    # Fallback: clean ASCII via accent stripping
    clean_text = strip_vietnamese_accents(text)
    scale = font_size / 30.0
    cv2.putText(frame, clean_text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------
# GEOMETRIC INVARIANT CLASSIFIER (HANDEDNESS INVARIANT)
# ---------------------------------------------------------
class GeometricASLClassifier:
    """
    Geometric Invariant Feature Classifier:
    1. Horizontal mirror sync: cv2.flip(frame, 1).
    2. Handedness normalization: Inverts normalized X axis for Left hand so all lateral rules apply symmetrically.
    3. Shift origin: Wrist (0) as (0, 0).
    4. Scale normalization: divided by Wrist (0) to Middle MCP (9) distance.
    5. Kinematic analysis: finger vectors, extension states, inter-finger distances, and posture constraints.
    """
    def __init__(self):
        self.classes = CORE_SIGNS

    @staticmethod
    def _dist(p1: Union[Tuple[float, float], Dict[str, float]], p2: Union[Tuple[float, float], Dict[str, float]]) -> float:
        x1 = p1['x'] if isinstance(p1, dict) else p1[0]
        y1 = p1['y'] if isinstance(p1, dict) else p1[1]
        x2 = p2['x'] if isinstance(p2, dict) else p2[0]
        y2 = p2['y'] if isinstance(p2, dict) else p2[1]
        return math.hypot(x1 - x2, y1 - y2)

    def extract_features(self, landmarks_21, is_left_hand: bool = False) -> Optional[Dict[str, Any]]:
        if not landmarks_21 or len(landmarks_21) < 21:
            return None

        # 1. Extract 2D (x, y) & preserve raw 3D z
        raw_pts = []
        for lm in landmarks_21:
            if hasattr(lm, 'x'):
                raw_pts.append({'x': lm.x, 'y': lm.y, 'z': getattr(lm, 'z', 0.0)})
            elif isinstance(lm, dict):
                raw_pts.append({'x': lm.get('x', 0.0), 'y': lm.get('y', 0.0), 'z': lm.get('z', 0.0)})
            else:
                raw_pts.append({'x': lm[0], 'y': lm[1], 'z': lm[2] if len(lm) > 2 else 0.0})

        # 2. Shift origin to Wrist (landmark 0)
        wrist = raw_pts[0]
        rel_pts = [{'x': p['x'] - wrist['x'], 'y': p['y'] - wrist['y'], 'z': p['z']} for p in raw_pts]

        # 3. Scale invariance & Handedness normalization:
        # If is_left_hand=True, invert X axis so lateral rules (A, L, S, T, N, M, G, etc.) work identically
        ref_scale = self._dist(rel_pts[0], rel_pts[9])
        if ref_scale < 1e-5:
            ref_scale = 1.0

        x_factor = -1.0 if is_left_hand else 1.0
        norm_pts = [{'x': (p['x'] / ref_scale) * x_factor, 'y': p['y'] / ref_scale} for p in rel_pts]

        # Distance from fingertips to Wrist (0)
        d_thumb_wrist  = self._dist(norm_pts[4],  norm_pts[0])
        d_index_wrist  = self._dist(norm_pts[8],  norm_pts[0])
        d_middle_wrist = self._dist(norm_pts[12], norm_pts[0])
        d_ring_wrist   = self._dist(norm_pts[16], norm_pts[0])
        d_pinky_wrist  = self._dist(norm_pts[20], norm_pts[0])

        # Thumb extension: natural, comfortable threshold
        d_thumb_mcp = self._dist(norm_pts[4], norm_pts[2])
        d_thumb_index_mcp = self._dist(norm_pts[4], norm_pts[5])
        thumb_extended = d_thumb_wrist > 0.85 or (d_thumb_wrist > 0.76 and (d_thumb_index_mcp > 0.48 or d_thumb_mcp > 0.40))

        # Finger vectors from MCP to Tip
        dx_index = norm_pts[8]['x'] - norm_pts[5]['x']
        dy_index = norm_pts[8]['y'] - norm_pts[5]['y']
        dx_middle = norm_pts[12]['x'] - norm_pts[9]['x']
        dy_middle = norm_pts[12]['y'] - norm_pts[9]['y']

        # Upright extensions: MUST point UP into the sky (dy < -0.45, |dy| > |dx| * 1.15, tip.y < pip.y)
        is_index_pointing_up = dy_index < -0.45 and abs(dy_index) > abs(dx_index) * 1.15 and norm_pts[8]['y'] < norm_pts[6]['y']
        is_middle_pointing_up = dy_middle < -0.45 and abs(dy_middle) > abs(dx_middle) * 1.15 and norm_pts[12]['y'] < norm_pts[10]['y']

        index_ext_up = is_index_pointing_up and d_index_wrist > 1.20
        middle_ext_up = is_middle_pointing_up and d_middle_wrist > 1.20

        # Sideways extensions: MUST point horizontally across (|dx| > 0.45 and |dx| > |dy| * 1.15)
        is_index_pointing_side = abs(dx_index) > 0.45 and abs(dx_index) > abs(dy_index) * 1.15
        is_middle_pointing_side = abs(dx_middle) > 0.45 and abs(dx_middle) > abs(dy_middle) * 1.15

        index_ext_side = is_index_pointing_side and d_index_wrist > 1.15
        middle_ext_side = is_middle_pointing_side and d_middle_wrist > 1.15

        index_extended  = index_ext_up or index_ext_side
        middle_extended = middle_ext_up or middle_ext_side
        ring_extended   = norm_pts[16]['y'] < norm_pts[14]['y'] and d_ring_wrist > 1.15
        pinky_extended  = norm_pts[20]['y'] < norm_pts[18]['y'] and d_pinky_wrist > 1.15

        # Upright hook for X sign:
        # Lower half (MCP 5 -> PIP 6) extends upright, upper half (PIP 6 -> DIP 7 -> Tip 8) bent down into a hook
        is_index_lower_ext = (
            norm_pts[6]['y'] < norm_pts[5]['y'] - 0.12
            and abs(norm_pts[6]['x'] - norm_pts[5]['x']) < abs(norm_pts[6]['y'] - norm_pts[5]['y']) * 1.5
        )
        is_index_upper_bent_hook = (
            norm_pts[8]['y'] >= norm_pts[6]['y'] - 0.08
            or norm_pts[7]['y'] >= norm_pts[6]['y'] - 0.04
            or self._dist(norm_pts[8], norm_pts[5]) < self._dist(norm_pts[6], norm_pts[5]) * 1.45
        )
        is_index_hooked = is_index_lower_ext and is_index_upper_bent_hook and not index_ext_up and not index_ext_side

        # Inter-finger distances
        d_thumb_index  = self._dist(norm_pts[4], norm_pts[8])
        d_index_middle = self._dist(norm_pts[8], norm_pts[12])
        d_middle_ring  = self._dist(norm_pts[12], norm_pts[16])
        d_ring_pinky   = self._dist(norm_pts[16], norm_pts[20])
        d_thumb_middle = self._dist(norm_pts[4], norm_pts[12])
        d_thumb_pinky  = self._dist(norm_pts[4], norm_pts[20])

        return {
            "norm_pts": norm_pts,
            "raw_pts": raw_pts,
            "d_thumb_wrist": d_thumb_wrist,
            "d_index_wrist": d_index_wrist,
            "d_middle_wrist": d_middle_wrist,
            "d_ring_wrist": d_ring_wrist,
            "d_pinky_wrist": d_pinky_wrist,
            "d_thumb_index": d_thumb_index,
            "d_index_middle": d_index_middle,
            "d_middle_ring": d_middle_ring,
            "d_ring_pinky": d_ring_pinky,
            "d_thumb_middle": d_thumb_middle,
            "d_thumb_pinky": d_thumb_pinky,
            "thumb_ext": thumb_extended,
            "index_ext": index_extended,
            "middle_ext": middle_extended,
            "ring_ext": ring_extended,
            "pinky_ext": pinky_extended,
            "index_ext_up": index_ext_up,
            "middle_ext_up": middle_ext_up,
            "index_ext_side": index_ext_side,
            "middle_ext_side": middle_ext_side,
            "is_index_hooked": is_index_hooked,
            "is_left_hand": is_left_hand
        }

    def predict(self, landmarks_21, is_left_hand: bool = False) -> Tuple[Optional[str], float]:
        feats = self.extract_features(landmarks_21, is_left_hand=is_left_hand)
        if not feats:
            return None, 0.0

        scores: Dict[str, float] = {k: 0.0 for k in self.classes}
        p = feats["norm_pts"]
        raw = feats["raw_pts"]

        d_ti = feats["d_thumb_index"]
        d_im = feats["d_index_middle"]
        d_mr = feats["d_middle_ring"]
        d_rp = feats["d_ring_pinky"]
        d_tp = feats["d_thumb_pinky"]
        d_tm = feats["d_thumb_middle"]

        i_ext = feats["index_ext"]
        m_ext = feats["middle_ext"]
        r_ext = feats["ring_ext"]
        p_ext = feats["pinky_ext"]
        t_ext = feats["thumb_ext"]

        i_ext_up = feats["index_ext_up"]
        m_ext_up = feats["middle_ext_up"]
        i_ext_side = feats["index_ext_side"]
        m_ext_side = feats["middle_ext_side"]
        is_index_hooked = feats["is_index_hooked"]

        d_tw = feats["d_thumb_wrist"]
        d_iw = feats["d_index_wrist"]
        d_mw = feats["d_middle_wrist"]
        d_rw = feats["d_ring_wrist"]
        d_pw = feats["d_pinky_wrist"]

        # 1. Sign 'B': 4 fingers straight up touching, thumb folded across palm
        if i_ext and m_ext and r_ext and p_ext:
            finger_spread = d_im + d_mr + d_rp
            is_closed = finger_spread < 1.35
            thumb_folded = not t_ext or p[4]['x'] > p[5]['x'] - 0.25
            if is_closed and thumb_folded:
                scores['B'] = max(scores['B'], 0.88 + 0.12 * max(0.0, 1.0 - finger_spread / 1.35))

        # 1b. Sign 'D': Index finger pointing straight UP, 3 fingers (middle, ring, pinky) folded down, thumb touching near middle finger
        if i_ext_up and not m_ext and not r_ext and not p_ext:
            d_thumb_middle_nearest = min(
                d_tm,
                self._dist(p[4], p[10]),
                self._dist(p[4], p[11]),
                self._dist(p[4], p[12])
            )
            is_thumb_near_middle = d_thumb_middle_nearest <= 0.48
            if is_thumb_near_middle:
                closeness = max(0.0, 1.0 - d_thumb_middle_nearest / 0.48)
                scores['D'] = max(scores['D'], 0.90 + 0.10 * closeness)

        # 1c. Sign 'F': Middle, ring, and pinky fingers extended UP, thumb and index folded touching tips
        if not i_ext and m_ext and r_ext and p_ext:
            is_tip_touching = d_ti <= 0.42
            is_three_upright = p[12]['y'] < p[10]['y'] and p[16]['y'] < p[14]['y'] and p[20]['y'] < p[18]['y']
            if is_tip_touching and is_three_upright:
                tightness = max(0.0, 1.0 - d_ti / 0.42)
                scores['F'] = max(scores['F'], 0.90 + 0.10 * tightness)

        # 1d. Sign 'G': Index finger and thumb extended pointing sideways, 3 fingers folded completely down
        if i_ext_side and not i_ext_up and not m_ext and not r_ext and not p_ext:
            is_three_folded_completely = d_mw < 1.15 and d_rw < 1.15 and d_pw < 1.15
            d_thumb_middle_nearest = min(
                d_tm,
                self._dist(p[4], p[10]),
                self._dist(p[4], p[11]),
                self._dist(p[4], p[12])
            )
            if is_three_folded_completely and d_thumb_middle_nearest > 0.35:
                scores['G'] = max(scores['G'], 0.92)

        # 1e. Sign 'X': 4 fingers (pinky, ring, middle, thumb) folded completely down, index finger lower half upright and upper half bent into hook
        if is_index_hooked and not m_ext and not r_ext and not p_ext:
            is_three_folded_completely = d_mw < 1.18 and d_rw < 1.18 and d_pw < 1.18
            is_thumb_folded_completely = not t_ext or d_tw < 1.15 or p[4]['y'] >= p[5]['y'] - 0.05
            not_touching_index_tip = d_ti > 0.38
            if is_three_folded_completely and is_thumb_folded_completely and not_touching_index_tip:
                scores['X'] = 0.95
                scores['D'] = 0.0
                scores['S'] = 0.0
                scores['A'] = 0.0

        # 2. Sign 'V': Index and middle fingers pointing UP spread in a V, ring and pinky curled
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            if d_im > 0.35:
                scores['V'] = max(scores['V'], 0.88 + 0.12 * min(1.0, (d_im - 0.35) / 0.5))

        # 2a. Sign 'K': Index and middle fingers pointing UP spread, thumb upright placed directly between index and middle fingers
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            is_ring_pinky_curled = d_rw < 1.15 and d_pw < 1.15
            min_x = min(p[5]['x'], p[8]['x']) - 0.05
            max_x = max(p[9]['x'], p[12]['x']) + 0.05
            is_thumb_between_x = p[4]['x'] >= min_x and p[4]['x'] <= max_x
            is_thumb_upright = p[4]['y'] < p[5]['y']
            mid_index_middle_mcp = {'x': (p[5]['x'] + p[9]['x']) / 2, 'y': (p[5]['y'] + p[9]['y']) / 2}
            d_thumb_mid_mcp = self._dist(p[4], mid_index_middle_mcp)

            if is_ring_pinky_curled and is_thumb_between_x and is_thumb_upright and d_thumb_mid_mcp <= 0.40:
                scores['K'] = 0.94
                scores['V'] = 0.0

        # 2b. Sign 'W': Index, middle, and ring fingers extended UP in a W formation, pinky curled down, thumb folded
        if i_ext and m_ext and r_ext and not p_ext and not i_ext_side and not m_ext_side:
            is_upright = p[8]['y'] < p[6]['y'] and p[12]['y'] < p[10]['y'] and p[16]['y'] < p[14]['y']
            if is_upright:
                scores['W'] = max(scores['W'], 0.94)

        # 3. Sign 'U': Index and middle fingers pointing strictly UP together touching (parallel, not crossed), ring and pinky curled down
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            if d_im <= 0.35:
                closeness = max(0.0, 1.0 - d_im / 0.35)
                scores['U'] = max(scores['U'], 0.90 + 0.10 * closeness)

        # 3b. Sign 'R': Index and middle fingers extended UP and crossed together, middle finger crossed behind index finger
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            is_ring_pinky_curled = d_rw < 1.15 and d_pw < 1.15
            d_thumb_ring_nearest = min(self._dist(p[4], p[14]), self._dist(p[4], p[15]), self._dist(p[4], p[16]))
            d_thumb_pinky_nearest = min(self._dist(p[4], p[18]), self._dist(p[4], p[19]), self._dist(p[4], p[20]))
            is_thumb_touching_ring_pinky = (d_thumb_ring_nearest <= 0.48 or d_thumb_pinky_nearest <= 0.48) and p[4]['y'] >= p[5]['y'] - 0.05

            base_separation = p[9]['x'] - p[5]['x']
            tip_separation = p[12]['x'] - p[8]['x']
            is_genuine_crossing = (
                (base_separation > 0.04 and tip_separation <= -0.01)
                or (base_separation < -0.04 and tip_separation >= 0.01)
            )

            # Z-depth check (middle finger behind index finger)
            has_z = raw and len(raw) > 12 and 'z' in raw[8] and 'z' in raw[12]
            is_middle_behind = (raw[12]['z'] >= raw[8]['z'] - 0.04) if has_z else True

            if is_ring_pinky_curled and is_thumb_touching_ring_pinky and is_genuine_crossing and is_middle_behind:
                scores['R'] = 0.95
                scores['U'] = 0.0
                scores['V'] = 0.0

        # 4. Sign 'H': Index and middle fingers extended together pointing SIDEWAYS, ring and pinky curled
        if i_ext_side and m_ext_side and not i_ext_up and not m_ext_up and not r_ext and not p_ext:
            if d_im <= 0.38:
                parallelism = max(0.0, 1.0 - d_im / 0.38)
                scores['H'] = max(scores['H'], 0.88 + 0.12 * parallelism)

        # 5. Sign 'L': Thumb and index finger extended at right angle (~90°), others folded
        if i_ext and not m_ext and not r_ext and not p_ext:
            d_thumb_middle_nearest = min(d_tm, self._dist(p[4], p[10]), self._dist(p[4], p[11]), self._dist(p[4], p[12]))
            if d_ti > 0.65 and (t_ext or p[4]['x'] < p[3]['x']) and d_thumb_middle_nearest > 0.50:
                scores['L'] = max(scores['L'], 0.90 + 0.10 * min(1.0, d_ti / 1.2))

        # 6. Sign 'I': Only pinky extended straight up, others curled
        if p_ext and not i_ext and not m_ext and not r_ext:
            if not t_ext or d_tp <= 1.25:
                scores['I'] = max(scores['I'], 0.93)

        # 7. Sign 'Y': Thumb and pinky extended wide (Shaka), middle three fingers curled
        if p_ext and t_ext and not i_ext and not m_ext and not r_ext:
            if d_tp > 1.15:
                scores['Y'] = max(scores['Y'], 0.95)

        # 8. Sign 'O': Hand forming an open circular O-ring (both frontal palm and sideways views)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            is_tip_touching = d_ti <= 0.40
            tips_meet_directly = abs(p[8]['y'] - p[4]['y']) <= 0.22 and abs(p[8]['x'] - p[4]['x']) <= 0.28

            is_sideways_o = (
                is_tip_touching
                and p[6]['y'] < p[5]['y'] + 0.08
                and self._dist(p[6], p[2]) >= 0.30
                and self._dist(p[5], p[4]) >= 0.20
            )
            is_frontal_o = (
                is_tip_touching
                and tips_meet_directly
                and d_tm <= 0.52
                and d_im <= 0.38
                and d_tw >= 0.65
            )

            if is_sideways_o or is_frontal_o:
                touch_factor = max(0.0, 1.0 - d_ti / 0.40)
                scores['O'] = max(scores['O'], 0.88 + 0.12 * touch_factor)

        # 9. Sign 'C': Hand curved into a distinct C shape (with false-positive protection)
        # Prevents triggering on relaxed hand or fist: all 4 fingers must form an open curved cup
        if not i_ext and not m_ext and not r_ext and not p_ext:
            is_open_arc = 0.42 <= d_ti <= 1.05 and p[4]['y'] > p[8]['y'] - 0.15
            # Distinguish from closed fist: fingers must extend outwards in arc, not tightly curled against palm
            not_tight_fist = d_iw >= 0.70 and d_mw >= 0.70 and d_rw >= 0.62 and d_pw >= 0.55
            # Distinguish from O: tips are definitely open and not touching
            not_touching_o = d_ti > 0.40 and scores['O'] == 0.0
            # Ring and pinky curve along with middle finger
            ring_pinky_arched = d_mr <= 0.45 and d_rp <= 0.45

            if is_open_arc and not_tight_fist and not_touching_o and ring_pinky_arched:
                scores['C'] = 0.88

        # 10. Sign 'S': All 5 fingers folded into a fist with thumb crossed horizontally across fingers
        if not i_ext and not m_ext and not r_ext and not p_ext and not t_ext:
            is_fist_curled = p[6]['y'] >= p[5]['y'] - 0.08 and p[8]['y'] >= p[6]['y'] * 0.9
            is_thumb_crossed = p[4]['y'] >= p[5]['y'] - 0.12 and (
                p[4]['x'] >= p[5]['x'] - 0.18
                or self._dist(p[4], p[6]) < 0.48
                or self._dist(p[4], p[9]) < 0.58
            )
            is_compact_fist = d_iw < 1.08 and d_mw < 1.08
            not_o = scores['O'] == 0.0
            if is_fist_curled and is_thumb_crossed and is_compact_fist and not_o:
                scores['S'] = max(scores['S'], 0.92)

        # 11. Sign 'A': Tight fist with thumb straight up alongside index finger (lateral side, not crossed)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            is_thumb_upright = p[4]['y'] < p[5]['y'] - 0.02
            is_thumb_alongside = p[4]['x'] < p[5]['x'] + 0.12 and self._dist(p[4], p[9]) > 0.40
            if is_thumb_upright and is_thumb_alongside and d_ti > 0.30:
                scores['A'] = max(scores['A'], 0.90)

        # 12. Sign 'T': Fist with 4 fingers folded completely down, thumb placed specifically between index and middle fingers
        if not i_ext and not m_ext and not r_ext and not p_ext:
            not_o = scores['O'] == 0.0
            is_four_folded = d_iw < 1.15 and d_mw < 1.15 and d_rw < 1.15 and d_pw < 1.15
            mid_im_mcp = {'x': (p[5]['x'] + p[9]['x']) / 2, 'y': (p[5]['y'] + p[9]['y']) / 2}
            mid_im_pip = {'x': (p[6]['x'] + p[10]['x']) / 2, 'y': (p[6]['y'] + p[10]['y']) / 2}
            d_thumb_mid_im = min(self._dist(p[4], mid_im_mcp), self._dist(p[4], mid_im_pip))

            min_x = min(p[5]['x'], p[6]['x']) - 0.06
            max_x = max(p[9]['x'], p[10]['x']) + 0.06
            is_between_im_x = p[4]['x'] >= min_x and p[4]['x'] <= max_x
            is_closer_to_im = self._dist(p[4], p[5]) < self._dist(p[4], p[13]) and self._dist(p[4], p[9]) < self._dist(p[4], p[17])

            if not_o and is_four_folded and is_between_im_x and d_thumb_mid_im <= 0.36 and is_closer_to_im:
                scores['T'] = 0.94
                scores['S'] = 0.0
                scores['A'] = 0.0

        # 13. Sign 'N': Fist with 4 fingers folded completely down, thumb placed specifically between middle and ring fingers
        if not i_ext and not m_ext and not r_ext and not p_ext:
            not_o = scores['O'] == 0.0
            is_four_folded = d_iw < 1.15 and d_mw < 1.15 and d_rw < 1.15 and d_pw < 1.15
            mid_mr_mcp = {'x': (p[9]['x'] + p[13]['x']) / 2, 'y': (p[9]['y'] + p[13]['y']) / 2}
            mid_mr_pip = {'x': (p[10]['x'] + p[14]['x']) / 2, 'y': (p[10]['y'] + p[14]['y']) / 2}
            d_thumb_mid_mr = min(self._dist(p[4], mid_mr_mcp), self._dist(p[4], mid_mr_pip))

            min_x = min(p[9]['x'], p[10]['x']) - 0.06
            max_x = max(p[13]['x'], p[14]['x']) + 0.06
            is_between_mr_x = p[4]['x'] >= min_x and p[4]['x'] <= max_x
            is_closer_to_mr = self._dist(p[4], p[13]) <= self._dist(p[4], p[5]) and self._dist(p[4], p[9]) < self._dist(p[4], p[17])

            if not_o and is_four_folded and is_between_mr_x and d_thumb_mid_mr <= 0.36 and is_closer_to_mr:
                scores['N'] = 0.93
                scores['S'] = 0.0
                scores['A'] = 0.0
                scores['T'] = 0.0

        # 14. Sign 'M': Fist with 4 fingers folded completely down, thumb placed specifically between ring and pinky fingers
        if not i_ext and not m_ext and not r_ext and not p_ext:
            not_o = scores['O'] == 0.0
            is_four_folded = d_iw < 1.15 and d_mw < 1.15 and d_rw < 1.15 and d_pw < 1.15
            mid_rp_mcp = {'x': (p[13]['x'] + p[17]['x']) / 2, 'y': (p[13]['y'] + p[17]['y']) / 2}
            mid_rp_pip = {'x': (p[14]['x'] + p[18]['x']) / 2, 'y': (p[14]['y'] + p[18]['y']) / 2}
            d_thumb_mid_rp = min(self._dist(p[4], mid_rp_mcp), self._dist(p[4], mid_rp_pip))

            min_x = min(p[13]['x'], p[14]['x']) - 0.06
            max_x = max(p[17]['x'], p[18]['x']) + 0.06
            is_between_rp_x = p[4]['x'] >= min_x and p[4]['x'] <= max_x
            is_closer_to_rp = self._dist(p[4], p[17]) < self._dist(p[4], p[9]) and self._dist(p[4], p[13]) < self._dist(p[4], p[5])

            if not_o and is_four_folded and is_between_rp_x and d_thumb_mid_rp <= 0.36 and is_closer_to_rp:
                scores['M'] = 0.93
                scores['S'] = 0.0
                scores['A'] = 0.0
                scores['T'] = 0.0
                scores['N'] = 0.0

        # 5-Finger Open Hand check (for Space gesture)
        is_open_palm_five = bool(t_ext and i_ext and m_ext and r_ext and p_ext)
        if is_open_palm_five:
            scores['B'] = 0.0
            return "SPACE", 0.95

        # Best letter selection
        best_letter = max(scores, key=scores.get)
        best_score = scores[best_letter]

        if best_score >= 0.70:
            return best_letter, round(best_score, 2)
        return None, 0.0


# ---------------------------------------------------------
# DEBOUNCE BUFFER (1.8s SPACE HOLD)
# ---------------------------------------------------------
class DebounceBuffer:
    def __init__(
        self,
        required_frames: int = MIN_CONFIRM_FRAMES,
        min_conf: float = MIN_CONFIDENCE_THRESHOLD,
        cooldown: float = COOLDOWN_SECONDS,
        space_hold_time: float = SPACE_HOLD_SECONDS
    ):
        self.required_frames = required_frames
        self.min_conf = min_conf
        self.cooldown = cooldown
        self.space_hold_time = space_hold_time
        self.history = deque(maxlen=required_frames)
        self.last_confirmed_time = 0.0
        self.confirmed_chars: List[str] = []

        # 5-Finger Space Tracking
        self.space_start_time = 0.0
        self.space_confirmed = False
        self.space_progress = 0.0
        self.space_remaining = 0.0

    def update(self, detected_char: Optional[str], confidence: float) -> Optional[str]:
        now = time.time()

        # Check for 5-finger space gesture (Hold 1.8 seconds)
        if detected_char == "SPACE":
            self.history.clear()
            if not self.space_confirmed:
                if self.space_start_time == 0.0:
                    self.space_start_time = now
                elapsed = now - self.space_start_time
                self.space_progress = min(1.0, elapsed / self.space_hold_time)
                self.space_remaining = max(0.0, self.space_hold_time - elapsed)
                if elapsed >= self.space_hold_time:
                    if not self.confirmed_chars or self.confirmed_chars[-1] != " ":
                        self.confirmed_chars.append(" ")
                    self.space_confirmed = True
                    self.space_progress = 1.0
                    self.space_remaining = 0.0
                    return " "
            return None
        else:
            self.space_start_time = 0.0
            self.space_confirmed = False
            self.space_progress = 0.0
            self.space_remaining = 0.0

        if now - self.last_confirmed_time < self.cooldown:
            self.history.clear()
            return None

        if detected_char and detected_char != "SPACE" and confidence >= self.min_conf:
            self.history.append(detected_char)
        else:
            self.history.clear()
            return None

        if len(self.history) == self.required_frames:
            if len(set(self.history)) == 1:
                confirmed = self.history[0]
                self.confirmed_chars.append(confirmed)
                self.last_confirmed_time = now
                self.history.clear()
                return confirmed
        return None

    def get_progress(self) -> float:
        if self.space_progress > 0:
            return self.space_progress
        if len(self.history) == 0:
            return 0.0
        return len(self.history) / float(self.required_frames)

    def get_cooldown_remaining(self) -> float:
        rem = self.cooldown - (time.time() - self.last_confirmed_time)
        return max(0.0, rem)

    def clear(self):
        self.confirmed_chars.clear()
        self.history.clear()

    def remove_last(self):
        if self.confirmed_chars:
            self.confirmed_chars.pop()


# ---------------------------------------------------------
# THREAD-SAFE NON-BLOCKING TTS (QUEUE + DEDICATED WORKER)
# ---------------------------------------------------------
class NonBlockingTTS:
    """
    Non-blocking Thread-safe Text-to-Speech Engine:
    Initializes pyttsx3 once in a dedicated long-running background worker thread
    with a FIFO queue to prevent COM threading errors on Windows/macOS/Linux.
    """
    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self.is_speaking = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        engine = None
        if HAS_PYTTSX3:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
            except Exception as e:
                print(f"Error initializing pyttsx3 engine: {e}")
                engine = None

        while True:
            item = self.queue.get()
            if item is None:
                break
            text, lang = item
            if not text.strip():
                self.queue.task_done()
                continue

            self.is_speaking = True
            try:
                if engine is not None:
                    engine.say(text)
                    engine.runAndWait()
                else:
                    print(f"\n[TTS PLAYING ({lang.upper()})]: \"{text}\"\n")
                    time.sleep(1.2)
            except Exception as ex:
                print(f"TTS Speech error: {ex}")
            finally:
                self.is_speaking = False
                self.queue.task_done()

    def speak(self, text: str, lang: str = "en"):
        if not text.strip():
            return
        self.queue.put((text, lang))

    def stop(self):
        self.queue.put(None)


# ---------------------------------------------------------
# BILINGUAL AI SENTENCE BUILDER
# ---------------------------------------------------------
class AISentenceBuilder:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.client = None
        self.language = "vi"  # Default 'vi' (Vietnamese) or 'en' (English)
        api_key = os.environ.get("GEMINI_API_KEY")

        if HAS_GOOGLE_GENAI and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Error initializing Google GenAI Client: {e}")
        elif not api_key:
            print("Notice: GEMINI_API_KEY environment variable not set. Using offline fallback dictionary.")

        # Offline fallback dictionary (English)
        self.dict_en = {
            "HI": "Hi there!",
            "HELLO": "Hello, how are you?",
            "ILY": "I love you.",
            "I L Y": "I love you.",
            "CALL": "Please give me a call.",
            "BOY": "That is a boy.",
            "CAR": "I see a car.",
            "CAT": "The cat is playing.",
            "CHILL": "Let us chill and relax.",
            "DAY": "Have a wonderful day!",
            "DOG": "The dog is happy.",
            "FOOD": "The food is delicious.",
            "FUN": "We are having great fun!",
            "GO": "Let us go together.",
            "GOOD": "Everything is good.",
            "HOT": "The weather is very hot.",
            "KID": "The kid is smiling happily.",
            "LOVE": "Spread love and positivity.",
            "MOM": "I love my mom dearly.",
            "MAN": "The man is working hard.",
            "NO": "No, thank you.",
            "NOW": "Let us start right now.",
            "OH": "Oh, I see!",
            "OK": "Everything is okay.",
            "RAIN": "The rain is falling softly.",
            "ROOM": "Please come into the room.",
            "RUN": "Let us go for a morning run.",
            "SUN": "The sun is shining brightly.",
            "THANK": "Thank you very much!",
            "TIME": "Time is precious.",
            "WATER": "Please drink some water.",
            "WE": "We are here together.",
            "WIN": "We win the game!",
            "WOW": "Wow, that is amazing!",
            "YES": "Yes, of course!",
            "YOU": "Nice to meet you!",
            "BYE": "Goodbye, see you soon!",
            "A": "Letter A.",
            "B": "Letter B.",
            "C": "Letter C.",
            "D": "Letter D.",
            "F": "Letter F.",
            "G": "Letter G.",
            "H": "Letter H.",
            "I": "Letter I.",
            "K": "Letter K.",
            "L": "Letter L.",
            "M": "Letter M.",
            "N": "Letter N.",
            "O": "Letter O.",
            "R": "Letter R.",
            "S": "Letter S.",
            "T": "Letter T.",
            "U": "Letter U.",
            "V": "Letter V.",
            "W": "Letter W.",
            "X": "Letter X.",
            "Y": "Letter Y."
        }

        # Offline fallback dictionary (Vietnamese)
        self.dict_vi = {
            "HI": "Xin chào bạn!",
            "HELLO": "Xin chào, bạn khỏe không?",
            "ILY": "Tôi yêu bạn.",
            "I L Y": "Tôi yêu bạn.",
            "CALL": "Hãy gọi cho tôi nhé.",
            "BOY": "Đó là một cậu bé.",
            "CAR": "Tôi thấy một chiếc xe hơi.",
            "CAT": "Con mèo này rất dễ thương.",
            "CHILL": "Hãy cùng thư giãn nào.",
            "DAY": "Chúc bạn một ngày tuyệt vời!",
            "DOG": "Chú chó đang chơi đùa.",
            "FOOD": "Món ăn hôm nay rất ngon.",
            "FUN": "Chúng tôi đang có rất nhiều niềm vui!",
            "GO": "Chúng ta cùng đi thôi.",
            "GOOD": "Mọi việc đều rất tốt đẹp.",
            "HOT": "Thời tiết hôm nay khá nóng.",
            "KID": "Đứa trẻ đang cười rất tươi.",
            "LOVE": "Lan tỏa tình yêu thương.",
            "MOM": "Tôi rất yêu mẹ của mình.",
            "MAN": "Người đàn ông đang chăm chỉ làm việc.",
            "NO": "Không, cảm ơn bạn.",
            "NOW": "Chúng ta hãy bắt đầu ngay bây giờ.",
            "OH": "Ồ, tôi hiểu rồi!",
            "OK": "Mọi việc đều ổn thỏa.",
            "RAIN": "Trời đang đổ cơn mưa nhẹ.",
            "ROOM": "Mời bạn bước vào trong phòng.",
            "RUN": "Cùng nhau chạy bộ buổi sáng nào.",
            "SUN": "Ánh mặt trời đang chiếu sáng rực rỡ.",
            "THANK": "Cảm ơn bạn rất nhiều!",
            "TIME": "Thời gian là điều vô giá.",
            "WATER": "Uống một chút nước mát nhé.",
            "WE": "Chúng ta đang ở bên nhau.",
            "WIN": "Chúng ta đã chiến thắng!",
            "WOW": "Ồ, thật là tuyệt vời!",
            "YES": "Vâng, chắc chắn rồi!",
            "YOU": "Rất vui được gặp bạn!",
            "BYE": "Tạm biệt, hẹn gặp lại bạn nhé!",
            "A": "Chữ cái A.",
            "B": "Chữ cái B.",
            "C": "Chữ cái C.",
            "D": "Chữ cái D.",
            "F": "Chữ cái F.",
            "G": "Chữ cái G.",
            "H": "Chữ cái H.",
            "I": "Chữ cái I.",
            "K": "Chữ cái K.",
            "L": "Chữ cái L.",
            "M": "Chữ cái M.",
            "N": "Chữ cái N.",
            "O": "Chữ cái O.",
            "R": "Chữ cái R.",
            "S": "Chữ cái S.",
            "T": "Chữ cái T.",
            "U": "Chữ cái U.",
            "V": "Chữ cái V.",
            "W": "Chữ cái W.",
            "X": "Chữ cái X.",
            "Y": "Chữ cái Y."
        }

    def toggle_language(self) -> str:
        self.language = "en" if self.language == "vi" else "vi"
        return self.language

    def build_sentence(self, char_list: List[str]) -> str:
        raw_text = "".join(char_list).strip()
        if not raw_text:
            return ""

        clean_upper = raw_text.replace(" ", "").upper()
        if clean_upper == "ILY":
            return "Tôi yêu bạn." if self.language == "vi" else "I love you."

        if self.client:
            if self.language == "vi":
                prompt = (
                    "Bạn là trợ lý dịch ngôn ngữ ký hiệu ASL (American Sign Language). "
                    "Hãy chuyển đổi các ký tự đánh vần sau thành một câu tiếng Việt hoàn chỉnh, tự nhiên và đúng ngữ cảnh. "
                    "Quy tắc đặc biệt: 'ILY' dịch là 'Tôi yêu bạn'. "
                    f"Chỉ trả về DUY NHẤT câu hoàn chỉnh, không có dấu ngoặc kép hay giải thích:\n{raw_text}"
                )
            else:
                prompt = (
                    "You are an expert ASL (American Sign Language) translation assistant. "
                    "Convert the following fingerspelled characters into a complete, polite, natural English sentence. "
                    "Special rule: 'ILY' means 'I love you.' "
                    f"Return ONLY the final sentence with no quotes and no explanations:\n{raw_text}"
                )

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                text = response.text.strip().strip('"\'')
                return text
            except Exception as e:
                print(f"AI API error: {e}. Using offline fallback dictionary.")

        # Offline fallback dictionary
        active_dict = self.dict_vi if self.language == "vi" else self.dict_en
        upper = raw_text.upper()
        if upper in active_dict:
            return active_dict[upper]

        # Space-separated lookup
        parts = [active_dict.get(word, word) for word in upper.split()]
        if any(w in active_dict for w in upper.split()):
            return " ".join(parts)

        return f"{raw_text.capitalize()}."


# ---------------------------------------------------------
# MINIMALIST DARK HUD INTERFACE (OPENCV DRAWING)
# ---------------------------------------------------------
def draw_ui(
    frame: Any,
    current_char: Optional[str],
    confidence: float,
    hand_label: str,
    buffer_chars: List[str],
    debounce_progress: float,
    cooldown_rem: float,
    final_sentence: str,
    is_speaking: bool,
    is_ai_loading: bool,
    language: str,
) -> Any:
    if cv2 is None or frame is None:
        return frame

    h, w, _ = frame.shape

    COLOR_CYAN = (255, 240, 0)      # BGR for #00f0ff (Cyan neon)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (140, 140, 140)
    COLOR_DARK_CARD = (20, 20, 20)
    COLOR_GREEN = (60, 230, 110)

    # 1. Subtle dark contrast overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.22, frame, 0.78, 0)

    # 2. Outer border
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (45, 45, 45), 1)

    # Header Bar
    cv2.putText(frame, "ASL RECOGNITION", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.putText(frame, "Real-time American Sign Language Recognition", (30, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1, cv2.LINE_AA)

    # Language & Hand Badges
    badge_x = w - 160
    cv2.rectangle(frame, (badge_x, 22), (w - 30, 52), (35, 35, 35), -1)
    cv2.rectangle(frame, (badge_x, 22), (w - 30, 52), COLOR_CYAN, 1)
    lang_text = f"LANG: {language.upper()}"
    cv2.putText(frame, lang_text, (badge_x + 12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_CYAN, 2, cv2.LINE_AA)

    # 3. LIVE DETECTION CARD
    card_w, card_h = 240, 220
    card_x, card_y = w - card_w - 25, 75

    card_roi = frame[card_y:card_y + card_h, card_x:card_x + card_w]
    card_bg = np.full_like(card_roi, COLOR_DARK_CARD)
    frame[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(card_roi, 0.25, card_bg, 0.75, 0)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 60), 1)

    cv2.putText(frame, f"LIVE DETECTION ({hand_label.upper()})", (card_x + 16, card_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GRAY, 1, cv2.LINE_AA)

    display_char = current_char if current_char else "-"
    char_color = COLOR_CYAN if (current_char and confidence >= MIN_CONFIDENCE_THRESHOLD) else COLOR_WHITE

    if current_char == "SPACE":
        cv2.putText(frame, "SPACE", (card_x + 35, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GREEN, 3, cv2.LINE_AA)
        cv2.putText(frame, f"HOLD {SPACE_HOLD_SECONDS}s FOR SPACE", (card_x + 16, card_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, display_char, (card_x + 75, card_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 3.2, char_color, 4, cv2.LINE_AA)

    # Confidence bar
    bar_x = card_x + 20
    bar_y = card_y + 155
    bar_w = card_w - 40
    bar_h = 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    fill_w = int(bar_w * min(1.0, confidence))
    if fill_w > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), COLOR_GREEN if current_char == "SPACE" else COLOR_CYAN, -1)

    conf_text = "5-FINGER SPACE" if current_char == "SPACE" else f"CONFIDENCE {int(confidence * 100)}%"
    cv2.putText(frame, conf_text, (bar_x, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

    if debounce_progress > 0:
        hold_w = int(bar_w * debounce_progress)
        cv2.rectangle(frame, (bar_x, bar_y + 35), (bar_x + hold_w, bar_y + 40), COLOR_GREEN, -1)
    elif cooldown_rem > 0:
        cv2.putText(frame, f"WAIT {cooldown_rem:.1f}s", (bar_x + 60, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_CYAN, 1, cv2.LINE_AA)

    # 4. BUFFER TEXT CARD
    buf_box_w = min(w - card_w - 60, 620)
    buf_box_h = 95
    buf_x, buf_y = 25, 75

    buf_roi = frame[buf_y:buf_y + buf_box_h, buf_x:buf_x + buf_box_w]
    buf_bg = np.full_like(buf_roi, COLOR_DARK_CARD)
    frame[buf_y:buf_y + buf_box_h, buf_x:buf_x + buf_box_w] = cv2.addWeighted(buf_roi, 0.25, buf_bg, 0.75, 0)
    cv2.rectangle(frame, (buf_x, buf_y), (buf_x + buf_box_w, buf_y + buf_box_h), (50, 50, 50), 1)

    cv2.putText(frame, "BUFFER TEXT", (buf_x + 18, buf_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    if buffer_chars:
        buf_str = "".join([c if c != " " else " [SPC] " for c in buffer_chars])
        if len(buf_str) > 24:
            buf_str = "..." + buf_str[-24:]
        cv2.putText(frame, f"[ {buf_str} ]", (buf_x + 18, buf_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.15, COLOR_WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "[ BUFFER EMPTY - HOLD SIGN ]", (buf_x + 18, buf_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1, cv2.LINE_AA)

    # 5. AI SENTENCE BUILDER & TTS
    bot_h = 135
    bot_y = h - bot_h - 20
    bot_x = 25
    bot_w = w - 50

    bot_roi = frame[bot_y:bot_y + bot_h, bot_x:bot_x + bot_w]
    bot_bg = np.full_like(bot_roi, (18, 18, 18))
    frame[bot_y:bot_y + bot_h, bot_x:bot_x + bot_w] = cv2.addWeighted(bot_roi, 0.2, bot_bg, 0.8, 0)
    cv2.rectangle(frame, (bot_x, bot_y), (bot_x + bot_w, bot_y + bot_h), (65, 65, 65), 1)

    cv2.putText(frame, "AI SENTENCE BUILDER", (bot_x + 20, bot_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CYAN, 1, cv2.LINE_AA)

    status_x = bot_x + bot_w - 240
    if is_speaking:
        cv2.circle(frame, (status_x, bot_y + 24), 6, COLOR_CYAN, -1)
        cv2.putText(frame, "TTS: SPEAKING...", (status_x + 14, bot_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1, cv2.LINE_AA)
    elif is_ai_loading:
        cv2.circle(frame, (status_x, bot_y + 24), 6, COLOR_GREEN, -1)
        cv2.putText(frame, "SYNTHESIZING...", (status_x + 14, bot_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1, cv2.LINE_AA)

    if final_sentence:
        disp_sentence = f"\"{final_sentence}\""
        frame = draw_text_unicode(
            frame=frame,
            text=disp_sentence,
            pos=(bot_x + 20, bot_y + 82),
            font_size=24,
            color_bgr=COLOR_WHITE,
            thickness=2,
            is_unicode=(language == "vi")
        )
    else:
        hint_text = "Nhấn [SPACE] để ghép câu tiếng Việt hoàn chỉnh" if language == "vi" else "Press [SPACE] to synthesize fluent sentence from buffer"
        frame = draw_text_unicode(
            frame=frame,
            text=hint_text,
            pos=(bot_x + 20, bot_y + 78),
            font_size=18,
            color_bgr=(120, 120, 120),
            thickness=1,
            is_unicode=(language == "vi")
        )

    help_text = "[SPACE]: Synthesize & Speak  |  [L]: Toggle Lang (EN/VI)  |  [C]: Clear  |  [BACKSPACE]: Delete  |  [Q/ESC]: Quit"
    cv2.putText(frame, help_text, (bot_x + 20, bot_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
def main():
    print("==================================================================")
    print("           ASL RECOGNITION - REAL-TIME ASL AI PIPELINE            ")
    print(f"  Signs (21): {CORE_SIGNS}")
    print("  Controls: [SPACE] Synthesize | [L] Toggle Lang | [C] Clear | [Q] Quit")
    print("==================================================================")

    if cv2 is None:
        print("Error: opencv-python is not installed. Please run: pip install opencv-python")
        return
    if mp is None:
        print("Error: mediapipe is not installed. Please run: pip install mediapipe")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access Webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
    except AttributeError:
        import mediapipe.python.solutions.hands as mp_hands
        import mediapipe.python.solutions.drawing_utils as mp_drawing

    hand_connection_style = mp_drawing.DrawingSpec(color=(255, 240, 0), thickness=2, circle_radius=2)
    hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=3)

    classifier = GeometricASLClassifier()
    debounce = DebounceBuffer(
        required_frames=MIN_CONFIRM_FRAMES,
        min_conf=MIN_CONFIDENCE_THRESHOLD,
        cooldown=COOLDOWN_SECONDS,
        space_hold_time=SPACE_HOLD_SECONDS
    )
    tts = NonBlockingTTS()
    ai_builder = AISentenceBuilder()

    final_sentence = ""
    is_ai_loading = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            detected_letter = None
            detected_confidence = 0.0
            detected_hand_label = "Right"

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Handedness Invariance Check
                    is_left_hand = False
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        classification = results.multi_handedness[idx].classification[0]
                        detected_hand_label = classification.label  # "Left" or "Right"
                        is_left_hand = (detected_hand_label == "Left")

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_style,
                        connection_drawing_spec=hand_connection_style
                    )
                    detected_letter, detected_confidence = classifier.predict(
                        hand_landmarks.landmark,
                        is_left_hand=is_left_hand
                    )

            confirmed = debounce.update(detected_letter, detected_confidence)
            if confirmed:
                print(f"[CONFIRMED SIGN]: {confirmed} (Confidence: {detected_confidence*100:.1f}%)")

            key = cv2.waitKey(1) & 0xFF

            # [SPACE]: Synthesize sentence
            if key == ord(' '):
                if debounce.confirmed_chars and not is_ai_loading:
                    is_ai_loading = True
                    chars_to_process = list(debounce.confirmed_chars)
                    current_lang = ai_builder.language

                    def _run_synthesis(chars, lang):
                        nonlocal final_sentence, is_ai_loading
                        res = ai_builder.build_sentence(chars)
                        final_sentence = res
                        is_ai_loading = False
                        tts.speak(res, lang=lang)

                    threading.Thread(target=_run_synthesis, args=(chars_to_process, current_lang), daemon=True).start()

            # [L]: Toggle language (English / Vietnamese)
            elif key == ord('l') or key == ord('L'):
                new_lang = ai_builder.toggle_language()
                print(f"[LANGUAGE TOGGLED]: {new_lang.upper()}")

            # [C]: Clear buffer
            elif key == ord('c') or key == ord('C'):
                debounce.clear()
                final_sentence = ""

            # [BACKSPACE]: Delete last confirmed character
            elif key == 8:
                debounce.remove_last()

            # [Q] or [ESC]: Quit
            elif key == ord('q') or key == ord('Q') or key == 27:
                break

            display_frame = draw_ui(
                frame=frame,
                current_char=detected_letter,
                confidence=detected_confidence,
                hand_label=detected_hand_label,
                buffer_chars=debounce.confirmed_chars,
                debounce_progress=debounce.get_progress(),
                cooldown_rem=debounce.get_cooldown_remaining(),
                final_sentence=final_sentence,
                is_speaking=tts.is_speaking,
                is_ai_loading=is_ai_loading,
                language=ai_builder.language,
            )

            cv2.imshow("ASL Recognition - Real-time American Sign Language Recognition", display_frame)

    tts.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
