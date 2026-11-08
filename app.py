"""
ASL SignLens - Realtime ASL AI Pipeline
Pipeline:
  [Webcam Live]
    └──> [MediaPipe Hand Tracking] (21 Landmarks)
          └──> [Geometric Invariant Classifier] (Core Signs: A, B, C, H, I, L, O, V, Y)
                └──> [Debounce Buffer] (10-15 frames > 80% + 1.0s Cooldown)
                      └──> [LLM API - Gemini Flash] (Fluent English Sentence Builder)
                            └──> [Text-to-Speech (TTS)] (Non-blocking speech audio)

Installation:
  pip install -r requirements.txt
Run application:
  python app.py
"""

import os
import sys
import time
import math
import threading
from collections import deque
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

# Verify MediaPipe
try:
    import mediapipe as mp
except ImportError:
    print("Please install mediapipe: pip install mediapipe")
    sys.exit(1)

# Verify TTS (pyttsx3)
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False
    print("Note: pyttsx3 not installed. Audio output will be printed to console.")

# Verify Google GenAI SDK
try:
    from google import genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False
    print("Note: google-genai not installed. Intelligent offline fallback will be used.")

# ---------------------------------------------------------
# SYSTEM CONFIGURATION (CORE 10 SIGNS)
# ---------------------------------------------------------
CORE_SIGNS = ['A', 'B', 'C', 'H', 'I', 'L', 'O', 'S', 'U', 'V', 'W', 'Y']
MIN_CONFIRM_FRAMES = 12       # Require 10 - 15 consecutive frames
MIN_CONFIDENCE_THRESHOLD = 0.80  # Confidence threshold > 80%
COOLDOWN_SECONDS = 1.0        # 1.0s cooldown after confirm
GEMINI_MODEL = "gemini-2.5-flash"

# ---------------------------------------------------------
# GEOMETRIC INVARIANT CLASSIFIER
# ---------------------------------------------------------
class GeometricASLClassifier:
    """
    Geometric Invariant Feature Classifier:
    1. Horizontal mirror sync: cv2.flip(frame, 1).
    2. Drop Z coordinate (pure 2D analysis).
    3. Shift origin: Wrist (0) as (0, 0).
    4. Scale normalization: divided by Wrist (0) to Middle MCP (9) length.
    5. Finger extension and inter-finger distance vectors.
    """
    def __init__(self):
        self.classes = CORE_SIGNS

    @staticmethod
    def _dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def extract_features(self, landmarks_21) -> Optional[Dict[str, Any]]:
        if not landmarks_21 or len(landmarks_21) < 21:
            return None

        # 1. Drop Z, extract 2D (x, y)
        raw_pts = [(lm.x, lm.y) for lm in landmarks_21]

        # 2. Shift origin to Wrist (landmark 0)
        wrist = raw_pts[0]
        rel_pts = [(p[0] - wrist[0], p[1] - wrist[1]) for p in raw_pts]

        # 3. Normalize scale by distance from Wrist (0) to Middle MCP (9)
        ref_scale = self._dist(rel_pts[0], rel_pts[9])
        if ref_scale < 1e-5:
            ref_scale = 1.0

        norm_pts = [(p[0] / ref_scale, p[1] / ref_scale) for p in rel_pts]

        # Distance from fingertips to Wrist (0)
        d_thumb_wrist  = self._dist(norm_pts[4],  norm_pts[0])
        d_index_wrist  = self._dist(norm_pts[8],  norm_pts[0])
        d_middle_wrist = self._dist(norm_pts[12], norm_pts[0])
        d_ring_wrist   = self._dist(norm_pts[16], norm_pts[0])
        d_pinky_wrist  = self._dist(norm_pts[20], norm_pts[0])

        # Finger extension states
        # Lowered threshold from 1.15 down to 0.85 for natural, comfortable thumb extension
        d_thumb_mcp = self._dist(norm_pts[4], norm_pts[2])
        d_thumb_index_mcp = self._dist(norm_pts[4], norm_pts[5])
        thumb_extended = d_thumb_wrist > 0.85 or (d_thumb_wrist > 0.76 and (d_thumb_index_mcp > 0.48 or d_thumb_mcp > 0.40))

        # Finger vectors from MCP to Tip
        dx_index = norm_pts[8][0] - norm_pts[5][0]
        dy_index = norm_pts[8][1] - norm_pts[5][1]
        dx_middle = norm_pts[12][0] - norm_pts[9][0]
        dy_middle = norm_pts[12][1] - norm_pts[9][1]

        # Upright extensions: MUST point UP into the sky (vertical vector: dy < -0.45, |dy| > |dx| * 1.15, tip.y < pip.y)
        is_index_pointing_up = dy_index < -0.45 and abs(dy_index) > abs(dx_index) * 1.15 and norm_pts[8][1] < norm_pts[6][1]
        is_middle_pointing_up = dy_middle < -0.45 and abs(dy_middle) > abs(dx_middle) * 1.15 and norm_pts[12][1] < norm_pts[10][1]

        index_ext_up = is_index_pointing_up and d_index_wrist > 1.20
        middle_ext_up = is_middle_pointing_up and d_middle_wrist > 1.20

        # Sideways extensions: MUST point horizontally across to either side (|dx| > 0.45 and |dx| > |dy| * 1.15)
        is_index_pointing_side = abs(dx_index) > 0.45 and abs(dx_index) > abs(dy_index) * 1.15
        is_middle_pointing_side = abs(dx_middle) > 0.45 and abs(dx_middle) > abs(dy_middle) * 1.15

        index_ext_side = is_index_pointing_side and d_index_wrist > 1.15
        middle_ext_side = is_middle_pointing_side and d_middle_wrist > 1.15

        index_extended  = index_ext_up or index_ext_side
        middle_extended = middle_ext_up or middle_ext_side
        ring_extended   = norm_pts[16][1] < norm_pts[14][1] and d_ring_wrist > 1.15
        pinky_extended  = norm_pts[20][1] < norm_pts[18][1] and d_pinky_wrist > 1.15

        # Inter-finger distances
        d_thumb_index  = self._dist(norm_pts[4], norm_pts[8])
        d_index_middle = self._dist(norm_pts[8], norm_pts[12])
        d_middle_ring  = self._dist(norm_pts[12], norm_pts[16])
        d_ring_pinky   = self._dist(norm_pts[16], norm_pts[20])
        d_thumb_middle = self._dist(norm_pts[4], norm_pts[12])
        d_thumb_pinky  = self._dist(norm_pts[4], norm_pts[20])

        return {
            "norm_pts": norm_pts,
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
        }

    def predict(self, landmarks_21) -> Tuple[Optional[str], float]:
        feats = self.extract_features(landmarks_21)
        if not feats:
            return None, 0.0

        scores: Dict[str, float] = {k: 0.0 for k in self.classes}
        p = feats["norm_pts"]

        d_ti = feats["d_thumb_index"]
        d_im = feats["d_index_middle"]
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

        # 1. Sign 'B': 4 fingers straight up touching, thumb folded across palm
        if i_ext and m_ext and r_ext and p_ext:
            if not t_ext or p[4][0] > p[5][0] - 0.2:
                tightness = 1.0 - min(1.0, (d_im + feats["d_middle_ring"] + feats["d_ring_pinky"]) / 1.5)
                scores['B'] = max(scores['B'], 0.85 + 0.15 * max(0.0, tightness))

        # 2. Sign 'V': Index and middle fingers pointing UP spread in a V, ring and pinky curled
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            if d_im > 0.35:
                scores['V'] = max(scores['V'], 0.88 + 0.12 * min(1.0, (d_im - 0.35) / 0.5))

        # 2b. Sign 'W': Index, middle, and ring fingers extended UP in a W formation, pinky curled down, thumb folded
        # "nhận diện bằng cách giơ 3 ngón trỏ, ngón giữa và ngón nhẫn"
        if i_ext and m_ext and r_ext and not p_ext and not i_ext_side and not m_ext_side:
            is_upright = p[8][1] < p[6][1] and p[12][1] < p[10][1] and p[16][1] < p[14][1]
            if is_upright:
                scores['W'] = max(scores['W'], 0.94)

        # 3. Sign 'U': Index and middle fingers pointing strictly UP together touching, ring and pinky curled down, thumb folded
        # "ngón trỏ và ngón giữa giơ lên giáp với nhau, còn ngón tay út, ngón cái và ngón nhẫn co lại. Hai ngón giữa và trỏ chỉ lên trời."
        # Strictly require pointing UP into the sky, never triggered by horizontal/sideways gestures
        if i_ext_up and m_ext_up and not i_ext_side and not m_ext_side and not r_ext and not p_ext:
            if d_im <= 0.35:
                closeness = 1.0 - (d_im / 0.35)
                scores['U'] = max(scores['U'], 0.90 + 0.10 * max(0.0, closeness))

        # 4. Sign 'H': Index and middle fingers extended together pointing SIDEWAYS ("chỉ qua hai bên vuông góc với mặt đất"), ring and pinky curled
        # Strictly require pointing SIDEWAYS horizontally, never triggered by upright gestures
        if i_ext_side and m_ext_side and not i_ext_up and not m_ext_up and not r_ext and not p_ext:
            if d_im <= 0.38:
                parallelism = 1.0 - (d_im / 0.38)
                scores['H'] = max(scores['H'], 0.88 + 0.12 * max(0.0, parallelism))

        # 5. Sign 'L': Thumb and index finger extended at right angle (~90°), others folded
        if i_ext and not m_ext and not r_ext and not p_ext:
            if d_ti > 0.65 and p[4][0] < p[3][0]:
                scores['L'] = max(scores['L'], 0.90 + 0.10 * min(1.0, d_ti / 1.2))

        # 6. Sign 'I': Only pinky extended straight up, others curled
        if p_ext and not i_ext and not m_ext and not r_ext:
            if not t_ext or d_tp <= 1.25:
                scores['I'] = max(scores['I'], 0.93)

        # 7. Sign 'Y': Thumb and pinky extended wide (Shaka), middle three fingers curled
        if p_ext and t_ext and not i_ext and not m_ext and not r_ext:
            if d_tp > 1.15:
                scores['Y'] = max(scores['Y'], 0.95)

        # 8. Sign 'O': Hand forming circular O-ring (either facing camera or turned sideways)
        # Supports BOTH:
        # - Palm facing towards camera (frontal view where thumb and fingertips curve to meet in front of palm)
        # - Hand turned sideways (profile view showing the open circular hole)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            is_tip_touching = d_ti <= 0.40
            tips_meet_directly = abs(p[8][1] - p[4][1]) <= 0.22 and abs(p[8][0] - p[4][0]) <= 0.28

            # Case A: Hand turned sideways (profile aperture)
            is_sideways_o = (
                is_tip_touching
                and p[6][1] < p[5][1] + 0.08
                and self._dist(p[6], p[2]) >= 0.30
                and self._dist(p[5], p[4]) >= 0.20
            )

            # Case B: Palm facing towards camera (frontal or 3/4 view)
            is_frontal_o = (
                is_tip_touching
                and tips_meet_directly
                and d_tm <= 0.52
                and d_im <= 0.38
                and feats["d_thumb_wrist"] >= 0.65
            )

            if is_sideways_o or is_frontal_o:
                touch_factor = 1.0 - (d_ti / 0.40)
                scores['O'] = max(scores['O'], 0.88 + 0.12 * touch_factor)

        # 9. Sign 'C': Hand curved into a distinct C shape
        if not i_ext and not m_ext:
            if 0.45 <= d_ti <= 0.98 and p[4][1] > p[8][1]:
                scores['C'] = max(scores['C'], 0.87)

        # 10. Sign 'S': All 5 fingers folded into a fist with thumb crossed horizontally across fingers
        # "cả năm ngón tay đều gập lại như hình một nắm đấm và ngón cái bắt chéo qua các ngón kia"
        if not i_ext and not m_ext and not r_ext and not p_ext and not t_ext:
            is_fist_curled = p[6][1] >= p[5][1] - 0.08 and p[8][1] >= p[6][1] * 0.9
            is_thumb_crossed = p[4][1] >= p[5][1] - 0.12 and (p[4][0] >= p[5][0] - 0.18 or self._dist(p[4], p[6]) < 0.48 or self._dist(p[4], p[9]) < 0.58)
            is_compact_fist = feats["d_index_wrist"] < 1.08 and feats["d_middle_wrist"] < 1.08
            not_o = scores['O'] == 0.0
            if is_fist_curled and is_thumb_crossed and is_compact_fist and not_o:
                scores['S'] = max(scores['S'], 0.92)

        # 11. Sign 'A': Tight fist with thumb straight up alongside index finger (lateral side, not crossed)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            is_thumb_upright = p[4][1] < p[5][1] - 0.02
            is_thumb_alongside = p[4][0] < p[5][0] + 0.12 and self._dist(p[4], p[9]) > 0.40
            if is_thumb_upright and is_thumb_alongside and d_ti > 0.30:
                scores['A'] = max(scores['A'], 0.90)

        # 11. 5-Finger Open Hand: space trigger
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
# DEBOUNCE BUFFER
# ---------------------------------------------------------
class DebounceBuffer:
    def __init__(self, required_frames: int = MIN_CONFIRM_FRAMES, min_conf: float = MIN_CONFIDENCE_THRESHOLD, cooldown: float = COOLDOWN_SECONDS):
        self.required_frames = required_frames
        self.min_conf = min_conf
        self.cooldown = cooldown
        self.history = deque(maxlen=required_frames)
        self.last_confirmed_time = 0.0
        self.confirmed_chars: List[str] = []

        # 5-Finger 5-Second Space Tracking
        self.space_start_time = 0.0
        self.space_confirmed = False
        self.space_progress = 0.0
        self.space_remaining = 0.0

    def update(self, detected_char: Optional[str], confidence: float) -> Optional[str]:
        now = time.time()

        # Check for 5-finger space gesture (Hold 5 seconds)
        if detected_char == "SPACE":
            self.history.clear()
            if not self.space_confirmed:
                if self.space_start_time == 0.0:
                    self.space_start_time = now
                elapsed = now - self.space_start_time
                self.space_progress = min(1.0, elapsed / 5.0)
                self.space_remaining = max(0.0, 5.0 - elapsed)
                if elapsed >= 5.0:
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
# NON-BLOCKING TTS
# ---------------------------------------------------------
class NonBlockingTTS:
    def __init__(self):
        self.is_speaking = False
        self._lock = threading.Lock()
        if HAS_PYTTSX3:
            try:
                eng = pyttsx3.init()
                eng.setProperty('rate', 150)
                eng.stop()
            except Exception as e:
                print(f"Error initializing pyttsx3: {e}")

    def speak(self, text: str):
        if not text.strip():
            return

        def _worker(msg: str):
            with self._lock:
                self.is_speaking = True
            try:
                if HAS_PYTTSX3:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(msg)
                    engine.runAndWait()
                else:
                    print(f"\n[TTS PLAYING]: \"{msg}\"\n")
                    time.sleep(1.2)
            except Exception as ex:
                print(f"TTS Speech error: {ex}")
            finally:
                with self._lock:
                    self.is_speaking = False

        thread = threading.Thread(target=_worker, args=(text,), daemon=True)
        thread.start()


# ---------------------------------------------------------
# GEMINI FLASH LLM BUILDER
# ---------------------------------------------------------
class GeminiSentenceBuilder:
    def __init__(self, model_name: str = GEMINI_MODEL):
        self.model_name = model_name
        self.client = None
        api_key = os.environ.get("GEMINI_API_KEY")

        if HAS_GOOGLE_GENAI and api_key:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"Error initializing Google GenAI Client: {e}")
        elif not api_key:
            print("Notice: GEMINI_API_KEY environment variable not set. Using offline fallback dictionary.")

    def build_sentence(self, char_list: List[str]) -> str:
        raw_text = "".join(char_list).strip()
        if not raw_text:
            return ""

        clean_upper = raw_text.replace(" ", "").upper()
        if clean_upper == "ILY":
            return "I love you."

        if self.client:
            prompt = (
                f"You are an expert ASL (American Sign Language) translation assistant. "
                f"Convert the following fingerspelled characters into a complete, polite, natural English sentence. "
                f"Special rule: 'ILY' means 'I love you.' "
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
                print(f"Gemini API error: {e}. Using fallback.")

        # Offline fallback dictionary
        known_words = {
            "HI": "Hi there!",
            "ILY": "I love you.",
            "I L Y": "I love you.",
            "CALL": "Please give me a call.",
            "BOY": "That is a boy.",
            "CAR": "I see a car.",
            "CHILL": "Let us chill and relax.",
            "OH": "Oh, I see!",
            "OK": "Everything is okay.",
            "BYE": "Goodbye, see you soon!",
            "A": "Letter A.",
            "B": "Letter B.",
            "C": "Letter C.",
            "H": "Letter H.",
            "I": "Letter I.",
            "L": "Letter L.",
            "O": "Letter O.",
            "U": "Letter U.",
            "YOU": "Nice to see you!",
            "BUY": "I would like to buy this.",
            "CLUB": "Welcome to the club!",
            "V": "Letter V.",
            "W": "Letter W.",
            "WE": "We are here.",
            "WIN": "We win!",
            "WOW": "Wow, that is amazing!",
            "Y": "Letter Y."
        }
        upper = raw_text.upper()
        if upper in known_words:
            return known_words[upper]

        return f"{raw_text.capitalize()}."


# ---------------------------------------------------------
# MINIMALIST DARK MODE UI (OPENCV DRAWING)
# ---------------------------------------------------------
def draw_ui(
    frame: np.ndarray,
    current_char: Optional[str],
    confidence: float,
    buffer_chars: List[str],
    debounce_progress: float,
    cooldown_rem: float,
    final_sentence: str,
    is_speaking: bool,
    is_gemini_loading: bool,
) -> np.ndarray:
    h, w, _ = frame.shape

    COLOR_CYAN = (255, 240, 0)      # BGR for #00f0ff (Cyan neon)
    COLOR_WHITE = (255, 255, 255)
    COLOR_GRAY = (120, 120, 120)
    COLOR_DARK_CARD = (22, 22, 22)
    COLOR_GREEN = (50, 220, 100)

    # 1. Subtle dark contrast overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # 2. Outer border
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (45, 45, 45), 1)

    # 3. LIVE DETECTION CARD
    card_w, card_h = 240, 220
    card_x, card_y = w - card_w - 25, 25

    card_roi = frame[card_y:card_y + card_h, card_x:card_x + card_w]
    card_bg = np.full_like(card_roi, COLOR_DARK_CARD)
    frame[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(card_roi, 0.3, card_bg, 0.7, 0)
    cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 60), 1)

    cv2.putText(frame, "LIVE DETECTION", (card_x + 18, card_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GRAY, 1, cv2.LINE_AA)

    display_char = current_char if current_char else "-"
    char_color = COLOR_CYAN if (current_char and confidence >= MIN_CONFIDENCE_THRESHOLD) else COLOR_WHITE

    if current_char == "SPACE":
        cv2.putText(frame, "SPACE", (card_x + 35, card_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_GREEN, 3, cv2.LINE_AA)
        cv2.putText(frame, "HOLD 5s FOR SPACE", (card_x + 20, card_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GREEN, 1, cv2.LINE_AA)
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

    # 4. BUFFER TEXT
    buf_box_w = min(w - card_w - 60, 600)
    buf_box_h = 100
    buf_x, buf_y = 25, 25

    buf_roi = frame[buf_y:buf_y + buf_box_h, buf_x:buf_x + buf_box_w]
    buf_bg = np.full_like(buf_roi, COLOR_DARK_CARD)
    frame[buf_y:buf_y + buf_box_h, buf_x:buf_x + buf_box_w] = cv2.addWeighted(buf_roi, 0.3, buf_bg, 0.7, 0)
    cv2.rectangle(frame, (buf_x, buf_y), (buf_x + buf_box_w, buf_y + buf_box_h), (50, 50, 50), 1)

    cv2.putText(frame, "BUFFER TEXT", (buf_x + 18, buf_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    if buffer_chars:
        buf_str = "".join([c if c != " " else " [SPC] " for c in buffer_chars])
        if len(buf_str) > 24:
            buf_str = "..." + buf_str[-24:]
        cv2.putText(frame, f"[ {buf_str} ]", (buf_x + 18, buf_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_WHITE, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "[ BUFFER EMPTY - HOLD SIGN ]", (buf_x + 18, buf_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1, cv2.LINE_AA)

    # 5. FINAL SENTENCE & TTS
    bot_h = 130
    bot_y = h - bot_h - 20
    bot_x = 25
    bot_w = w - 50

    bot_roi = frame[bot_y:bot_y + bot_h, bot_x:bot_x + bot_w]
    bot_bg = np.full_like(bot_roi, (15, 15, 15))
    frame[bot_y:bot_y + bot_h, bot_x:bot_x + bot_w] = cv2.addWeighted(bot_roi, 0.2, bot_bg, 0.8, 0)
    cv2.rectangle(frame, (bot_x, bot_y), (bot_x + bot_w, bot_y + bot_h), (70, 70, 70), 1)

    cv2.putText(frame, "FINAL SENTENCE (GEMINI FLASH)", (bot_x + 20, bot_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_CYAN, 1, cv2.LINE_AA)

    status_x = bot_x + bot_w - 220
    if is_speaking:
        cv2.circle(frame, (status_x, bot_y + 26), 6, COLOR_CYAN, -1)
        cv2.putText(frame, "TTS: SPEAKING...", (status_x + 14, bot_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_CYAN, 1, cv2.LINE_AA)
    elif is_gemini_loading:
        cv2.circle(frame, (status_x, bot_y + 26), 6, COLOR_GREEN, -1)
        cv2.putText(frame, "SYNTHESIZING...", (status_x + 14, bot_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 1, cv2.LINE_AA)

    if final_sentence:
        disp_sentence = final_sentence if len(final_sentence) < 45 else final_sentence[:42] + "..."
        cv2.putText(frame, f"\"{disp_sentence}\"", (bot_x + 20, bot_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 1.25, COLOR_WHITE, 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, "Press [SPACE] to synthesize fluent English sentence with Gemini Flash", (bot_x + 20, bot_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 120, 120), 1, cv2.LINE_AA)

    help_text = "[SPACE]: Synthesize & Speak  |  [C]: Clear Buffer  |  [BACKSPACE]: Delete Char  |  [Q / ESC]: Quit"
    cv2.putText(frame, help_text, (bot_x + 20, bot_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("=====================================================")
    print("      ASL SIGNLENS - REALTIME ASL AI PIPELINE        ")
    print("  Signs: ['A', 'B', 'C', 'H', 'I', 'L', 'O', 'U', 'V', 'Y'] ")
    print("  Controls: [SPACE] Synthesize | [C] Clear | [Q] Quit ")
    print("=====================================================")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access Webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hand_connection_style = mp_drawing.DrawingSpec(color=(255, 240, 0), thickness=2, circle_radius=2)
    hand_landmark_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=3)

    classifier = GeometricASLClassifier()
    debounce = DebounceBuffer(required_frames=MIN_CONFIRM_FRAMES, min_conf=MIN_CONFIDENCE_THRESHOLD, cooldown=COOLDOWN_SECONDS)
    tts = NonBlockingTTS()
    gemini = GeminiSentenceBuilder()

    final_sentence = ""
    is_gemini_loading = False

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

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_style,
                        connection_drawing_spec=hand_connection_style
                    )
                    detected_letter, detected_confidence = classifier.predict(hand_landmarks.landmark)

            confirmed = debounce.update(detected_letter, detected_confidence)
            if confirmed:
                print(f"[CONFIRMED SIGN]: {confirmed} (Confidence: {detected_confidence*100:.1f}%)")

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if debounce.confirmed_chars and not is_gemini_loading:
                    is_gemini_loading = True
                    chars_to_process = list(debounce.confirmed_chars)

                    def _run_gemini(chars):
                        nonlocal final_sentence, is_gemini_loading
                        res = gemini.build_sentence(chars)
                        final_sentence = res
                        is_gemini_loading = False
                        tts.speak(res)

                    threading.Thread(target=_run_gemini, args=(chars_to_process,), daemon=True).start()

            elif key == ord('c') or key == ord('C'):
                debounce.clear()
                final_sentence = ""

            elif key == 8:
                debounce.remove_last()

            elif key == ord('q') or key == 27:
                break

            display_frame = draw_ui(
                frame=frame,
                current_char=detected_letter,
                confidence=detected_confidence,
                buffer_chars=debounce.confirmed_chars,
                debounce_progress=debounce.get_progress(),
                cooldown_rem=debounce.get_cooldown_remaining(),
                final_sentence=final_sentence,
                is_speaking=tts.is_speaking,
                is_gemini_loading=is_gemini_loading
            )

            cv2.imshow("ASL SignLens - Realtime ASL AI Pipeline", display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
