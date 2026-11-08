/**
 * ASL Invariant Geometric Classifier for 10 Core Signs:
 * ['A', 'B', 'C', 'E', 'H', 'I', 'L', 'O', 'V', 'Y']
 * 
 * Tuân thủ nghiêm ngặt 5 bước chuẩn hóa hình học:
 * 1. Đồng bộ lật ảnh webcam (horizontal mirror).
 * 2. Bỏ hoàn toàn trục Z (chỉ dùng x, y 2D).
 * 3. Dời gốc tọa độ về Wrist (điểm 0).
 * 4. Chuẩn hóa tỷ lệ: chia cho khoảng cách từ Wrist (0) đến Middle MCP (9).
 * 5. Phân tích vector ngón tay, trạng thái gập/duỗi và khoảng cách liên ngón.
 */

export interface Landmark2D {
  x: number;
  y: number;
}

export interface Landmark3D {
  x: number;
  y: number;
  z?: number;
}

export interface GeometricFeatures {
  normPts: Landmark2D[];
  dThumbWrist: number;
  dIndexWrist: number;
  dMiddleWrist: number;
  dRingWrist: number;
  dPinkyWrist: number;
  dThumbIndex: number;
  dIndexMiddle: number;
  dMiddleRing: number;
  dRingPinky: number;
  dThumbMiddle: number;
  dThumbPinky: number;
  thumbExt: boolean;
  indexExt: boolean;
  middleExt: boolean;
  ringExt: boolean;
  pinkyExt: boolean;
  indexExtUp: boolean;
  middleExtUp: boolean;
  indexExtSide: boolean;
  middleExtSide: boolean;
  isIndexHooked: boolean;
  rawLandmarks: Landmark3D[];
}

export interface PredictionResult {
  letter: string | null;
  confidence: number;
  scores: Record<string, number>;
  features: GeometricFeatures | null;
  fingerStates: {
    thumb: boolean;
    index: boolean;
    middle: boolean;
    ring: boolean;
    pinky: boolean;
  };
  isOpenPalmFive: boolean;
}

export const CORE_SIGNS = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'] as const;
export type CoreSign = typeof CORE_SIGNS[number];

function dist(p1: Landmark2D, p2: Landmark2D): number {
  return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

export function extractGeometricFeatures(rawLandmarks: Landmark3D[]): GeometricFeatures | null {
  if (!rawLandmarks || rawLandmarks.length < 21) {
    return null;
  }

  // 1 & 2. Drop Z axis, extract 2D (x, y)
  const rawPts: Landmark2D[] = rawLandmarks.map((lm) => ({ x: lm.x, y: lm.y }));

  // 3. Translate origin to Wrist (point 0)
  const wrist = rawPts[0];
  const relPts: Landmark2D[] = rawPts.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
  }));

  // 4. Scale invariance: divide by distance from Wrist (0) to Middle MCP (9)
  const refScale = dist(relPts[0], relPts[9]);
  const scale = refScale > 0.0001 ? refScale : 1.0;

  const normPts: Landmark2D[] = relPts.map((p) => ({
    x: p.x / scale,
    y: p.y / scale,
  }));

  const dThumbWrist  = dist(normPts[4],  normPts[0]);
  const dIndexWrist  = dist(normPts[8],  normPts[0]);
  const dMiddleWrist = dist(normPts[12], normPts[0]);
  const dRingWrist   = dist(normPts[16], normPts[0]);
  const dPinkyWrist  = dist(normPts[20], normPts[0]);

  // Extension status
  // Lowered threshold from 1.15 down to 0.85 and added outward extension check from MCP
  // so thumb extension is recognized smoothly and comfortably without requiring extreme hyperextension
  const dThumbMcp = dist(normPts[4], normPts[2]);
  const dThumbIndexMcp = dist(normPts[4], normPts[5]);
  const thumbExt = dThumbWrist > 0.85 || (dThumbWrist > 0.76 && (dThumbIndexMcp > 0.48 || dThumbMcp > 0.40));

  // Finger vectors from MCP to Tip
  const dxIndex = normPts[8].x - normPts[5].x;
  const dyIndex = normPts[8].y - normPts[5].y;
  const dxMiddle = normPts[12].x - normPts[9].x;
  const dyMiddle = normPts[12].y - normPts[9].y;

  // Upright extensions: MUST point UP into the sky (vertical vector: dy < -0.45, |dy| > |dx| * 1.15, tip.y < pip.y)
  const isIndexPointingUp = dyIndex < -0.45 && Math.abs(dyIndex) > Math.abs(dxIndex) * 1.15 && normPts[8].y < normPts[6].y;
  const isMiddlePointingUp = dyMiddle < -0.45 && Math.abs(dyMiddle) > Math.abs(dxMiddle) * 1.15 && normPts[12].y < normPts[10].y;

  const indexExtUp = isIndexPointingUp && dIndexWrist > 1.20;
  const middleExtUp = isMiddlePointingUp && dMiddleWrist > 1.20;

  // Sideways extensions: MUST point horizontally across to either side (|dx| > 0.45 and |dx| > |dy| * 1.15)
  const isIndexPointingSide = Math.abs(dxIndex) > 0.45 && Math.abs(dxIndex) > Math.abs(dyIndex) * 1.15;
  const isMiddlePointingSide = Math.abs(dxMiddle) > 0.45 && Math.abs(dxMiddle) > Math.abs(dyMiddle) * 1.15;

  const indexExtSide = isIndexPointingSide && dIndexWrist > 1.15;
  const middleExtSide = isMiddlePointingSide && dMiddleWrist > 1.15;

  const indexExt  = indexExtUp || indexExtSide;
  const middleExt = middleExtUp || middleExtSide;
  const ringExt   = normPts[16].y < normPts[14].y && dRingWrist > 1.15;
  const pinkyExt  = normPts[20].y < normPts[18].y && dPinkyWrist > 1.15;

  // Upright hook (X sign):
  // "ngón trỏ duỗi nửa dưới, nửa trên gập xuống như hình một cái móc"
  // Lower half (MCP 5 -> PIP 6): extends upright (p[6].y < p[5].y - 0.12, pointing up)
  // Upper half (PIP 6 -> DIP 7 -> Tip 8): curls/bends downwards (p[8].y > p[6].y - 0.08, tip curls down)
  const isIndexLowerExtended = normPts[6].y < normPts[5].y - 0.12 && Math.abs(normPts[6].x - normPts[5].x) < Math.abs(normPts[6].y - normPts[5].y) * 1.5;
  const isIndexUpperBentHook = (normPts[8].y >= normPts[6].y - 0.08 || normPts[7].y >= normPts[6].y - 0.04 || dist(normPts[8], normPts[5]) < dist(normPts[6], normPts[5]) * 1.45);
  const isIndexHooked = isIndexLowerExtended && isIndexUpperBentHook && !indexExtUp && !indexExtSide;

  // Inter-finger distances
  const dThumbIndex  = dist(normPts[4], normPts[8]);
  const dIndexMiddle = dist(normPts[8], normPts[12]);
  const dMiddleRing  = dist(normPts[12], normPts[16]);
  const dRingPinky   = dist(normPts[16], normPts[20]);
  const dThumbMiddle = dist(normPts[4], normPts[12]);
  const dThumbPinky  = dist(normPts[4], normPts[20]);

  return {
    normPts,
    dThumbWrist,
    dIndexWrist,
    dMiddleWrist,
    dRingWrist,
    dPinkyWrist,
    dThumbIndex,
    dIndexMiddle,
    dMiddleRing,
    dRingPinky,
    dThumbMiddle,
    dThumbPinky,
    thumbExt,
    indexExt,
    middleExt,
    ringExt,
    pinkyExt,
    indexExtUp,
    middleExtUp,
    indexExtSide,
    middleExtSide,
    isIndexHooked,
    rawLandmarks,
  };
}

export function classifyASL(landmarks: Landmark3D[]): PredictionResult {
  const emptyScores: Record<string, number> = {
    A: 0, B: 0, C: 0, D: 0, F: 0, G: 0, H: 0, I: 0, K: 0, L: 0, M: 0, N: 0, O: 0, R: 0, S: 0, T: 0, U: 0, V: 0, W: 0, X: 0, Y: 0,
  };

  const feats = extractGeometricFeatures(landmarks);
  if (!feats) {
    return {
      letter: null,
      confidence: 0,
      scores: emptyScores,
      features: null,
      fingerStates: { thumb: false, index: false, middle: false, ring: false, pinky: false },
      isOpenPalmFive: false,
    };
  }

  const scores = { ...emptyScores };
  const p = feats.normPts;
  const {
    dThumbWrist,
    dIndexWrist,
    dMiddleWrist,
    dRingWrist,
    dPinkyWrist,
    dThumbIndex,
    dIndexMiddle,
    dMiddleRing,
    dRingPinky,
    dThumbPinky,
    dThumbMiddle,
    thumbExt,
    indexExt,
    middleExt,
    ringExt,
    pinkyExt,
    indexExtUp,
    middleExtUp,
    indexExtSide,
    middleExtSide,
    isIndexHooked,
  } = feats;

  // 1. Sign 'B': 4 fingers straight up touching, thumb folded across palm
  if (indexExt && middleExt && ringExt && pinkyExt) {
    const fingerSpread = dIndexMiddle + dMiddleRing + dRingPinky;
    const isClosed = fingerSpread < 1.35;
    const thumbFolded = !thumbExt || p[4].x > p[5].x - 0.25;
    if (isClosed && thumbFolded) {
      scores.B = 0.88 + 0.12 * Math.max(0, 1.0 - fingerSpread / 1.35);
    }
  }

  // 1b. Sign 'D': Index finger pointing straight UP to the sky, 3 fingers (middle, ring, pinky) folded down, thumb touching near middle finger
  // "lòng bàn tay hướng đến phía màn hình, 3 ngón út, nhẫn và giữa gập xuống. Còn ngón trỏ chỉ lên trời, ngón cái chạm gần ngón giữa."
  if (indexExtUp && !middleExt && !ringExt && !pinkyExt) {
    const dThumbMiddleNearest = Math.min(
      dThumbMiddle,
      dist(p[4], p[10]),
      dist(p[4], p[11]),
      dist(p[4], p[12])
    );
    // Thumb tip rests near middle finger (pad or joints)
    const isThumbNearMiddle = dThumbMiddleNearest <= 0.48;
    if (isThumbNearMiddle) {
      const touchCloseness = Math.max(0, 1.0 - dThumbMiddleNearest / 0.48);
      scores.D = 0.90 + 0.10 * touchCloseness;
    }
  }

  // 1c. Sign 'F': Middle, ring, and pinky fingers extended UP, thumb and index finger folded touching tips
  // "lòng bàn tay hướng về phía màn hình, 3 ngón út, nhẫn và giữa duỗi, ngón cái và ngón trỏ gập lại và chạm vào nhau."
  if (!indexExt && middleExt && ringExt && pinkyExt) {
    const isTipTouching = dThumbIndex <= 0.42;
    const isThreeUpright = p[12].y < p[10].y && p[16].y < p[14].y && p[20].y < p[18].y;
    if (isTipTouching && isThreeUpright) {
      const touchTightness = Math.max(0, 1.0 - dThumbIndex / 0.42);
      scores.F = 0.90 + 0.10 * touchTightness;
    }
  }

  // 1d. Sign 'G': Index finger and thumb extended pointing sideways (to the right/left of the screen), 3 fingers completely folded down
  // "- Ba ngón út, nhẫn và giữa gập xuống hoàn toàn.
  //  - Ngón cái và ngón trỏ duỗi ra và chỉ về phía vuông góc với màn hình, tức là bên phải và trái so với màn hình."
  if (indexExtSide && !indexExtUp && !middleExt && !ringExt && !pinkyExt) {
    const isThreeFoldedCompletely = dMiddleWrist < 1.15 && dRingWrist < 1.15 && dPinkyWrist < 1.15;
    const dThumbMiddleNearest = Math.min(
      dThumbMiddle,
      dist(p[4], p[10]),
      dist(p[4], p[11]),
      dist(p[4], p[12])
    );
    if (isThreeFoldedCompletely && dThumbMiddleNearest > 0.35) {
      scores.G = 0.92;
    }
  }

  // 1e. Sign 'X': Cả 4 ngón út, nhẫn, giữa, cái gập xuống hoàn toàn, ngón trỏ duỗi nửa dưới, nửa trên gập xuống như hình một cái móc.
  // "Sửa lại hoàn toàn chữ X sao cho cả 4 ngón út, nhẫn, giữa, cái gập xuống hoàn toàn, ngón trỏ duỗi nửa dưới, nửa trên gập xuống như hình một cái móc."
  if (isIndexHooked && !middleExt && !ringExt && !pinkyExt) {
    // 3 fingers (middle, ring, pinky) folded completely down
    const isThreeFoldedCompletely = dMiddleWrist < 1.18 && dRingWrist < 1.18 && dPinkyWrist < 1.18;
    // Thumb folded completely down (not extended outward / tucked over or against fingers/palm)
    const isThumbFoldedCompletely = !thumbExt || dThumbWrist < 1.15 || p[4].y >= p[5].y - 0.05;
    // Index finger does not touch thumb tip (distinguishing from O)
    const notTouchingIndexTip = dThumbIndex > 0.38;

    if (isThreeFoldedCompletely && isThumbFoldedCompletely && notTouchingIndexTip) {
      scores.X = 0.95;
      // Ensure X does not trigger D, O, S, A
      scores.D = 0;
      scores.S = 0;
      scores.A = 0;
    }
  }

  // 2. Sign 'V': Index and middle fingers pointing UP spread in a V, ring and pinky curled
  // Distinguish from K: In V, thumb does not stand upright between index and middle
  if (indexExtUp && middleExtUp && !indexExtSide && !middleExtSide && !ringExt && !pinkyExt) {
    if (dIndexMiddle > 0.35) {
      scores.V = 0.88 + 0.12 * Math.min(1.0, (dIndexMiddle - 0.35) / 0.5);
    }
  }

  // 2a. Sign 'K': Index and middle fingers pointing UP into the sky, pinky and ring folded completely down,
  // thumb tip extended upright placed directly between index and middle fingers
  // "lòng bàn tay hướng về phía màn hình, ngón út và ngón nhẫn gập xuống hoàn toàn và ngón giữa, ngón trỏ chỉ lên trời, kèm theo ngón cái đặt giữa hai ngón: giữa và trỏ."
  if (indexExtUp && middleExtUp && !indexExtSide && !middleExtSide && !ringExt && !pinkyExt) {
    const isRingPinkyCurled = dRingWrist < 1.15 && dPinkyWrist < 1.15;
    // Check horizontal placement of thumb tip strictly between index and middle fingers
    const minX = Math.min(p[5].x, p[8].x) - 0.05;
    const maxX = Math.max(p[9].x, p[12].x) + 0.05;
    const isThumbBetweenX = p[4].x >= minX && p[4].x <= maxX;
    // For K, thumb tip must point UP (protrude upwards between the two fingers, p[4].y < p[5].y or p[4].y < p[6].y)
    // whereas in V, thumb is folded/curled down over ring finger (p[4].y >= p[5].y)
    const isThumbUpright = p[4].y < p[5].y;
    const midIndexMiddleMcp = { x: (p[5].x + p[9].x) / 2, y: (p[5].y + p[9].y) / 2 };
    const dThumbMidMcp = dist(p[4], midIndexMiddleMcp);

    if (isRingPinkyCurled && isThumbBetweenX && isThumbUpright && dThumbMidMcp <= 0.40) {
      scores.K = 0.94;
      // K takes precedence over V when thumb is upright between index and middle
      scores.V = 0;
    }
  }

  // 2b. Sign 'W': Index, middle, and ring fingers extended UP in a W formation, pinky curled down, thumb folded
  // "nhận diện bằng cách giơ 3 ngón trỏ, ngón giữa và ngón nhẫn"
  if (indexExt && middleExt && ringExt && !pinkyExt && !indexExtSide && !middleExtSide) {
    const isUpright = p[8].y < p[6].y && p[12].y < p[10].y && p[16].y < p[14].y;
    if (isUpright) {
      scores.W = 0.94;
    }
  }

  // 3. Sign 'U': Index and middle fingers pointing strictly UP together touching (parallel, not crossed), ring and pinky curled down, thumb folded
  // "ngón trỏ và ngón giữa giơ lên giáp với nhau, còn ngón tay út, ngón cái và ngón nhẫn co lại. Hai ngón giữa và trỏ chỉ lên trời."
  // Strictly prevent sideways H gestures from triggering U: must be upright (indexExtUp && middleExtUp) and not sideways (!indexExtSide && !middleExtSide)
  if (indexExtUp && middleExtUp && !indexExtSide && !middleExtSide && !ringExt && !pinkyExt) {
    if (dIndexMiddle <= 0.35) {
      const touchCloseness = Math.max(0, 1.0 - dIndexMiddle / 0.35);
      scores.U = 0.90 + 0.10 * touchCloseness;
    }
  }

  // 3b. Sign 'R': Index and middle fingers extended UP and crossed together, middle finger crossed behind index finger.
  // Three fingers (pinky, ring, thumb) folded completely down, thumb touching ring and pinky fingers.
  // "lòng bàn tay hướng về phía màn hình và khi ba ngón út, nhẫn và cái gập xuống hoàn toàn. Ngón cái chạm vào ngón nhẫn và út, còn ngón trỏ và ngón giữa duỗi. Ngón giữa bắt chéo với ngón trỏ và nằm phía sau ngón trỏ."
  if (indexExtUp && middleExtUp && !indexExtSide && !middleExtSide && !ringExt && !pinkyExt) {
    const isRingPinkyCurled = dRingWrist < 1.15 && dPinkyWrist < 1.15;
    // Thumb folded down and touching near ring/pinky
    const dThumbRingNearest = Math.min(
      dist(p[4], p[14]),
      dist(p[4], p[15]),
      dist(p[4], p[16])
    );
    const dThumbPinkyNearest = Math.min(
      dist(p[4], p[18]),
      dist(p[4], p[19]),
      dist(p[4], p[20])
    );
    const isThumbTouchingRingPinky = (dThumbRingNearest <= 0.48 || dThumbPinkyNearest <= 0.48) && p[4].y >= p[5].y - 0.05;

    // Crossing geometry:
    // At MCP base: index MCP (5) and middle MCP (9) define base direction
    // At TIP level: fingers MUST actually cross over each other:
    // If baseSeparation > 0 (index is to the left of middle, standard right hand view),
    // then middle finger crosses over index so tipSeparation <= -0.01 (middle tip is to the left of index tip).
    // If baseSeparation < 0 (left hand view), tipSeparation >= 0.01.
    const baseSeparation = p[9].x - p[5].x;
    const tipSeparation = p[12].x - p[8].x;
    const isGenuineCrossing = (baseSeparation > 0.04 && tipSeparation <= -0.01) ||
                              (baseSeparation < -0.04 && tipSeparation >= 0.01);

    // Check if middle finger is behind index finger (Z-depth if available)
    const rawLm = feats.rawLandmarks;
    const hasZ = rawLm && rawLm[8]?.z !== undefined && rawLm[12]?.z !== undefined;
    // In camera space: more positive z or deeper away from camera means behind
    const isMiddleBehind = hasZ ? (rawLm[12].z! >= rawLm[8].z! - 0.04) : true;

    if (isRingPinkyCurled && isThumbTouchingRingPinky && isGenuineCrossing && isMiddleBehind) {
      scores.R = 0.95;
      // R takes precedence over U and V when fingers are crossed
      scores.U = 0;
      scores.V = 0;
    }
  }

  // 4. Sign 'H': Index and middle fingers extended together pointing SIDEWAYS ("chỉ qua hai bên vuông góc với mặt đất"), ring and pinky curled
  // Strictly prevent upright U gestures from triggering H: must be sideways (indexExtSide && middleExtSide) and not upright (!indexExtUp && !middleExtUp)
  if (indexExtSide && middleExtSide && !indexExtUp && !middleExtUp && !ringExt && !pinkyExt) {
    if (dIndexMiddle <= 0.38) {
      const parallelism = Math.max(0, 1.0 - dIndexMiddle / 0.38);
      scores.H = 0.88 + 0.12 * parallelism;
    }
  }

  // 5. Sign 'L': Thumb and index finger extended at right angle (~90°), others folded
  // Must have thumb spread wide away from middle finger to prevent confusion with D
  if (indexExt && !middleExt && !ringExt && !pinkyExt) {
    const dThumbMiddleNearest = Math.min(
      dThumbMiddle,
      dist(p[4], p[10]),
      dist(p[4], p[11]),
      dist(p[4], p[12])
    );
    if (dThumbIndex > 0.65 && (thumbExt || p[4].x < p[3].x) && dThumbMiddleNearest > 0.50) {
      scores.L = 0.90 + 0.10 * Math.min(1.0, dThumbIndex / 1.2);
    }
  }

  // 6. Sign 'I': Only pinky extended straight up, others curled
  if (pinkyExt && !indexExt && !middleExt && !ringExt) {
    if (!thumbExt || dThumbPinky <= 1.25) {
      scores.I = 0.93;
    }
  }

  // 7. Sign 'Y': Thumb and pinky extended wide (Shaka), middle three fingers curled
  if (pinkyExt && thumbExt && !indexExt && !middleExt && !ringExt) {
    if (dThumbPinky > 1.15) {
      scores.Y = 0.95;
    }
  }

  // 8. Sign 'O': Hand forming an open circular O-ring
  // Supports BOTH orientations:
  // - Palm facing towards camera (frontal view where thumb and fingertips curve to meet in front of palm)
  // - Hand turned sideways (profile view showing the open circular hole)
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    const isTipTouching = dThumbIndex <= 0.40;
    const tipsMeetDirectly = Math.abs(p[8].y - p[4].y) <= 0.22 && Math.abs(p[8].x - p[4].x) <= 0.28;

    // Case A: Hand turned sideways (profile aperture)
    const isSidewaysO = isTipTouching &&
      p[6].y < p[5].y + 0.08 &&
      dist(p[6], p[2]) >= 0.30 &&
      dist(p[5], p[4]) >= 0.20;

    // Case B: Palm facing towards camera (frontal or 3/4 view)
    // Thumb arches out from base and meets index & middle fingertips in front of palm
    const isFrontalO = isTipTouching &&
      tipsMeetDirectly &&
      dThumbMiddle <= 0.52 &&
      dist(p[8], p[12]) <= 0.38 &&
      dThumbWrist >= 0.65;

    if (isSidewaysO || isFrontalO) {
      const touchTightness = Math.max(0, 1.0 - dThumbIndex / 0.40);
      scores.O = 0.88 + 0.12 * touchTightness;
    }
  }

  // 9. Sign 'C': Hand curved into a distinct C shape, thumb and fingers curved but not touching
  if (!indexExt && !middleExt) {
    if (dThumbIndex >= 0.45 && dThumbIndex <= 0.98 && p[4].y > p[8].y) {
      scores.C = 0.87;
    }
  }

  // 10. Sign 'S': All 5 fingers folded into a fist with thumb crossed horizontally across fingers
  // "cả năm ngón tay đều gập lại như hình một nắm đấm và ngón cái bắt chéo qua các ngón kia"
  if (!indexExt && !middleExt && !ringExt && !pinkyExt && !thumbExt) {
    const isFistCurled = p[6].y >= p[5].y - 0.08 && p[8].y >= p[6].y * 0.9;
    const isThumbCrossed = p[4].y >= p[5].y - 0.12 && (p[4].x >= p[5].x - 0.18 || dist(p[4], p[6]) < 0.48 || dist(p[4], p[9]) < 0.58);
    const isCompactFist = dIndexWrist < 1.08 && dMiddleWrist < 1.08;
    // Strictly ensure it is NOT an O sign
    const notO = scores.O === 0;

    if (isFistCurled && isThumbCrossed && isCompactFist && notO) {
      scores.S = 0.92;
    }
  }

  // 11. Sign 'A': Tight fist with thumb straight UP alongside index finger (lateral side, not crossed)
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    const isThumbUpright = p[4].y < p[5].y - 0.02;
    const isThumbAlongSide = p[4].x < p[5].x + 0.12 && dist(p[4], p[9]) > 0.40;
    if (isThumbUpright && isThumbAlongSide && dThumbIndex > 0.30) {
      scores.A = 0.90;
    }
  }

  // 12. Sign 'T': Fist with 4 fingers (index, middle, ring, pinky) folded completely down,
  // and thumb extended, placed specifically between index and middle fingers.
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    const notO = scores.O === 0;
    const isFourFolded = dIndexWrist < 1.15 && dMiddleWrist < 1.15 && dRingWrist < 1.15 && dPinkyWrist < 1.15;
    // Thumb tip placed between index finger (MCP 5 / PIP 6) and middle finger (MCP 9 / PIP 10)
    const midIndexMiddleMcp = { x: (p[5].x + p[9].x) / 2, y: (p[5].y + p[9].y) / 2 };
    const midIndexMiddlePip = { x: (p[6].x + p[10].x) / 2, y: (p[6].y + p[10].y) / 2 };
    const dThumbMidIndexMiddle = Math.min(dist(p[4], midIndexMiddleMcp), dist(p[4], midIndexMiddlePip));
    
    // Check horizontal positioning between index and middle
    const minX = Math.min(p[5].x, p[6].x) - 0.06;
    const maxX = Math.max(p[9].x, p[10].x) + 0.06;
    const isBetweenIndexMiddleX = p[4].x >= minX && p[4].x <= maxX;
    // Closer to index-middle cleft than to ring finger
    const isCloserToIndexMiddle = dist(p[4], p[5]) < dist(p[4], p[13]) && dist(p[4], p[9]) < dist(p[4], p[17]);

    if (notO && isFourFolded && isBetweenIndexMiddleX && dThumbMidIndexMiddle <= 0.36 && isCloserToIndexMiddle) {
      scores.T = 0.94;
      // Exclude S and A when T is detected
      scores.S = 0;
      scores.A = 0;
    }
  }

  // 13. Sign 'N': Fist with 4 fingers (index, middle, ring, pinky) folded completely down,
  // and thumb extended, placed specifically between middle and ring fingers.
  // "lòng bàn tay hướng về phía màn hình, khi bốn ngón: trỏ, giữa, nhẫn và út gập xuống hoàn toàn và ngón cái duỗi ra, đặt giữa ngón nhẫn và giữa đang gập xuống."
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    const notO = scores.O === 0;
    const isFourFolded = dIndexWrist < 1.15 && dMiddleWrist < 1.15 && dRingWrist < 1.15 && dPinkyWrist < 1.15;
    // Thumb tip placed between middle finger (MCP 9 / PIP 10) and ring finger (MCP 13 / PIP 14)
    const midMiddleRingMcp = { x: (p[9].x + p[13].x) / 2, y: (p[9].y + p[13].y) / 2 };
    const midMiddleRingPip = { x: (p[10].x + p[14].x) / 2, y: (p[10].y + p[14].y) / 2 };
    const dThumbMidMiddleRing = Math.min(dist(p[4], midMiddleRingMcp), dist(p[4], midMiddleRingPip));

    // Check horizontal positioning between middle and ring
    const minX = Math.min(p[9].x, p[10].x) - 0.06;
    const maxX = Math.max(p[13].x, p[14].x) + 0.06;
    const isBetweenMiddleRingX = p[4].x >= minX && p[4].x <= maxX;
    // Farther from index MCP than T is, closer to ring MCP, but NOT past ring MCP towards pinky
    const isCloserToMiddleRing = dist(p[4], p[13]) <= dist(p[4], p[5]) && dist(p[4], p[9]) < dist(p[4], p[17]);

    if (notO && isFourFolded && isBetweenMiddleRingX && dThumbMidMiddleRing <= 0.36 && isCloserToMiddleRing) {
      scores.N = 0.93;
      // Exclude S, A, and T when N is detected
      scores.S = 0;
      scores.A = 0;
      scores.T = 0;
    }
  }

  // 14. Sign 'M': Fist with 4 fingers (index, middle, ring, pinky) folded completely down,
  // and thumb extended, placed specifically between ring and pinky fingers.
  // "lòng bàn tay hướng về phía màn hình, khi bốn ngón: trỏ, giữa, nhẫn và út gập xuống hoàn toàn và ngón cái duỗi ra, đặt giữa ngón nhẫn và út đang gập xuống."
  if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
    const notO = scores.O === 0;
    const isFourFolded = dIndexWrist < 1.15 && dMiddleWrist < 1.15 && dRingWrist < 1.15 && dPinkyWrist < 1.15;
    // Thumb tip placed between ring finger (MCP 13 / PIP 14) and pinky finger (MCP 17 / PIP 18)
    const midRingPinkyMcp = { x: (p[13].x + p[17].x) / 2, y: (p[13].y + p[17].y) / 2 };
    const midRingPinkyPip = { x: (p[14].x + p[18].x) / 2, y: (p[14].y + p[18].y) / 2 };
    const dThumbMidRingPinky = Math.min(dist(p[4], midRingPinkyMcp), dist(p[4], midRingPinkyPip));

    // Check horizontal positioning between ring and pinky
    const minX = Math.min(p[13].x, p[14].x) - 0.06;
    const maxX = Math.max(p[17].x, p[18].x) + 0.06;
    const isBetweenRingPinkyX = p[4].x >= minX && p[4].x <= maxX;
    // Closer to pinky than to middle or index
    const isCloserToRingPinky = dist(p[4], p[17]) < dist(p[4], p[9]) && dist(p[4], p[13]) < dist(p[4], p[5]);

    if (notO && isFourFolded && isBetweenRingPinkyX && dThumbMidRingPinky <= 0.36 && isCloserToRingPinky) {
      scores.M = 0.93;
      // Exclude S, A, T, and N when M is detected
      scores.S = 0;
      scores.A = 0;
      scores.T = 0;
      scores.N = 0;
    }
  }

  // 5-Finger Open Hand check (for 5s Space gesture)
  const isOpenPalmFive = Boolean(thumbExt && indexExt && middleExt && ringExt && pinkyExt);
  if (isOpenPalmFive) {
    scores.B = 0;
  }

  // Best letter selection
  let bestLetter: string | null = null;
  let bestScore = 0;

  for (const sign of CORE_SIGNS) {
    if (scores[sign] > bestScore) {
      bestScore = scores[sign];
      bestLetter = sign;
    }
  }

  return {
    letter: isOpenPalmFive ? 'SPACE' : (bestScore >= 0.70 ? bestLetter : null),
    confidence: isOpenPalmFive ? 0.95 : (bestScore >= 0.70 ? Number(bestScore.toFixed(2)) : 0),
    scores,
    features: feats,
    fingerStates: {
      thumb: thumbExt,
      index: indexExt,
      middle: middleExt,
      ring: ringExt,
      pinky: pinkyExt,
    },
    isOpenPalmFive,
  };
}

export const SIGN_DESCRIPTIONS: Record<string, { title: string; en: string }> = {
  A: {
    title: 'Fist with upright thumb',
    en: 'Make a tight fist with thumb resting straight up alongside index finger.',
  },
  B: {
    title: 'Flat hand fingers together',
    en: 'All 4 fingers straight up touching each other, thumb tucked across palm.',
  },
  C: {
    title: 'Curved C-arc',
    en: 'Curve all fingers and thumb into an open C-arc silhouette.',
  },
  D: {
    title: 'Index finger UP, thumb touching middle finger',
    en: 'Palm facing camera, 3 fingers (middle, ring, pinky) folded down, index finger pointing straight up, thumb touching near middle finger.',
  },
  F: {
    title: 'OK-sign with 3 fingers upright',
    en: 'Palm facing camera, middle, ring, and pinky fingers extended straight up, thumb and index finger folded touching tips.',
  },
  G: {
    title: 'Index and thumb pointing sideways',
    en: 'Three fingers (pinky, ring, middle) folded completely, index finger and thumb extended pointing sideways to the right or left of the screen.',
  },
  H: {
    title: 'Dual parallel fingers pointing sideways',
    en: 'Index and middle fingers extended together pointing horizontally to the side, perpendicular to the ground.',
  },
  I: {
    title: 'Pinky finger up',
    en: 'Only pinky finger extended straight up, other fingers curled.',
  },
  K: {
    title: 'Peace sign with upright thumb in between',
    en: 'Palm facing camera, pinky and ring fingers folded down completely, index and middle fingers pointing straight up into the sky, thumb tip upright placed between index and middle fingers.',
  },
  L: {
    title: 'L-shape right angle',
    en: 'Thumb and index finger form a perpendicular L shape (~90°).',
  },
  M: {
    title: 'Fist with thumb between ring and pinky fingers',
    en: 'Palm facing camera, all 4 fingers (index, middle, ring, pinky) folded completely down into a fist, thumb extended placed between ring and pinky fingers.',
  },
  N: {
    title: 'Fist with thumb between middle and ring fingers',
    en: 'Palm facing camera, all 4 fingers (index, middle, ring, pinky) folded completely down into a fist, thumb extended placed between middle and ring fingers.',
  },
  O: {
    title: 'O-ring circle (Palm facing camera or sideways)',
    en: 'Form an open circular ring with thumb and curved fingertips. Accurately recognized both when palm faces the camera or when turned sideways.',
  },
  R: {
    title: 'Crossed index and middle fingers',
    en: 'Palm facing camera, pinky, ring and thumb folded completely down, thumb touching ring and pinky fingers, index and middle fingers extended with middle finger crossed behind index finger.',
  },
  S: {
    title: 'Fist with thumb crossed in front',
    en: 'Make a tight fist with all 5 fingers folded and thumb crossed horizontally across the front.',
  },
  T: {
    title: 'Fist with thumb between index and middle fingers',
    en: 'Palm facing camera, all 4 fingers (index, middle, ring, pinky) folded completely down into a fist, thumb extended placed between index and middle fingers.',
  },
  U: {
    title: 'Dual upright fingers touching',
    en: 'Index and middle fingers extended straight up touching each other, thumb curled over ring and pinky.',
  },
  V: {
    title: 'Victory V-shape',
    en: 'Index and middle fingers spread in an open V sign.',
  },
  W: {
    title: 'Three upright fingers (W-shape)',
    en: 'Raise index, middle, and ring fingers upright in a W formation, with pinky curled down and thumb folded across.',
  },
  X: {
    title: 'Hooked index finger with other 4 fingers folded completely down',
    en: 'Palm facing camera, all 4 fingers (pinky, ring, middle, thumb) folded completely down, index finger lower half extended upright and upper half bent down into a hook.',
  },
  Y: {
    title: 'Shaka sign',
    en: 'Thumb and pinky extended outwards, middle three fingers folded.',
  },
};
