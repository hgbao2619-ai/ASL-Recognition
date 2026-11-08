import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { classifyASL, CORE_SIGNS } from '../src/utils/aslClassifier.ts';

function createMockHand(pose: 'open_palm' | 'fist_A' | 'fist_S' | 'circle_O' | 'circle_O_frontal' | 'three_W' | 'peace_V' | 'upright_U' | 'sideways_H' | 'L_sign' | 'pinky_I' | 'index_D' | 'ok_F' | 'hook_X' | 'sideways_G' | 'peace_K' | 'fist_T' | 'fist_N' | 'fist_M' | 'crossed_R') {
  // 21 points: 0 is wrist at (0.5, 0.8)
  const pts = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.8, z: 0 }));

  // Set wrist
  pts[0] = { x: 0.5, y: 0.8, z: 0 };
  // Middle MCP (9) sets ref scale
  pts[9] = { x: 0.5, y: 0.5, z: 0 }; // distance = 0.3

  if (pose === 'open_palm') {
    // All 5 fingers extended and spread
    // Thumb
    pts[3] = { x: 0.35, y: 0.60, z: 0 };
    pts[4] = { x: 0.20, y: 0.55, z: 0 };
    // Index
    pts[5] = { x: 0.45, y: 0.52, z: 0 };
    pts[6] = { x: 0.42, y: 0.42, z: 0 };
    pts[8] = { x: 0.40, y: 0.18, z: 0 };
    // Middle
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.40, z: 0 };
    pts[12] = { x: 0.50, y: 0.15, z: 0 };
    // Ring
    pts[13] = { x: 0.55, y: 0.52, z: 0 };
    pts[14] = { x: 0.58, y: 0.42, z: 0 };
    pts[16] = { x: 0.60, y: 0.18, z: 0 };
    // Pinky
    pts[17] = { x: 0.60, y: 0.55, z: 0 };
    pts[18] = { x: 0.65, y: 0.45, z: 0 };
    pts[20] = { x: 0.70, y: 0.22, z: 0 };
  } else if (pose === 'fist_A') {
    // 4 fingers curled down into tight fist
    pts[5] = { x: 0.46, y: 0.52, z: 0 };
    pts[6] = { x: 0.46, y: 0.60, z: 0 };
    pts[8] = { x: 0.46, y: 0.68, z: 0 };
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };
    pts[13] = { x: 0.54, y: 0.52, z: 0 };
    pts[14] = { x: 0.54, y: 0.62, z: 0 };
    pts[16] = { x: 0.54, y: 0.70, z: 0 };
    pts[17] = { x: 0.58, y: 0.55, z: 0 };
    pts[18] = { x: 0.58, y: 0.64, z: 0 };
    pts[20] = { x: 0.58, y: 0.72, z: 0 };

    // Thumb pointing straight UP alongside index finger (lateral side)
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.38, y: 0.68, z: 0 };
    pts[3] = { x: 0.38, y: 0.58, z: 0 };
    pts[4] = { x: 0.38, y: 0.48, z: 0 }; // y=0.48 < MCP y=0.52
  } else if (pose === 'fist_S') {
    // 4 fingers curled down into tight fist
    pts[5] = { x: 0.46, y: 0.52, z: 0 };
    pts[6] = { x: 0.46, y: 0.60, z: 0 };
    pts[8] = { x: 0.46, y: 0.68, z: 0 };
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };
    pts[13] = { x: 0.54, y: 0.52, z: 0 };
    pts[14] = { x: 0.54, y: 0.62, z: 0 };
    pts[16] = { x: 0.54, y: 0.70, z: 0 };
    pts[17] = { x: 0.58, y: 0.55, z: 0 };
    pts[18] = { x: 0.58, y: 0.64, z: 0 };
    pts[20] = { x: 0.58, y: 0.72, z: 0 };

    // Thumb crossed HORIZONTALLY across front of curled fingers
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.38, y: 0.68, z: 0 };
    pts[3] = { x: 0.44, y: 0.62, z: 0 };
    pts[4] = { x: 0.50, y: 0.60, z: 0 }; // crossed horizontally across front of fingers
  } else if (pose === 'circle_O') {
    // Sideways profile view: Index finger arches UP and forward to form top rim of O
    pts[5] = { x: 0.48, y: 0.52, z: 0 };
    pts[6] = { x: 0.45, y: 0.38, z: 0 }; // y=0.38 < MCP y=0.52 (arches up)
    pts[7] = { x: 0.40, y: 0.42, z: 0 };
    pts[8] = { x: 0.36, y: 0.48, z: 0 }; // tip

    // Middle, Ring, Pinky follow the circular contour
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.47, y: 0.38, z: 0 };
    pts[12] = { x: 0.36, y: 0.49, z: 0 };
    pts[14] = { x: 0.49, y: 0.42, z: 0 };
    pts[16] = { x: 0.38, y: 0.52, z: 0 };
    pts[18] = { x: 0.52, y: 0.45, z: 0 };
    pts[20] = { x: 0.40, y: 0.54, z: 0 };

    // Thumb arches UP from base to meet index tip
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.38, y: 0.65, z: 0 };
    pts[3] = { x: 0.34, y: 0.56, z: 0 };
    pts[4] = { x: 0.36, y: 0.49, z: 0 }; // meets index tip at (0.36, 0.48)
  } else if (pose === 'circle_O_frontal') {
    // Frontal view: Palm facing directly towards camera, thumb and index/middle tips meet in front
    pts[5] = { x: 0.42, y: 0.52, z: 0 }; // Index MCP
    pts[9] = { x: 0.50, y: 0.50, z: 0 }; // Middle MCP
    pts[13] = { x: 0.56, y: 0.52, z: 0 }; // Ring MCP
    pts[17] = { x: 0.62, y: 0.55, z: 0 }; // Pinky MCP

    // Thumb base and joints
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.36, y: 0.68, z: 0 };
    pts[3] = { x: 0.34, y: 0.58, z: 0 };
    pts[4] = { x: 0.42, y: 0.50, z: 0 }; // Thumb tip

    // Index finger curves down/forward to touch thumb tip
    pts[6] = { x: 0.42, y: 0.46, z: 0 };
    pts[7] = { x: 0.42, y: 0.48, z: 0 };
    pts[8] = { x: 0.42, y: 0.50, z: 0 }; // Index tip meets thumb tip

    // Middle, Ring, Pinky curve alongside
    pts[10] = { x: 0.49, y: 0.45, z: 0 };
    pts[11] = { x: 0.46, y: 0.49, z: 0 };
    pts[12] = { x: 0.44, y: 0.51, z: 0 }; // Middle tip right next to thumb/index

    pts[14] = { x: 0.55, y: 0.47, z: 0 };
    pts[16] = { x: 0.50, y: 0.52, z: 0 };

    pts[18] = { x: 0.60, y: 0.50, z: 0 };
    pts[20] = { x: 0.54, y: 0.54, z: 0 };
  } else if (pose === 'three_W') {
    // Index, Middle, Ring extended UP in a W; Pinky curled down; Thumb folded
    // Index finger
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.42, y: 0.40, z: 0 };
    pts[8] = { x: 0.38, y: 0.18, z: 0 };

    // Middle finger
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.38, z: 0 };
    pts[12] = { x: 0.50, y: 0.15, z: 0 };

    // Ring finger
    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.58, y: 0.40, z: 0 };
    pts[16] = { x: 0.62, y: 0.18, z: 0 };

    // Pinky curled down
    pts[17] = { x: 0.62, y: 0.55, z: 0 };
    pts[18] = { x: 0.62, y: 0.64, z: 0 };
    pts[20] = { x: 0.62, y: 0.72, z: 0 };

    // Thumb folded across holding pinky
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.46, y: 0.68, z: 0 };
    pts[3] = { x: 0.52, y: 0.62, z: 0 };
    pts[4] = { x: 0.58, y: 0.60, z: 0 };
  } else if (pose === 'peace_V') {
    // Index tip (8) extended high up
    pts[5] = { x: 0.45, y: 0.52, z: 0 };
    pts[6] = { x: 0.42, y: 0.45, z: 0 };
    pts[8] = { x: 0.38, y: 0.20, z: 0 };

    // Middle tip (12) extended high up, spread from index
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.55, y: 0.45, z: 0 };
    pts[12] = { x: 0.62, y: 0.20, z: 0 };

    // Ring (16) & Pinky (20) curled down
    pts[14] = { x: 0.52, y: 0.6, z: 0 };
    pts[16] = { x: 0.52, y: 0.65, z: 0 };
    pts[18] = { x: 0.55, y: 0.65, z: 0 };
    pts[20] = { x: 0.55, y: 0.70, z: 0 };

    // Thumb curled
    pts[4] = { x: 0.48, y: 0.6, z: 0 };
  } else if (pose === 'upright_U') {
    // Both index and middle extended high up, touching each other (dIndexMiddle is small)
    // Index finger
    pts[5] = { x: 0.48, y: 0.52, z: 0 };
    pts[6] = { x: 0.48, y: 0.42, z: 0 };
    pts[8] = { x: 0.48, y: 0.20, z: 0 };

    // Middle finger right next to index finger
    pts[9] = { x: 0.52, y: 0.50, z: 0 };
    pts[10] = { x: 0.52, y: 0.42, z: 0 };
    pts[12] = { x: 0.52, y: 0.20, z: 0 };

    // Ring (16) & Pinky (20) curled down
    pts[14] = { x: 0.52, y: 0.6, z: 0 };
    pts[16] = { x: 0.52, y: 0.65, z: 0 };
    pts[18] = { x: 0.55, y: 0.65, z: 0 };
    pts[20] = { x: 0.55, y: 0.70, z: 0 };

    // Thumb curled
    pts[4] = { x: 0.48, y: 0.6, z: 0 };
  } else if (pose === 'sideways_H') {
    // Index and middle fingers extended horizontally to the side (leftwards)
    // Index finger
    pts[5] = { x: 0.50, y: 0.48, z: 0 };
    pts[6] = { x: 0.35, y: 0.48, z: 0 };
    pts[8] = { x: 0.15, y: 0.48, z: 0 };

    // Middle finger parallel to index finger, right below it
    pts[9] = { x: 0.50, y: 0.54, z: 0 };
    pts[10] = { x: 0.35, y: 0.54, z: 0 };
    pts[12] = { x: 0.15, y: 0.54, z: 0 };

    // Ring (16) & Pinky (20) curled down
    pts[14] = { x: 0.52, y: 0.65, z: 0 };
    pts[16] = { x: 0.52, y: 0.70, z: 0 };
    pts[18] = { x: 0.55, y: 0.70, z: 0 };
    pts[20] = { x: 0.55, y: 0.75, z: 0 };

    // Thumb curled
    pts[4] = { x: 0.48, y: 0.65, z: 0 };
  } else if (pose === 'L_sign') {
    // Index tip (8) extended high
    pts[5] = { x: 0.5, y: 0.52, z: 0 };
    pts[6] = { x: 0.5, y: 0.45, z: 0 };
    pts[8] = { x: 0.5, y: 0.20, z: 0 };

    // Thumb tip (4) extended far to the left
    pts[3] = { x: 0.35, y: 0.6, z: 0 };
    pts[4] = { x: 0.20, y: 0.6, z: 0 };

    // Middle, Ring, Pinky curled
    pts[10] = { x: 0.5, y: 0.6, z: 0 };
    pts[12] = { x: 0.5, y: 0.65, z: 0 };
    pts[14] = { x: 0.52, y: 0.6, z: 0 };
    pts[16] = { x: 0.52, y: 0.65, z: 0 };
    pts[18] = { x: 0.55, y: 0.65, z: 0 };
    pts[20] = { x: 0.55, y: 0.70, z: 0 };
  } else if (pose === 'pinky_I') {
    // Pinky extended
    pts[18] = { x: 0.60, y: 0.45, z: 0 };
    pts[20] = { x: 0.60, y: 0.20, z: 0 };

    // Index, Middle, Ring curled
    pts[6] = { x: 0.45, y: 0.6, z: 0 };
    pts[8] = { x: 0.45, y: 0.65, z: 0 };
    pts[10] = { x: 0.5, y: 0.6, z: 0 };
    pts[12] = { x: 0.5, y: 0.65, z: 0 };
    pts[14] = { x: 0.55, y: 0.6, z: 0 };
    pts[16] = { x: 0.55, y: 0.65, z: 0 };
    pts[4] = { x: 0.48, y: 0.6, z: 0 };
  } else if (pose === 'index_D') {
    // Index pointing UP to the sky
    pts[5] = { x: 0.45, y: 0.52, z: 0 };
    pts[6] = { x: 0.45, y: 0.40, z: 0 };
    pts[8] = { x: 0.45, y: 0.18, z: 0 };

    // Middle, Ring, Pinky curled down
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[11] = { x: 0.50, y: 0.64, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.54, y: 0.52, z: 0 };
    pts[14] = { x: 0.54, y: 0.62, z: 0 };
    pts[16] = { x: 0.54, y: 0.70, z: 0 };

    pts[17] = { x: 0.58, y: 0.55, z: 0 };
    pts[18] = { x: 0.58, y: 0.64, z: 0 };
    pts[20] = { x: 0.58, y: 0.72, z: 0 };

    // Thumb touches near curled middle finger tip/pad
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.44, y: 0.68, z: 0 };
    pts[3] = { x: 0.46, y: 0.65, z: 0 };
    pts[4] = { x: 0.49, y: 0.67, z: 0 };
  } else if (pose === 'ok_F') {
    // Middle, Ring, Pinky extended UP
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.38, z: 0 };
    pts[12] = { x: 0.50, y: 0.15, z: 0 };

    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.58, y: 0.40, z: 0 };
    pts[16] = { x: 0.60, y: 0.18, z: 0 };

    pts[17] = { x: 0.62, y: 0.55, z: 0 };
    pts[18] = { x: 0.65, y: 0.45, z: 0 };
    pts[20] = { x: 0.68, y: 0.22, z: 0 };

    // Index finger curved down to touch thumb
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.40, y: 0.58, z: 0 };
    pts[7] = { x: 0.40, y: 0.62, z: 0 };
    pts[8] = { x: 0.42, y: 0.65, z: 0 };

    // Thumb curved to meet index tip
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.40, y: 0.70, z: 0 };
    pts[3] = { x: 0.40, y: 0.66, z: 0 };
    pts[4] = { x: 0.42, y: 0.65, z: 0 };
  } else if (pose === 'hook_X') {
    // Index finger standing upright at base, but bent halfway into a hook
    pts[5] = { x: 0.45, y: 0.52, z: 0 }; // MCP
    pts[6] = { x: 0.45, y: 0.38, z: 0 }; // PIP (upright)
    pts[7] = { x: 0.42, y: 0.42, z: 0 }; // DIP (curled down)
    pts[8] = { x: 0.42, y: 0.46, z: 0 }; // Tip (hooked down)

    // Middle, Ring, Pinky curled down
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[11] = { x: 0.50, y: 0.64, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.54, y: 0.52, z: 0 };
    pts[14] = { x: 0.54, y: 0.62, z: 0 };
    pts[16] = { x: 0.54, y: 0.70, z: 0 };

    pts[17] = { x: 0.58, y: 0.55, z: 0 };
    pts[18] = { x: 0.58, y: 0.64, z: 0 };
    pts[20] = { x: 0.58, y: 0.72, z: 0 };

    // Thumb touches middle finger
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.44, y: 0.68, z: 0 };
    pts[3] = { x: 0.46, y: 0.65, z: 0 };
    pts[4] = { x: 0.49, y: 0.67, z: 0 };
  } else if (pose === 'sideways_G') {
    // 3 fingers completely folded down
    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[11] = { x: 0.50, y: 0.64, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.54, y: 0.52, z: 0 };
    pts[14] = { x: 0.54, y: 0.62, z: 0 };
    pts[16] = { x: 0.54, y: 0.70, z: 0 };

    pts[17] = { x: 0.58, y: 0.55, z: 0 };
    pts[18] = { x: 0.58, y: 0.64, z: 0 };
    pts[20] = { x: 0.58, y: 0.72, z: 0 };

    // Index finger pointing horizontally sideways to the left
    pts[5] = { x: 0.50, y: 0.48, z: 0 };
    pts[6] = { x: 0.35, y: 0.48, z: 0 };
    pts[8] = { x: 0.15, y: 0.48, z: 0 };

    // Thumb extended alongside pointing sideways
    pts[1] = { x: 0.50, y: 0.65, z: 0 };
    pts[2] = { x: 0.40, y: 0.60, z: 0 };
    pts[3] = { x: 0.30, y: 0.56, z: 0 };
    pts[4] = { x: 0.20, y: 0.54, z: 0 };
  } else if (pose === 'peace_K') {
    // Index and middle fingers extended UP into the sky, spread apart
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.42, y: 0.40, z: 0 };
    pts[8] = { x: 0.38, y: 0.18, z: 0 };

    pts[9] = { x: 0.52, y: 0.50, z: 0 };
    pts[10] = { x: 0.54, y: 0.38, z: 0 };
    pts[12] = { x: 0.60, y: 0.18, z: 0 };

    // Ring & Pinky folded completely down
    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.56, y: 0.62, z: 0 };
    pts[16] = { x: 0.56, y: 0.70, z: 0 };

    pts[17] = { x: 0.60, y: 0.55, z: 0 };
    pts[18] = { x: 0.60, y: 0.64, z: 0 };
    pts[20] = { x: 0.60, y: 0.72, z: 0 };

    // Thumb upright placed directly between index and middle fingers
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.44, y: 0.65, z: 0 };
    pts[3] = { x: 0.46, y: 0.55, z: 0 };
    pts[4] = { x: 0.48, y: 0.46, z: 0 }; // between x=0.44 and x=0.52, upright y=0.46
  } else if (pose === 'fist_T') {
    // All 4 fingers folded completely down into a fist
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.44, y: 0.60, z: 0 };
    pts[8] = { x: 0.44, y: 0.68, z: 0 };

    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.56, y: 0.62, z: 0 };
    pts[16] = { x: 0.56, y: 0.70, z: 0 };

    pts[17] = { x: 0.60, y: 0.55, z: 0 };
    pts[18] = { x: 0.60, y: 0.64, z: 0 };
    pts[20] = { x: 0.60, y: 0.72, z: 0 };

    // Thumb extended and placed specifically between index and middle fingers
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.42, y: 0.68, z: 0 };
    pts[3] = { x: 0.44, y: 0.60, z: 0 };
    pts[4] = { x: 0.47, y: 0.55, z: 0 }; // x=0.47 between index (0.44) and middle (0.50)
  } else if (pose === 'fist_N') {
    // All 4 fingers folded completely down into a fist
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.44, y: 0.60, z: 0 };
    pts[8] = { x: 0.44, y: 0.68, z: 0 };

    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.56, y: 0.62, z: 0 };
    pts[16] = { x: 0.56, y: 0.70, z: 0 };

    pts[17] = { x: 0.60, y: 0.55, z: 0 };
    pts[18] = { x: 0.60, y: 0.64, z: 0 };
    pts[20] = { x: 0.60, y: 0.72, z: 0 };

    // Thumb extended and placed specifically between middle and ring fingers
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.44, y: 0.68, z: 0 };
    pts[3] = { x: 0.48, y: 0.60, z: 0 };
    pts[4] = { x: 0.53, y: 0.55, z: 0 }; // x=0.53 between middle (0.50) and ring (0.56)
  } else if (pose === 'fist_M') {
    // All 4 fingers folded completely down into a fist
    pts[5] = { x: 0.44, y: 0.52, z: 0 };
    pts[6] = { x: 0.44, y: 0.60, z: 0 };
    pts[8] = { x: 0.44, y: 0.68, z: 0 };

    pts[9] = { x: 0.50, y: 0.50, z: 0 };
    pts[10] = { x: 0.50, y: 0.60, z: 0 };
    pts[12] = { x: 0.50, y: 0.68, z: 0 };

    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.56, y: 0.62, z: 0 };
    pts[16] = { x: 0.56, y: 0.70, z: 0 };

    pts[17] = { x: 0.62, y: 0.55, z: 0 };
    pts[18] = { x: 0.62, y: 0.64, z: 0 };
    pts[20] = { x: 0.62, y: 0.72, z: 0 };

    // Thumb extended and placed specifically between ring and pinky fingers
    pts[1] = { x: 0.42, y: 0.75, z: 0 };
    pts[2] = { x: 0.46, y: 0.68, z: 0 };
    pts[3] = { x: 0.52, y: 0.60, z: 0 };
    pts[4] = { x: 0.59, y: 0.55, z: 0 }; // x=0.59 between ring (0.56) and pinky (0.62)
  } else if (pose === 'crossed_R') {
    // Index finger extended UP
    pts[5] = { x: 0.46, y: 0.52, z: 0.00 }; // MCP
    pts[6] = { x: 0.47, y: 0.40, z: 0.00 }; // PIP
    pts[7] = { x: 0.48, y: 0.28, z: 0.00 }; // DIP
    pts[8] = { x: 0.49, y: 0.18, z: 0.00 }; // Index Tip in front (z=0)

    // Middle finger extended UP and crossed behind index finger (starts at MCP x=0.52, crosses to tip x=0.45, behind at z=0.04)
    pts[9] = { x: 0.52, y: 0.50, z: 0.01 };  // Middle MCP
    pts[10] = { x: 0.50, y: 0.38, z: 0.02 }; // Middle PIP
    pts[11] = { x: 0.47, y: 0.27, z: 0.03 }; // Middle DIP
    pts[12] = { x: 0.45, y: 0.18, z: 0.05 }; // Middle Tip crossed behind index (x=0.45 < x=0.49, z=0.05 > 0.00)

    // Ring & Pinky folded completely down
    pts[13] = { x: 0.56, y: 0.52, z: 0 };
    pts[14] = { x: 0.56, y: 0.62, z: 0 };
    pts[16] = { x: 0.56, y: 0.70, z: 0 };

    pts[17] = { x: 0.60, y: 0.55, z: 0 };
    pts[18] = { x: 0.60, y: 0.64, z: 0 };
    pts[20] = { x: 0.60, y: 0.72, z: 0 };

    // Thumb folded down and touching ring/pinky
    pts[1] = { x: 0.44, y: 0.75, z: 0 };
    pts[2] = { x: 0.48, y: 0.68, z: 0 };
    pts[3] = { x: 0.52, y: 0.64, z: 0 };
    pts[4] = { x: 0.56, y: 0.65, z: 0 }; // touches ring PIP/tip area
  }

  return pts;
}

describe('ASL Classifier Unit Tests', () => {
  test('handles empty or insufficient landmarks safely', () => {
    const res = classifyASL([]);
    assert.equal(res.letter, null);
    assert.equal(res.confidence, 0);
  });

  test('recognizes U sign with index and middle pointing UP together, never mistaking for H', () => {
    const mockHand = createMockHand('upright_U');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'U');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for U, got ${res.confidence}`);
    assert.equal(res.scores.H, 0, `Upright U must have 0 score for sideways H`);
  });

  test('recognizes H sign when index and middle point SIDEWAYS together, never mistaking for U', () => {
    const mockHand = createMockHand('sideways_H');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'H');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for H, got ${res.confidence}`);
    assert.equal(res.scores.U, 0, `Sideways H must have 0 score for upright U`);
  });

  test('recognizes V sign geometrically when spread apart', () => {
    const mockHand = createMockHand('peace_V');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'V');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for V, got ${res.confidence}`);
  });

  test('recognizes L sign geometrically', () => {
    const mockHand = createMockHand('L_sign');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'L');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80, got ${res.confidence}`);
  });

  test('recognizes I sign geometrically', () => {
    const mockHand = createMockHand('pinky_I');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'I');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80, got ${res.confidence}`);
  });

  test('recognizes S sign with fist and thumb crossed in front, never mistaking for O or A', () => {
    const mockHand = createMockHand('fist_S');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'S');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for S, got ${res.confidence}`);
    assert.equal(res.scores.O, 0, `Closed fist S must have 0 score for circular O`);
    assert.equal(res.scores.A, 0, `Crossed thumb S must have 0 score for upright A`);
  });

  test('recognizes O sign when hand is turned sideways (profile aperture), never mistaking for S', () => {
    const mockHand = createMockHand('circle_O');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'O');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for sideways O, got ${res.confidence}`);
    assert.equal(res.scores.S, 0, `Open circular O must have 0 score for closed fist S`);
  });

  test('recognizes O sign when palm is facing towards camera (frontal view), never mistaking for S', () => {
    const mockHand = createMockHand('circle_O_frontal');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'O');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for frontal O, got ${res.confidence}`);
    assert.equal(res.scores.S, 0, `Frontal O must have 0 score for closed fist S`);
  });

  test('recognizes A sign with thumb straight up alongside index finger, never mistaking for S', () => {
    const mockHand = createMockHand('fist_A');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'A');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for A, got ${res.confidence}`);
    assert.equal(res.scores.S, 0, `Upright thumb A must have 0 score for crossed thumb S`);
  });

  test('recognizes W sign with index, middle, and ring fingers upright, pinky curled, never mistaking for V or B', () => {
    const mockHand = createMockHand('three_W');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'W');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for W, got ${res.confidence}`);
    assert.equal(res.scores.V, 0, `3-finger W must have 0 score for 2-finger V`);
    assert.equal(res.scores.B, 0, `Pinky-curled W must have 0 score for 4-finger B`);
  });

  test('recognizes D sign with index finger UP and thumb touching middle finger, never mistaking for L', () => {
    const mockHand = createMockHand('index_D');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'D');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for D, got ${res.confidence}`);
    assert.equal(res.scores.L, 0, `D sign must have 0 score for L`);
  });

  test('recognizes F sign with middle, ring, pinky extended UP and thumb-index touching, never mistaking for B or W', () => {
    const mockHand = createMockHand('ok_F');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'F');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for F, got ${res.confidence}`);
    assert.equal(res.scores.B, 0, `F sign must have 0 score for B`);
    assert.equal(res.scores.W, 0, `F sign must have 0 score for W`);
  });

  test('recognizes G sign with 3 fingers completely folded and index-thumb pointing sideways, never mistaking for H or L', () => {
    const mockHand = createMockHand('sideways_G');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'G');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for G, got ${res.confidence}`);
    assert.equal(res.scores.H, 0, `G sign must have 0 score for H`);
    assert.equal(res.scores.L, 0, `G sign must have 0 score for L`);
  });

  test('recognizes X sign with index finger standing but bent halfway into a hook, 3 fingers folded, thumb touching middle, never mistaking for D', () => {
    const mockHand = createMockHand('hook_X');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'X');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for X, got ${res.confidence}`);
    assert.equal(res.scores.D, 0, `Hooked X must have 0 score for straight-up D`);
  });

  test('recognizes K sign with index and middle pointing UP into the sky, thumb placed between them, never mistaking for V', () => {
    const mockHand = createMockHand('peace_K');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'K');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for K, got ${res.confidence}`);
    assert.equal(res.scores.V, 0, `K sign must override/zero out V score when thumb is between index & middle`);
  });

  test('recognizes T sign with fist and thumb placed between index and middle fingers, never mistaking for S, A, or N', () => {
    const mockHand = createMockHand('fist_T');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'T');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for T, got ${res.confidence}`);
    assert.equal(res.scores.S, 0, `T sign must zero out S`);
    assert.equal(res.scores.A, 0, `T sign must zero out A`);
    assert.equal(res.scores.N, 0, `T sign must not match N`);
  });

  test('recognizes N sign with fist and thumb placed between middle and ring fingers, never mistaking for S, A, or T', () => {
    const mockHand = createMockHand('fist_N');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'N');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for N, got ${res.confidence}`);
    assert.equal(res.scores.S, 0, `N sign must zero out S`);
    assert.equal(res.scores.A, 0, `N sign must zero out A`);
    assert.equal(res.scores.T, 0, `N sign must not match T`);
  });

  test('recognizes M sign with fist and thumb placed between ring and pinky fingers, never mistaking for N, T, S, or A', () => {
    const mockHand = createMockHand('fist_M');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'M');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for M, got ${res.confidence}`);
    assert.equal(res.scores.N, 0, `M sign must zero out N`);
    assert.equal(res.scores.T, 0, `M sign must zero out T`);
    assert.equal(res.scores.S, 0, `M sign must zero out S`);
    assert.equal(res.scores.A, 0, `M sign must zero out A`);
  });

  test('recognizes R sign with index & middle crossed (middle behind index) and thumb touching ring/pinky, never mistaking for U or V', () => {
    const mockHand = createMockHand('crossed_R');
    const res = classifyASL(mockHand);
    assert.equal(res.letter, 'R');
    assert.ok(res.confidence >= 0.80, `Expected conf >= 0.80 for R, got ${res.confidence}`);
    assert.equal(res.scores.U, 0, `Crossed R must zero out U`);
    assert.equal(res.scores.V, 0, `Crossed R must zero out V`);
  });

  test('defines exactly the 21 core signs including M, R, K, T, N, G, X, D, F, S, U, and W', () => {
    assert.deepEqual(CORE_SIGNS, ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']);
  });

  test('detects 5-finger open hand gesture for space trigger', () => {
    const mockHand = createMockHand('open_palm');
    const res = classifyASL(mockHand);
    assert.equal(res.isOpenPalmFive, true);
    assert.equal(res.letter, 'SPACE');
    assert.ok(res.confidence >= 0.90, `Expected conf >= 0.90, got ${res.confidence}`);
  });

  test('recognizes thumb extension with lowered threshold comfortably', () => {
    // Hand where thumb is comfortably extended at moderate distance (dThumbWrist ~ 0.95)
    const pts = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.8, z: 0 }));
    pts[0] = { x: 0.5, y: 0.8, z: 0 };
    pts[9] = { x: 0.5, y: 0.5, z: 0 }; // scale = 0.3
    pts[2] = { x: 0.42, y: 0.72, z: 0 };
    pts[3] = { x: 0.35, y: 0.67, z: 0 };
    pts[4] = { x: 0.28, y: 0.62, z: 0 };
    const res = classifyASL(pts);
    assert.equal(res.fingerStates.thumb, true);
  });
});
