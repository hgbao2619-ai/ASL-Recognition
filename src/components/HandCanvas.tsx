import React, { useEffect, useRef, useState } from 'react';
import { Landmark3D, classifyASL, PredictionResult } from '../utils/aslClassifier';

// MediaPipe Hands connections pairs
const HAND_CONNECTIONS = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm base
  [5, 9], [9, 13], [13, 17],
];

interface HandCanvasProps {
  onFramePrediction: (pred: PredictionResult, landmarks: Landmark3D[] | null) => void;
  isPaused?: boolean;
}

export const HandCanvas: React.FC<HandCanvasProps> = ({
  onFramePrediction,
  isPaused = false,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [cameraActive, setCameraActive] = useState<boolean>(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [fps, setFps] = useState<number>(0);
  const [isMediaPipeReady, setIsMediaPipeReady] = useState<boolean>(false);

  const handsInstanceRef = useRef<any>(null);
  const cameraInstanceRef = useRef<any>(null);
  const lastTimeRef = useRef<number>(performance.now());
  const frameCountRef = useRef<number>(0);

  // Initialize MediaPipe Hands via window.Hands
  useEffect(() => {
    let checkTimer: any = null;

    const initMediaPipe = () => {
      const w = window as any;
      if (w.Hands && w.Camera) {
        try {
          const hands = new w.Hands({
            locateFile: (file: string) => {
              return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            },
          });

          hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.65,
            minTrackingConfidence: 0.65,
          });

          hands.onResults((results: any) => {
            if (isPaused) return;

            // Update FPS counter
            frameCountRef.current++;
            const now = performance.now();
            if (now - lastTimeRef.current >= 1000) {
              setFps(frameCountRef.current);
              frameCountRef.current = 0;
              lastTimeRef.current = now;
            }

            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            // Synchronize dimensions
            if (videoRef.current && videoRef.current.videoWidth > 0) {
              if (canvas.width !== videoRef.current.videoWidth) {
                canvas.width = videoRef.current.videoWidth;
                canvas.height = videoRef.current.videoHeight;
              }
            }

            ctx.save();
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // 1. Horizontal mirror transform: cv2.flip(frame, 1) equivalent
            ctx.scale(-1, 1);
            ctx.translate(-canvas.width, 0);

            if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
              const landmarks = results.multiHandLandmarks[0] as Landmark3D[];

              // Classify hand
              const pred = classifyASL(landmarks);
              onFramePrediction(pred, landmarks);

              // Draw minimal skeleton
              drawMinimalistSkeleton(ctx, landmarks, canvas.width, canvas.height, pred.letter);
            } else {
              onFramePrediction(
                {
                  letter: null,
                  confidence: 0,
                  scores: {},
                  features: null,
                  fingerStates: { thumb: false, index: false, middle: false, ring: false, pinky: false },
                  isOpenPalmFive: false,
                },
                null
              );
            }

            ctx.restore();
          });

          handsInstanceRef.current = hands;
          setIsMediaPipeReady(true);
        } catch (err: any) {
          console.error('Error initializing MediaPipe:', err);
          setCameraError('Unable to initialize MediaPipe: ' + err.message);
        }
      } else {
        checkTimer = setTimeout(initMediaPipe, 200);
      }
    };

    initMediaPipe();

    return () => {
      if (checkTimer) clearTimeout(checkTimer);
      if (handsInstanceRef.current) {
        try {
          handsInstanceRef.current.close();
        } catch {}
      }
    };
  }, [isPaused, onFramePrediction]);

  // Start Webcam
  useEffect(() => {
    if (!isMediaPipeReady || !videoRef.current) return;

    const w = window as any;
    const video = videoRef.current;

    // Check if mediaDevices exists
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCameraError('Browser does not support getUserMedia or webcam permissions are restricted.');
      return;
    }

    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          setCameraActive(true);
          setCameraError(null);
        };

        // Process frames with requestAnimationFrame or interval
        let isProcessing = false;
        const processLoop = async () => {
          if (!handsInstanceRef.current || isPaused) {
            animFrameId = requestAnimationFrame(processLoop);
            return;
          }

          if (!isProcessing && video.readyState >= 2) {
            isProcessing = true;
            try {
              await handsInstanceRef.current.send({ image: video });
            } catch (err) {
              // Frame send error
            } finally {
              isProcessing = false;
            }
          }
          animFrameId = requestAnimationFrame(processLoop);
        };

        let animFrameId = requestAnimationFrame(processLoop);
        cameraInstanceRef.current = {
          stop: () => {
            cancelAnimationFrame(animFrameId);
            stream.getTracks().forEach((track) => track.stop());
          },
        };
      })
      .catch((err: any) => {
        console.warn('Camera access unavailable:', err?.message || err);
        setCameraError(
          'Webcam device not detected or permission was not granted. You can use the Quick Sign Simulator below to test the full pipeline!'
        );
      });

    return () => {
      if (cameraInstanceRef.current) {
        try {
          cameraInstanceRef.current.stop();
        } catch {}
      }
    };
  }, [isMediaPipeReady, isPaused]);

  // Draw sleek minimalist skeleton
  const drawMinimalistSkeleton = (
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark3D[],
    width: number,
    height: number,
    letter: string | null
  ) => {
    // Convert to pixel coords
    const pts = landmarks.map((lm) => ({
      x: lm.x * width,
      y: lm.y * height,
    }));

    // 1. Draw connection lines (Thin neon cyan)
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#00f0ff';
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 8;

    for (const [i, j] of HAND_CONNECTIONS) {
      const p1 = pts[i];
      const p2 = pts[j];
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Reset shadow for joints
    ctx.shadowBlur = 0;

    // 2. Draw Joints (Clean white circles)
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const isTip = [4, 8, 12, 16, 20].includes(i);
      const isWrist = i === 0;

      ctx.beginPath();
      if (isWrist) {
        ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#00f0ff';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      } else if (isTip) {
        ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#00f0ff';
        ctx.stroke();
      } else {
        ctx.arc(p.x, p.y, 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
      }
    }
  };

  return (
    <div className="relative w-full h-full min-h-[420px] max-h-[640px] flex items-center justify-center bg-[#0d0d0d] rounded-2xl overflow-hidden border border-neutral-800 shadow-2xl">
      {/* Video stream element (hidden, mirrored) */}
      <video
        ref={videoRef}
        playsInline
        muted
        className="w-full h-full object-cover -scale-x-100 opacity-80 filter brightness-95 contrast-105"
      />

      {/* MediaPipe Skeleton Overlay Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
      />

      {/* Grid crosshair backdrop for high-tech aesthetic */}
      <div className="absolute inset-0 pointer-events-none bg-[radial-gradient(#22d3ee08_1px,transparent_1px)] [background-size:24px_24px] opacity-60" />

      {/* Minimal Corner Frame Markers */}
      <div className="absolute top-4 left-4 w-6 h-6 border-t-2 border-l-2 border-cyan-400/50 rounded-tl-sm pointer-events-none" />
      <div className="absolute top-4 right-4 w-6 h-6 border-t-2 border-r-2 border-cyan-400/50 rounded-tr-sm pointer-events-none" />
      <div className="absolute bottom-4 left-4 w-6 h-6 border-b-2 border-l-2 border-cyan-400/50 rounded-bl-sm pointer-events-none" />
      <div className="absolute bottom-4 right-4 w-6 h-6 border-b-2 border-r-2 border-cyan-400/50 rounded-br-sm pointer-events-none" />

      {/* Top Overlay Badge: FPS & Status */}
      <div className="absolute top-4 left-6 flex items-center gap-2 px-3 py-1 bg-black/60 backdrop-blur-md border border-neutral-800 rounded-full text-xs font-mono text-neutral-400">
        <span
          className={`w-2 h-2 rounded-full ${
            cameraActive ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
          }`}
        />
        <span>{cameraActive ? `LIVE WEBCAM • ${fps} FPS` : 'INITIALIZING SENSOR...'}</span>
      </div>

      {/* Camera permission fallback or loading alert */}
      {cameraError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center p-6 bg-black/85 backdrop-blur-md text-center z-10">
          <div className="w-12 h-12 rounded-full bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 mb-3 text-xl font-bold">
            !
          </div>
          <h3 className="text-white font-semibold text-lg mb-1">Webcam Access</h3>
          <p className="text-neutral-400 text-sm max-w-md mb-4">{cameraError}</p>
          <div className="flex gap-3">
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-neutral-800 hover:bg-neutral-700 text-white text-xs font-medium rounded-lg transition"
            >
              Retry Camera
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
