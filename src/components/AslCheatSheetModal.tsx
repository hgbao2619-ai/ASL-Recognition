import React from 'react';
import { CORE_SIGNS, SIGN_DESCRIPTIONS } from '../utils/aslClassifier';
import { BookOpen, X, Sparkles, CheckCircle2 } from 'lucide-react';

interface AslCheatSheetModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentLetter: string | null;
  onSelectSignToSimulate?: (letter: string) => void;
}

export const AslCheatSheetModal: React.FC<AslCheatSheetModalProps> = ({
  isOpen,
  onClose,
  currentLetter,
  onSelectSignToSimulate,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md">
      <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-[#121212] border border-neutral-800 rounded-2xl shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-[#0e0e0e]">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-wide">
                ASL SIGNS REFERENCE GUIDE
              </h2>
              <p className="text-xs text-neutral-400 font-mono">
                Invariant geometric classification for core communication signs
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-neutral-400 hover:text-white rounded-lg hover:bg-neutral-800 transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Grid of Signs */}
        <div className="flex-1 overflow-auto p-6 bg-[#0a0a0a]">
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-3 gap-4">
            {CORE_SIGNS.map((sign) => {
              const info = SIGN_DESCRIPTIONS[sign];
              const isMatch = currentLetter === sign;

              return (
                <div
                  key={sign}
                  className={`relative p-4 rounded-xl border transition-all flex flex-col justify-between ${
                    isMatch
                      ? 'bg-cyan-950/40 border-cyan-400 shadow-lg shadow-cyan-500/20 ring-1 ring-cyan-400'
                      : 'bg-[#151515] border-neutral-800 hover:border-neutral-700'
                  }`}
                >
                  {isMatch && (
                    <div className="absolute top-2 right-2 flex items-center gap-1 text-[10px] font-mono font-bold text-cyan-300 bg-cyan-950/80 px-1.5 py-0.5 rounded border border-cyan-500/40">
                      <CheckCircle2 className="w-3 h-3 text-cyan-400" />
                      LIVE
                    </div>
                  )}

                  <div>
                    {/* Big Letter Display */}
                    <div className="flex items-baseline gap-2 mb-2">
                      <span className="text-4xl font-black font-mono text-white tracking-tighter">
                        {sign}
                      </span>
                      <span className="text-xs font-medium text-neutral-400 truncate">
                        {info?.title}
                      </span>
                    </div>

                    {/* English instructions */}
                    <p className="text-xs text-neutral-300 leading-snug mb-2 font-sans">
                      {info?.en}
                    </p>
                  </div>

                  <div className="pt-2 border-t border-neutral-800/80 mt-2">
                    {onSelectSignToSimulate && (
                      <button
                        onClick={() => onSelectSignToSimulate(sign)}
                        className="w-full py-1.5 px-2 rounded bg-neutral-800 hover:bg-cyan-500 hover:text-black text-neutral-300 text-[11px] font-mono transition flex items-center justify-center gap-1"
                      >
                        <Sparkles className="w-3 h-3" />
                        Test Sign {sign}
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* 5-Finger Space Gesture Feature Highlight */}
          <div className="mt-6 p-4 rounded-xl bg-emerald-950/30 border border-emerald-500/30 text-xs text-neutral-300 flex items-start gap-3">
            <span className="text-2xl">🖐</span>
            <div>
              <div className="font-bold text-emerald-400 mb-0.5 flex items-center gap-1.5">
                <span>5-FINGER SPACE GESTURE:</span>
                <span className="px-2 py-0.5 rounded bg-emerald-900/60 border border-emerald-500/40 text-[10px] text-emerald-300 font-mono">
                  HOLD 5 SECONDS
                </span>
              </div>
              <p className="text-neutral-300 leading-relaxed">
                Giơ cả 5 ngón tay mở (bàn tay mở tự nhiên) và giữ yên trong <span className="font-semibold text-emerald-400">5 giây</span> liên tục để tự động chèn một khoảng trắng <span className="font-mono bg-neutral-900 px-1.5 py-0.5 rounded border border-neutral-700 text-emerald-300">[ SPACE ]</span> phân tách giữa các từ.
              </p>
            </div>
          </div>

          {/* Quick tip box */}
          <div className="mt-3 p-4 rounded-xl bg-neutral-900/60 border border-neutral-800 text-xs text-neutral-400">
            <span className="font-semibold text-cyan-400">Tip for optimal recognition: </span>
            Keep your hand facing forward towards the camera at a comfortable distance (0.5m – 2.5m). Scale and translation invariant normalization ensures distance or hand positioning will not affect detection accuracy.
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-3 border-t border-neutral-800 bg-[#0e0e0e]">
          <span className="text-xs text-neutral-500">
            Hold signs for 10–15 frames (&gt;80%) | Hold 5 fingers for 5s to insert Space
          </span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-neutral-800 hover:bg-neutral-700 text-white text-xs rounded-lg transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
