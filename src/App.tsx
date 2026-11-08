import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Volume2,
  VolumeX,
  Sparkles,
  RotateCcw,
  Delete,
  BookOpen,
  Keyboard,
  Layers,
  Loader2,
} from 'lucide-react';
import { HandCanvas } from './components/HandCanvas';
import { AslCheatSheetModal } from './components/AslCheatSheetModal';
import { PredictionResult, CORE_SIGNS } from './utils/aslClassifier';
import { TTSService, soundFx } from './utils/soundTts';

// Debounce Buffer Configuration
const REQUIRED_FRAMES = 12; // 10 - 15 consecutive frames
const CONFIDENCE_THRESHOLD = 0.80; // Confidence > 80%
const COOLDOWN_DURATION = 1000; // 1.0s cooldown after confirm
const SPACE_HOLD_DURATION = 5000; // 5.0 seconds hold for 5-finger space gesture

export default function App() {
  // Live detection states
  const [liveChar, setLiveChar] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [isOpenPalm, setIsOpenPalm] = useState<boolean>(false);
  const [fingerStates, setFingerStates] = useState({
    thumb: false,
    index: false,
    middle: false,
    ring: false,
    pinky: false,
  });

  // Debounce Buffer State
  const [buffer, setBuffer] = useState<string[]>([]);
  const [holdProgress, setHoldProgress] = useState<number>(0);
  const [cooldownRemaining, setCooldownRemaining] = useState<number>(0);

  // 5-Finger Space Gesture State (Hold 5 fingers for 5 seconds)
  const [spaceProgress, setSpaceProgress] = useState<number>(0);
  const [spaceRemaining, setSpaceRemaining] = useState<number>(0);
  const [spaceSuccessToast, setSpaceSuccessToast] = useState<boolean>(false);
  const spaceStartTimeRef = useRef<number>(0);
  const spaceConfirmedRef = useRef<boolean>(false);

  // Gemini & Final Sentence State
  const [finalSentence, setFinalSentence] = useState<string>('');
  const [isGeminiLoading, setIsGeminiLoading] = useState<boolean>(false);
  const [geminiSource, setGeminiSource] = useState<string>('gemini-3.1-flash-lite');
  const [targetLanguage, setTargetLanguage] = useState<'vi' | 'en'>('vi');

  // TTS & Audio State
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(true);

  // Modals & Panels
  const [isCheatSheetOpen, setIsCheatSheetOpen] = useState<boolean>(false);
  const [isSimulatorOpen, setIsSimulatorOpen] = useState<boolean>(false);

  // Internal buffer tracking refs
  const historyRef = useRef<string[]>([]);
  const lastConfirmedTimeRef = useRef<number>(0);

  // Cooldown timer tick
  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = Date.now() - lastConfirmedTimeRef.current;
      if (elapsed < COOLDOWN_DURATION) {
        setCooldownRemaining(Math.max(0, (COOLDOWN_DURATION - elapsed) / 1000));
      } else {
        setCooldownRemaining(0);
      }
    }, 100);
    return () => clearInterval(interval);
  }, []);

  // Frame Prediction & Debounce Logic
  const handleFramePrediction = useCallback(
    (pred: PredictionResult) => {
      setLiveChar(pred.letter);
      setConfidence(pred.confidence);
      setFingerStates(pred.fingerStates);
      setIsOpenPalm(pred.isOpenPalmFive);

      const now = Date.now();

      // 1. FEATURE: 5-Finger Open Hand held for 5 seconds -> Insert Space ' '
      if (pred.isOpenPalmFive) {
        historyRef.current = [];
        setHoldProgress(0);

        if (!spaceConfirmedRef.current) {
          if (spaceStartTimeRef.current === 0) {
            spaceStartTimeRef.current = now;
          }
          const elapsed = now - spaceStartTimeRef.current;
          const progress = Math.min(1.0, elapsed / SPACE_HOLD_DURATION);
          setSpaceProgress(progress);
          setSpaceRemaining(Math.max(0, (SPACE_HOLD_DURATION - elapsed) / 1000));

          if (elapsed >= SPACE_HOLD_DURATION) {
            // Append space character (avoid duplicate trailing spaces)
            setBuffer((prev) => {
              if (prev.length === 0 || prev[prev.length - 1] === ' ') {
                return prev;
              }
              return [...prev, ' '];
            });
            soundFx.playConfirmBlip();
            spaceConfirmedRef.current = true;
            setSpaceProgress(1.0);
            setSpaceRemaining(0);
            setSpaceSuccessToast(true);
            setTimeout(() => setSpaceSuccessToast(false), 2200);
          }
        }
        return;
      } else {
        // When hand is lowered or no longer has 5 fingers extended: reset 5-second space timer
        spaceStartTimeRef.current = 0;
        spaceConfirmedRef.current = false;
        setSpaceProgress(0);
        setSpaceRemaining(0);
      }

      // Ignore regular letter frame if currently in cooldown
      if (now - lastConfirmedTimeRef.current < COOLDOWN_DURATION) {
        historyRef.current = [];
        setHoldProgress(0);
        return;
      }

      // 2. Regular ASL Letter Debounce Buffer (12 consecutive frames > 80%)
      if (pred.letter && pred.letter !== 'SPACE' && pred.confidence >= CONFIDENCE_THRESHOLD) {
        const hist = historyRef.current;

        // If different from current sequence, reset history
        if (hist.length > 0 && hist[hist.length - 1] !== pred.letter) {
          historyRef.current = [pred.letter];
          setHoldProgress(1 / REQUIRED_FRAMES);
          return;
        }

        hist.push(pred.letter);
        setHoldProgress(Math.min(1, hist.length / REQUIRED_FRAMES));

        // When reached required consecutive frames
        if (hist.length >= REQUIRED_FRAMES) {
          const confirmedChar = pred.letter;
          setBuffer((prev) => [...prev, confirmedChar]);
          soundFx.playConfirmBlip();

          lastConfirmedTimeRef.current = Date.now();
          historyRef.current = [];
          setHoldProgress(0);
        }
      } else {
        // Lost track or shaking -> reset hold progress
        historyRef.current = [];
        setHoldProgress(0);
      }
    },
    []
  );

  // Trigger Gemini Flash Lite to reconstruct natural sentence from discrete characters
  const handleSynthesizeSentence = useCallback(async () => {
    if (buffer.length === 0 || isGeminiLoading) return;

    setIsGeminiLoading(true);
    try {
      const response = await fetch('/api/gemini/sentence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          characters: buffer,
          language: targetLanguage,
        }),
      });

      const data = await response.json();
      if (data.sentence) {
        setFinalSentence(data.sentence);
        setGeminiSource(data.source || 'gemini-3.1-flash-lite');
        soundFx.playSuccessChime();

        // Speak via Text-to-Speech in selected language
        if (ttsEnabled) {
          speakText(data.sentence, targetLanguage === 'vi' ? 'vi-VN' : 'en-US');
        }
      }
    } catch (err) {
      console.error('Gemini Flash Lite synthesis error:', err);
      const clean = buffer.join('').replace(/\s+/g, '').toUpperCase();
      if (clean === 'ILY') {
        setFinalSentence(targetLanguage === 'vi' ? 'Tôi yêu bạn rất nhiều.' : 'I love you very much.');
      } else {
        const joined = buffer.join('');
        setFinalSentence(`${joined.charAt(0).toUpperCase() + joined.slice(1).toLowerCase()}.`);
      }
    } finally {
      setIsGeminiLoading(false);
    }
  }, [buffer, isGeminiLoading, ttsEnabled, targetLanguage]);

  // Non-blocking TTS via Web Speech API
  const speakText = (text: string, lang = 'vi-VN') => {
    if (!text.trim()) return;
    setIsSpeaking(true);
    TTSService.speak(text, {
      lang,
      rate: 1.0,
      pitch: 1.0,
      onStart: () => setIsSpeaking(true),
      onEnd: () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  };

  // Clear buffer
  const handleClearBuffer = () => {
    setBuffer([]);
    setFinalSentence('');
    historyRef.current = [];
    setHoldProgress(0);
    soundFx.playClearTone();
    TTSService.stop();
    setIsSpeaking(false);
  };

  // Delete last character
  const handleRemoveLast = () => {
    setBuffer((prev) => prev.slice(0, -1));
    historyRef.current = [];
    setHoldProgress(0);
  };

  // Manual character append (for simulator)
  const handleAddManualChar = (char: string) => {
    setBuffer((prev) => [...prev, char]);
    soundFx.playConfirmBlip();
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.tagName === 'INPUT') return;

      if (e.code === 'Space') {
        e.preventDefault();
        handleSynthesizeSentence();
      } else if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        handleClearBuffer();
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        handleRemoveLast();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSynthesizeSentence]);

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white flex flex-col font-sans select-none overflow-x-hidden">
      {/* 1. TOP HEADER - CLEAN & MINIMALIST (NO REDUNDANT TEXT) */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-neutral-900 bg-[#0d0d0d]/90 backdrop-blur-md sticky top-0 z-40">
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-9 h-9 rounded-lg bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 font-black font-mono">
            <span>ASL</span>
            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-wider text-white">
              ASL RECOGNITION
            </h1>
            <p className="text-xs text-neutral-400 font-mono">
              Real-time American Sign Language Recognition
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* TTS Audio Toggle */}
          <button
            onClick={() => setTtsEnabled(!ttsEnabled)}
            className={`p-2 rounded-lg border text-xs transition ${
              ttsEnabled
                ? 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
                : 'bg-neutral-900 border-neutral-800 text-neutral-500'
            }`}
            title={ttsEnabled ? 'Audio TTS: ON' : 'Audio TTS: OFF'}
          >
            {ttsEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* ASL Signs Reference Guide Button */}
          <button
            onClick={() => setIsCheatSheetOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neutral-900 border border-neutral-800 hover:border-neutral-700 text-xs text-neutral-200 transition"
          >
            <BookOpen className="w-3.5 h-3.5 text-cyan-400" />
            <span className="hidden md:inline font-medium">ASL Signs Guide</span>
          </button>
        </div>
      </header>

      {/* 2. MAIN BODY - MINIMALIST WORKSPACE */}
      <main className="flex-1 flex flex-col p-4 sm:p-6 max-w-7xl w-full mx-auto gap-4">
        {/* UPPER ROW: WEBCAM + LIVE DETECTION DASHBOARD */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">
          {/* WEBCAM CENTRAL FRAME */}
          <div className="lg:col-span-7 flex flex-col min-h-[380px] sm:min-h-[460px]">
            <HandCanvas onFramePrediction={handleFramePrediction} />
          </div>

          {/* LIVE DETECTION & STATUS (LARGE TYPOGRAPHY READABLE FROM 2-3 METERS) */}
          <div className="lg:col-span-5 flex flex-col justify-between gap-4">
            {/* CARD 1: LIVE DETECTION BIG CHARACTER */}
            <div className="relative flex-1 p-6 bg-[#121212] border border-neutral-800 rounded-2xl flex flex-col justify-between overflow-hidden shadow-xl">
              {/* Top Card Label */}
              <div className="flex items-center justify-between text-xs font-mono text-neutral-400">
                <span className="flex items-center gap-1.5 font-bold">
                  <span className="w-2 h-2 rounded-full bg-cyan-400" />
                  LIVE DETECTION
                </span>
                <span>CURRENT SIGN</span>
              </div>

              {/* GIGANTIC CHARACTER OR 5-FINGER SPACE TIMER */}
              <div className="flex flex-col items-center justify-center my-2 min-h-[140px]">
                {isOpenPalm || spaceProgress > 0 ? (
                  <div className="flex flex-col items-center justify-center text-center">
                    <div className="flex items-center gap-2">
                      <span className="text-6xl sm:text-7xl">🖐</span>
                      <span className="text-4xl sm:text-5xl font-mono font-bold text-emerald-400">␣</span>
                    </div>

                    <div className="mt-2 text-xs font-mono font-semibold">
                      {spaceSuccessToast || (spaceRemaining <= 0 && spaceProgress >= 1) ? (
                        <span className="text-emerald-300 bg-emerald-950/90 px-3 py-1 rounded-full border border-emerald-500/60 animate-pulse">
                          ✓ SPACE INSERTED (LOWER HAND TO RESET)
                        </span>
                      ) : (
                        <span className="text-neutral-300 bg-neutral-900/90 px-3 py-1 rounded-full border border-neutral-700">
                          HOLD 5s FOR SPACE: {(5 - spaceRemaining).toFixed(1)}s / 5.0s ({Math.round(spaceProgress * 100)}%)
                        </span>
                      )}
                    </div>

                    {/* 5-second Progress Bar */}
                    <div className="w-56 h-2.5 bg-neutral-900 rounded-full mt-2.5 overflow-hidden border border-neutral-800">
                      <div
                        className="h-full bg-emerald-400 shadow-[0_0_14px_#34d399] transition-all duration-75"
                        style={{ width: `${spaceProgress * 100}%` }}
                      />
                    </div>
                  </div>
                ) : (
                  <div className="relative">
                    <span
                      className={`text-8xl sm:text-9xl font-black font-mono tracking-tighter transition-all duration-150 ${
                        liveChar && confidence >= CONFIDENCE_THRESHOLD
                          ? 'text-cyan-400 drop-shadow-[0_0_35px_rgba(34,211,238,0.5)]'
                          : liveChar
                          ? 'text-white'
                          : 'text-neutral-700'
                      }`}
                    >
                      {liveChar || '-'}
                    </span>

                    {/* Hold-to-confirm Progress Ring */}
                    {holdProgress > 0 && (
                      <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 font-mono text-xs font-bold animate-pulse whitespace-nowrap">
                        HOLDING {Math.round(holdProgress * 100)}%
                      </div>
                    )}

                    {cooldownRemaining > 0 && (
                      <div className="absolute -bottom-3 left-1/2 -translate-x-1/2 px-2 py-0.5 rounded-full bg-cyan-500/20 border border-cyan-500/40 text-cyan-300 font-mono text-xs font-bold whitespace-nowrap">
                        COOLDOWN {cooldownRemaining.toFixed(1)}s
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Minimalist Confidence Bar */}
              <div className="space-y-1.5 pt-2 border-t border-neutral-800">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-neutral-400">
                    {isOpenPalm ? 'GESTURE: 5-FINGER SPACE' : 'CONFIDENCE'}
                  </span>
                  <span
                    className={`font-bold ${
                      isOpenPalm
                        ? 'text-emerald-400'
                        : confidence >= CONFIDENCE_THRESHOLD
                        ? 'text-cyan-400'
                        : 'text-neutral-300'
                    }`}
                  >
                    {isOpenPalm
                      ? `HOLD 5s: ${Math.round(spaceProgress * 100)}%`
                      : `${Math.round(confidence * 100)}%${
                          confidence >= CONFIDENCE_THRESHOLD ? ' (VALID > 80%)' : ''
                        }`}
                  </span>
                </div>
                <div className="w-full h-2 bg-neutral-900 rounded-full overflow-hidden border border-neutral-800">
                  <div
                    className={`h-full transition-all duration-100 ${
                      isOpenPalm
                        ? 'bg-emerald-400 shadow-[0_0_12px_#34d399]'
                        : confidence >= CONFIDENCE_THRESHOLD
                        ? 'bg-cyan-400 shadow-[0_0_12px_#00f0ff]'
                        : 'bg-neutral-600'
                    }`}
                    style={{
                      width: `${
                        isOpenPalm ? spaceProgress * 100 : Math.min(100, confidence * 100)
                      }%`,
                    }}
                  />
                </div>
              </div>

              {/* Finger States Quick Status */}
              <div className="grid grid-cols-5 gap-1 pt-3 text-center text-[10px] font-mono text-neutral-400">
                <div
                  className={`p-1 rounded ${
                    fingerStates.thumb ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50' : 'bg-neutral-900'
                  }`}
                >
                  THUMB
                </div>
                <div
                  className={`p-1 rounded ${
                    fingerStates.index ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50' : 'bg-neutral-900'
                  }`}
                >
                  INDEX
                </div>
                <div
                  className={`p-1 rounded ${
                    fingerStates.middle ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50' : 'bg-neutral-900'
                  }`}
                >
                  MIDDLE
                </div>
                <div
                  className={`p-1 rounded ${
                    fingerStates.ring ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50' : 'bg-neutral-900'
                  }`}
                >
                  RING
                </div>
                <div
                  className={`p-1 rounded ${
                    fingerStates.pinky ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50' : 'bg-neutral-900'
                  }`}
                >
                  PINKY
                </div>
              </div>
            </div>

            {/* CARD 2: BUFFER TEXT */}
            <div className="p-5 bg-[#121212] border border-neutral-800 rounded-2xl flex flex-col justify-between shadow-xl">
              <div className="flex items-center justify-between text-xs font-mono text-neutral-400 mb-2">
                <span className="flex items-center gap-1.5 font-bold">
                  <Layers className="w-3.5 h-3.5 text-cyan-400" />
                  BUFFER TEXT ({buffer.length} CHARS)
                </span>
                <div className="flex gap-1.5">
                  <button
                    onClick={handleRemoveLast}
                    disabled={buffer.length === 0}
                    className="p-1 px-2.5 rounded bg-neutral-900 hover:bg-neutral-800 disabled:opacity-30 text-[11px] font-mono text-neutral-300 transition"
                    title="Delete character [BACKSPACE]"
                  >
                    <Delete className="w-3.5 h-3.5 inline mr-1" />
                    Delete
                  </button>
                  <button
                    onClick={handleClearBuffer}
                    disabled={buffer.length === 0}
                    className="p-1 px-2.5 rounded bg-neutral-900 hover:bg-red-950/60 hover:text-red-300 disabled:opacity-30 text-[11px] font-mono text-neutral-300 transition"
                    title="Clear buffer [C]"
                  >
                    <RotateCcw className="w-3.5 h-3.5 inline mr-1" />
                    Clear (C)
                  </button>
                </div>
              </div>

              {/* Buffer Chips Display */}
              <div className="min-h-[58px] p-3 bg-[#0a0a0a] rounded-xl border border-neutral-850 flex items-center gap-2 overflow-x-auto">
                {buffer.length > 0 ? (
                  buffer.map((ch, idx) => (
                    ch === ' ' ? (
                      <div
                        key={idx}
                        className="flex-shrink-0 px-2.5 h-11 rounded-lg bg-emerald-950/40 border border-emerald-500/40 text-emerald-300 font-mono text-xs font-bold flex items-center justify-center gap-1 shadow-md shadow-emerald-500/10 animate-in fade-in zoom-in-95 duration-150"
                        title="Space word separator"
                      >
                        <span className="text-sm">␣</span>
                        <span>SPACE</span>
                      </div>
                    ) : (
                      <div
                        key={idx}
                        className="flex-shrink-0 w-9 h-11 rounded-lg bg-neutral-900 border border-cyan-500/40 text-cyan-300 font-mono font-black text-xl flex items-center justify-center shadow-md shadow-cyan-500/5 animate-in fade-in zoom-in-95 duration-150"
                      >
                        {ch}
                      </div>
                    )
                  ))
                ) : (
                  <span className="text-neutral-500 text-xs font-mono italic">
                    [ Buffer empty. Hold hand before camera or click signs below ]
                  </span>
                )}
              </div>

              {/* Language Selector + Model Pill */}
              <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-neutral-800/80">
                <div className="flex items-center gap-1.5 text-neutral-400">
                  <span className="text-[11px] font-semibold">TARGET:</span>
                  <div className="inline-flex rounded-lg p-0.5 bg-neutral-900 border border-neutral-800">
                    <button
                      onClick={() => setTargetLanguage('vi')}
                      className={`px-2.5 py-0.5 rounded text-[11px] font-bold transition ${
                        targetLanguage === 'vi'
                          ? 'bg-cyan-500 text-black shadow-sm'
                          : 'text-neutral-400 hover:text-white'
                      }`}
                    >
                      Tiếng Việt
                    </button>
                    <button
                      onClick={() => setTargetLanguage('en')}
                      className={`px-2.5 py-0.5 rounded text-[11px] font-bold transition ${
                        targetLanguage === 'en'
                          ? 'bg-cyan-500 text-black shadow-sm'
                          : 'text-neutral-400 hover:text-white'
                      }`}
                    >
                      English
                    </button>
                  </div>
                </div>

                <span className="text-[10px] text-cyan-300/90 font-mono flex items-center gap-1 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-700/40">
                  <Sparkles className="w-3 h-3 text-cyan-400" />
                  AI Sentence Builder
                </span>
              </div>

              {/* Trigger Button */}
              <div className="mt-3 flex gap-2">
                <button
                  onClick={handleSynthesizeSentence}
                  disabled={buffer.length === 0 || isGeminiLoading}
                  className="flex-1 py-2.5 px-4 rounded-xl bg-cyan-500 hover:bg-cyan-400 disabled:bg-neutral-800 disabled:text-neutral-600 text-black font-bold text-xs font-mono uppercase tracking-wider flex items-center justify-center gap-2 transition shadow-lg shadow-cyan-500/20"
                >
                  {isGeminiLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      COMPOSING COMPLETE SENTENCE...
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      SYNTHESIZE COMPLETE SENTENCE [SPACE]
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* 3. FINAL SENTENCE & TTS SECTION (LARGE TYPOGRAPHY READABLE FROM 2-3 METERS) */}
        <div className="p-6 bg-[#121212] border border-neutral-800 rounded-2xl shadow-2xl relative overflow-hidden">
          <div className="flex items-center justify-between mb-2">
            <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-cyan-400 font-bold uppercase tracking-wider">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span>COMPLETE SENTENCE</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-900 text-neutral-300 border border-neutral-800 font-mono">
                {targetLanguage === 'vi' ? 'Tiếng Việt' : 'English'}
              </span>
            </div>

            {/* TTS Indicator */}
            <div className="flex items-center gap-2">
              {isSpeaking && (
                <div className="flex items-center gap-1 text-cyan-300 font-mono text-xs bg-cyan-950/70 px-2.5 py-1 rounded-full border border-cyan-500/40">
                  <span className="w-1.5 h-3 bg-cyan-400 animate-pulse" />
                  <span className="w-1.5 h-4 bg-cyan-400 animate-pulse delay-75" />
                  <span className="w-1.5 h-2 bg-cyan-400 animate-pulse delay-150" />
                  <span className="font-semibold ml-1">SPEAKING TTS...</span>
                </div>
              )}

              {finalSentence && (
                <button
                  onClick={() => speakText(finalSentence, targetLanguage === 'vi' ? 'vi-VN' : 'en-US')}
                  className="px-3 py-1 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-mono flex items-center gap-1.5 transition"
                >
                  <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
                  Replay
                </button>
              )}
            </div>
          </div>

          {/* GIGANTIC FINAL SENTENCE TEXT (READABLE FROM 2-3 METERS) */}
          <div className="min-h-[70px] flex items-center py-2">
            {finalSentence ? (
              <p className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-white tracking-tight leading-snug">
                &ldquo;{finalSentence}&rdquo;
              </p>
            ) : (
              <p className="text-base sm:text-lg text-neutral-500 italic">
                {targetLanguage === 'vi'
                  ? 'Bấm [SPACE] hoặc nút "Synthesize Complete Sentence" để chuyển chuỗi ký tự thành câu tiếng Việt hoàn chỉnh...'
                  : 'Press [SPACE] or click "Synthesize Complete Sentence" to convert discrete letters into a complete sentence...'}
              </p>
            )}
          </div>

          {/* Bottom Keyboard Guide Footnote */}
          <div className="pt-3 border-t border-neutral-850 flex flex-wrap items-center justify-between text-xs text-neutral-400 font-mono gap-2">
            <div className="flex items-center gap-3">
              <span>
                <kbd className="px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-white font-bold">
                  SPACE
                </kbd>{' '}
                Synthesize &amp; Speak
              </span>
              <span>
                <kbd className="px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-white font-bold">
                  C
                </kbd>{' '}
                Clear buffer
              </span>
              <span>
                <kbd className="px-1.5 py-0.5 rounded bg-neutral-800 border border-neutral-700 text-white font-bold">
                  BACKSPACE
                </kbd>{' '}
                Delete character
              </span>
            </div>

            <button
              onClick={() => setIsSimulatorOpen(!isSimulatorOpen)}
              className="text-cyan-400 hover:underline flex items-center gap-1"
            >
              <Keyboard className="w-3.5 h-3.5" />
              {isSimulatorOpen ? 'Hide simulator' : 'Toggle Sign Simulator'}
            </button>
          </div>
        </div>

        {/* 4. QUICK SIMULATOR PANEL (CORE SIGNS EXCLUDING 'E') */}
        {isSimulatorOpen && (
          <div className="p-4 bg-[#111111] border border-neutral-800 rounded-xl space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
            <div className="flex items-center justify-between text-xs font-mono text-neutral-400">
              <span className="font-semibold text-cyan-300">
                QUICK SIGN SIMULATOR:
              </span>
              <span>Click any sign to append to buffer</span>
            </div>

            {/* Core Signs Buttons */}
            <div className="flex flex-wrap items-center gap-2">
              {CORE_SIGNS.map((sign) => (
                <button
                  key={sign}
                  onClick={() => handleAddManualChar(sign)}
                  className="w-10 h-10 rounded-lg bg-neutral-800 hover:bg-cyan-500 hover:text-black text-white font-mono font-black text-lg transition flex items-center justify-center border border-neutral-700"
                >
                  {sign}
                </button>
              ))}

              {/* 5-Finger Space Simulator Button */}
              <button
                onClick={() => handleAddManualChar(' ')}
                className="h-10 px-3 rounded-lg bg-emerald-950/60 border border-emerald-500/40 text-emerald-300 hover:bg-emerald-500 hover:text-black font-mono font-bold text-xs transition flex items-center justify-center gap-1.5 shadow-sm"
                title="Add space character (simulates holding 5 fingers for 5s)"
              >
                <span>🖐 ␣</span>
                <span>+SPACE (5s)</span>
              </button>
            </div>

            {/* Quick Word Presets (Excluding any word containing 'E') */}
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-neutral-800 text-xs font-mono">
              <span className="text-neutral-500">Preset Words:</span>
              {['HI', 'ILY', 'YOU', 'DAY', 'GOOD', 'GO', 'FOX', 'BOX', 'FLY', 'FOOD', 'HOLD', 'BUY', 'CLUB', 'BOY', 'CALL', 'CHILL', 'OH', 'WOW', 'CAT', 'KID', 'THANK', 'NO', 'TIME', 'MOM', 'RUN', 'ROOM', 'RAIN', 'MAN'].map((word) => (
                <button
                  key={word}
                  onClick={() => {
                    word.split('').forEach((c) => handleAddManualChar(c));
                  }}
                  className="px-2.5 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-300 transition"
                >
                  +{word}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* 5. MODALS */}

      <AslCheatSheetModal
        isOpen={isCheatSheetOpen}
        onClose={() => setIsCheatSheetOpen(false)}
        currentLetter={liveChar}
        onSelectSignToSimulate={(sign) => {
          handleAddManualChar(sign);
          setIsCheatSheetOpen(false);
        }}
      />
    </div>
  );
}
