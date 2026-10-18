import React, { useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw
} from 'lucide-react';
import { EditorClip } from '../../types/editor';

interface EditorPreviewPlayerProps {
  currentTime: number;
  totalDuration: number;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
  onStepFrame: (direction: -1 | 1) => void;
  onResetTime: () => void;
  activeClip: EditorClip | null;
  fps?: number;
}

export const EditorPreviewPlayer: React.FC<EditorPreviewPlayerProps> = ({
  currentTime,
  totalDuration,
  isPlaying,
  onTogglePlay,
  onSeek,
  onStepFrame,
  onResetTime,
  activeClip,
  fps = 16,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const hundredths = Math.floor((seconds % 1) * 100);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  };

  const currentFrame = Math.floor(currentTime * fps);
  const totalFrames = Math.max(1, Math.floor(totalDuration * fps));

  // Pure canvas rendering of the video scene (NO text drawn on top of the video)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Background gradient based on active clip or studio dark
    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    if (activeClip) {
      const seedHash = Math.abs(Math.sin(activeClip.seed)) * 360;
      bgGrad.addColorStop(0, `hsl(${seedHash % 360}, 45%, 11%)`);
      bgGrad.addColorStop(0.5, `hsl(${(seedHash + 40) % 360}, 38%, 7%)`);
      bgGrad.addColorStop(1, '#020617');
    } else {
      bgGrad.addColorStop(0, '#0f172a');
      bgGrad.addColorStop(1, '#020617');
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // Subtle film grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (activeClip) {
      const clipElapsed = Math.max(0, currentTime - activeClip.startTime);
      const progressInClip = Math.min(1, Math.max(0, clipElapsed / activeClip.duration));
      const t = progressInClip * Math.PI * 2;

      ctx.save();
      const cx = width / 2 + Math.sin(t) * 90;
      const cy = height / 2 + Math.cos(t * 1.5) * 45;

      // Glow halo
      const radialGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 200);
      radialGrad.addColorStop(0, 'rgba(56, 189, 248, 0.4)');
      radialGrad.addColorStop(0.4, 'rgba(6, 182, 212, 0.15)');
      radialGrad.addColorStop(1, 'rgba(2, 6, 23, 0)');
      ctx.fillStyle = radialGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, 200, 0, Math.PI * 2);
      ctx.fill();

      // Subject symbol
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.fill();

      // Dynamic cinematic rings
      ctx.strokeStyle = 'rgba(103, 232, 249, 0.45)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, 70 + Math.sin(t * 2) * 12, 35 + Math.cos(t * 2) * 12, t, 0, Math.PI * 2);
      ctx.stroke();

      ctx.restore();
    }
  }, [currentTime, activeClip, fps]);

  return (
    <div className="w-full h-full flex flex-col justify-between items-center p-2 sm:p-3 overflow-hidden bg-slate-950 min-w-[360px] select-none">
      {/* 16:9 Centered Screen Viewport with ONLY center play button overlay */}
      <div className="flex-1 w-full min-h-0 flex items-center justify-center relative p-1">
        <div className="relative w-full max-w-full max-h-full aspect-video bg-black rounded-lg overflow-hidden border border-slate-800 shadow-xl flex items-center justify-center group">
          <canvas
            ref={canvasRef}
            width={854}
            height={480}
            className="w-full h-full object-contain"
          />

          {/* Central Play/Pause Overlay: ONLY overlay allowed on top of video, shown on hover or when paused */}
          <button
            onClick={onTogglePlay}
            className={`absolute inset-0 flex items-center justify-center cursor-pointer transition-opacity z-10 ${
              isPlaying
                ? 'opacity-0 group-hover:opacity-100 bg-black/10'
                : 'opacity-100 bg-black/25'
            }`}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            aria-label={isPlaying ? 'Pause' : 'Play'}
          >
            <div className="w-12 h-12 rounded-full bg-cyan-400/90 hover:bg-cyan-300 text-slate-950 flex items-center justify-center shadow-xl shadow-cyan-950/60 transition-transform active:scale-95">
              {isPlaying ? (
                <Pause className="w-5 h-5 fill-slate-950" />
              ) : (
                <Play className="w-5 h-5 fill-slate-950 ml-0.5" />
              )}
            </div>
          </button>
        </div>
      </div>

      {/* Thin Status Line directly below video, above play controls */}
      <div className="w-full max-w-2xl px-2 pt-1 pb-0.5 flex items-center justify-between text-[11px] text-slate-400 font-mono select-none overflow-hidden">
        <div className="truncate whitespace-nowrap">
          <span className="hidden sm:inline text-slate-500">16:9 · 480p · {fps} FPS · </span>
          <span className="text-slate-300 font-medium">Frame {currentFrame} / {totalFrames}</span>
        </div>
        <div className="text-[10px] text-slate-500 truncate ml-2 font-sans shrink-0">
          {activeClip ? activeClip.title : 'Ready'}
        </div>
      </div>

      {/* Control Bar Underneath */}
      <div className="w-full max-w-2xl px-2 pb-0.5 shrink-0 space-y-1.5">
        {/* Scrub Bar Slider */}
        <div className="relative flex items-center">
          <input
            type="range"
            min={0}
            max={Math.max(1, totalDuration)}
            step={1 / fps}
            value={currentTime}
            onChange={(e) => onSeek(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-800 hover:h-1.5 rounded-lg appearance-none cursor-pointer accent-cyan-400 transition-all focus:outline-none"
          />
        </div>

        {/* Playback Buttons & Timestamp Display */}
        <div className="flex items-center justify-between gap-2 text-xs">
          {/* Left: Reset / Step Back */}
          <div className="flex items-center gap-0.5">
            <button
              onClick={onResetTime}
              className="p-1 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded transition-colors"
              title="Return to start (Home)"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onStepFrame(-1)}
              className="p-1 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded transition-colors"
              title="Step frame back (Left Arrow)"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Center: Play / Pause */}
          <button
            onClick={onTogglePlay}
            className="w-8 h-8 rounded-full bg-cyan-400 hover:bg-cyan-300 text-slate-950 flex items-center justify-center shadow-md shadow-cyan-950/60 active:scale-95 transition-all"
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          >
            {isPlaying ? (
              <Pause className="w-3.5 h-3.5 fill-slate-950" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-slate-950 ml-0.5" />
            )}
          </button>

          {/* Right: Step Forward & Time Code */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => onStepFrame(1)}
              className="p-1 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded transition-colors"
              title="Step frame forward (Right Arrow)"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>

            {/* Time display: MM:SS.ff / Total */}
            <div className="font-mono text-[11px] tabular-nums bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-300">
              <span className="text-cyan-400 font-semibold">{formatTime(currentTime)}</span>
              <span className="text-slate-600 mx-1">/</span>
              <span className="text-slate-400">{formatTime(totalDuration)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
