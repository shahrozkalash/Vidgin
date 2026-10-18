import React, { useRef, useEffect } from 'react';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  Maximize2,
  Volume2,
  Film,
  Sparkles
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

  // Format seconds to MM:SS.ff
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    const hundredths = Math.floor((seconds % 1) * 100);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(hundredths).padStart(2, '0')}`;
  };

  const currentFrame = Math.floor(currentTime * fps);
  const totalFrames = Math.max(1, Math.floor(totalDuration * fps));

  // Procedural canvas animation to render vivid preview frames based on activeClip and time
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Background gradient based on active clip or dark studio
    const bgGrad = ctx.createLinearGradient(0, 0, width, height);
    if (activeClip) {
      const seedHash = Math.abs(Math.sin(activeClip.seed)) * 360;
      bgGrad.addColorStop(0, `hsl(${seedHash % 360}, 50%, 12%)`);
      bgGrad.addColorStop(0.5, `hsl(${(seedHash + 40) % 360}, 40%, 8%)`);
      bgGrad.addColorStop(1, '#020617');
    } else {
      bgGrad.addColorStop(0, '#0f172a');
      bgGrad.addColorStop(1, '#020617');
    }
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, width, height);

    // Subtle film grid / grain pattern
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

    // Animated subject elements
    if (activeClip) {
      const clipElapsed = Math.max(0, currentTime - activeClip.startTime);
      const progressInClip = Math.min(1, Math.max(0, clipElapsed / activeClip.duration));
      const t = progressInClip * Math.PI * 2;

      // Draw artistic motion curve
      ctx.save();
      const cx = width / 2 + Math.sin(t) * 80;
      const cy = height / 2 + Math.cos(t * 1.5) * 40;

      // Glow halo
      const radialGrad = ctx.createRadialGradient(cx, cy, 10, cx, cy, 180);
      radialGrad.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
      radialGrad.addColorStop(0.4, 'rgba(6, 182, 212, 0.15)');
      radialGrad.addColorStop(1, 'rgba(2, 6, 23, 0)');
      ctx.fillStyle = radialGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, 180, 0, Math.PI * 2);
      ctx.fill();

      // Subject symbol / silhouette
      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(cx, cy, 16, 0, Math.PI * 2);
      ctx.fill();

      // Kinetic orbit rings
      ctx.strokeStyle = 'rgba(103, 232, 249, 0.4)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.ellipse(cx, cy, 60 + Math.sin(t * 2) * 10, 30 + Math.cos(t * 2) * 10, t, 0, Math.PI * 2);
      ctx.stroke();

      // Title overlay in preview
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.font = '600 13px system-ui, -apple-system, sans-serif';
      ctx.fillText(activeClip.title, 24, 32);

      // Prompt overlay
      ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
      ctx.font = '400 11px system-ui, -apple-system, sans-serif';
      const promptSnippet = activeClip.prompt.length > 55 ? activeClip.prompt.slice(0, 55) + '...' : activeClip.prompt;
      ctx.fillText(`"${promptSnippet}"`, 24, 52);

      ctx.restore();
    } else {
      // Empty timeline state
      ctx.fillStyle = 'rgba(148, 163, 184, 0.4)';
      ctx.font = '500 13px system-ui, -apple-system, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No active clip at current playhead position', width / 2, height / 2 - 10);
      ctx.font = '400 11px system-ui, -apple-system, sans-serif';
      ctx.fillStyle = 'rgba(148, 163, 184, 0.3)';
      ctx.fillText('Scrub the playhead or add clips from the left panel', width / 2, height / 2 + 14);
      ctx.textAlign = 'left';
    }

    // Timecode badge in corner
    ctx.fillStyle = 'rgba(2, 6, 23, 0.7)';
    ctx.fillRect(width - 120, height - 32, 108, 22);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.6)';
    ctx.strokeRect(width - 120, height - 32, 108, 22);

    ctx.fillStyle = '#38bdf8';
    ctx.font = '600 11px ui-monospace, monospace';
    ctx.fillText(formatTime(currentTime), width - 112, height - 17);
  }, [currentTime, activeClip, fps]);

  return (
    <div className="flex-1 flex flex-col bg-slate-950 p-4 min-w-0 select-none">
      {/* Player Screen Frame */}
      <div className="flex-1 min-h-0 flex items-center justify-center relative">
        <div className="relative w-full max-w-3xl aspect-video bg-black rounded-xl overflow-hidden border border-slate-800 shadow-2xl flex items-center justify-center">
          <canvas
            ref={canvasRef}
            width={854}
            height={480}
            className="w-full h-full object-contain"
          />

          {/* Top Info Bar inside Player */}
          <div className="absolute top-2.5 left-3 right-3 flex items-center justify-between text-[11px] pointer-events-none">
            <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800/80 text-slate-300">
              <Film className="w-3.5 h-3.5 text-cyan-400" />
              <span>Vidgen Canvas Preview</span>
              <span className="text-slate-600">·</span>
              <span className="text-cyan-400 font-mono">16:9 480p</span>
            </div>

            <div className="flex items-center gap-2 bg-slate-950/80 backdrop-blur-md px-2.5 py-1 rounded-md border border-slate-800/80 text-slate-300 font-mono tabular-nums">
              <span>{fps} FPS</span>
              <span className="text-slate-600">·</span>
              <span>Frame {currentFrame} / {totalFrames}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Control Bar Below Player Screen */}
      <div className="pt-3 max-w-3xl mx-auto w-full space-y-2">
        {/* Scrub Bar Slider */}
        <div className="relative group flex items-center">
          <input
            type="range"
            min={0}
            max={Math.max(1, totalDuration)}
            step={1 / fps}
            value={currentTime}
            onChange={(e) => onSeek(parseFloat(e.target.value))}
            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400 group-hover:h-2 transition-all focus:outline-none"
          />
        </div>

        {/* Playback Buttons & Timestamp Display */}
        <div className="flex items-center justify-between gap-4 text-xs">
          {/* Left: Reset / Step Back */}
          <div className="flex items-center gap-1">
            <button
              onClick={onResetTime}
              className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-md transition-colors"
              title="Return to start (Home)"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
            <button
              onClick={() => onStepFrame(-1)}
              className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-md transition-colors"
              title="Step frame back (Left Arrow)"
            >
              <SkipBack className="w-4 h-4" />
            </button>
          </div>

          {/* Center: Play / Pause */}
          <div className="flex items-center gap-3">
            <button
              onClick={onTogglePlay}
              className="w-9 h-9 rounded-full bg-cyan-400 hover:bg-cyan-300 text-slate-950 flex items-center justify-center shadow-lg shadow-cyan-950/60 active:scale-95 transition-all"
              title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            >
              {isPlaying ? (
                <Pause className="w-4 h-4 fill-slate-950" />
              ) : (
                <Play className="w-4 h-4 fill-slate-950 ml-0.5" />
              )}
            </button>
          </div>

          {/* Right: Step Forward & Time Code */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => onStepFrame(1)}
              className="p-1.5 text-slate-400 hover:text-slate-100 hover:bg-slate-800 rounded-md transition-colors"
              title="Step frame forward (Right Arrow)"
            >
              <SkipForward className="w-4 h-4" />
            </button>

            {/* Time display: MM:SS.ff / Total */}
            <div className="font-mono text-xs tabular-nums bg-slate-900 border border-slate-800 px-3 py-1 rounded-md text-slate-300">
              <span className="text-cyan-400 font-semibold">{formatTime(currentTime)}</span>
              <span className="text-slate-600 mx-1.5">/</span>
              <span className="text-slate-400">{formatTime(totalDuration)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
