import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Clock,
  Sparkles,
  ShieldCheck,
  ChevronRight,
  HardDrive,
  Film,
  Scissors,
  Cpu,
  Layers,
  Sliders,
} from 'lucide-react';

interface FilmstripHeroProps {
  onExploreClick: () => void;
  onTryItClick: () => void;
}

export const FilmstripHero: React.FC<FilmstripHeroProps> = ({
  onExploreClick,
  onTryItClick,
}) => {
  // Scrub time from 0.0s to 30.0s
  const [scrubTime, setScrubTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [isUserScrubbing, setIsUserScrubbing] = useState<boolean>(false);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(performance.now());

  // 6 chunks = 30 seconds total (5s each) with friendly, plain words
  const chunks = [
    { id: 0, startSec: 0, endSec: 5, time: '0s - 5s', frames: '1 - 81f', label: 'Scene setup & anchor', color: 'from-sky-500 to-cyan-500' },
    { id: 1, startSec: 5, endSec: 10, time: '5s - 10s', frames: '81 - 161f', label: 'Subject motion begins', color: 'from-cyan-500 to-teal-500' },
    { id: 2, startSec: 10, endSec: 15, time: '10s - 15s', frames: '161 - 241f', label: 'Camera pan & flow', color: 'from-teal-500 to-emerald-500' },
    { id: 3, startSec: 15, endSec: 20, time: '15s - 20s', frames: '241 - 321f', label: 'Lighting & texture shift', color: 'from-emerald-500 to-teal-500' },
    { id: 4, startSec: 20, endSec: 25, time: '20s - 25s', frames: '321 - 401f', label: 'Peak action & detail', color: 'from-teal-500 to-cyan-500' },
    { id: 5, startSec: 25, endSec: 30, time: '25s - 30s', frames: '401 - 481f', label: 'Seamless final frames', color: 'from-cyan-500 to-sky-500' },
  ];

  // Derived simulation metrics
  const activeChunkIndex = Math.min(5, Math.floor(scrubTime / 5));
  const currentChunk = chunks[activeChunkIndex] || chunks[0];
  const chunkProgress = ((scrubTime - currentChunk.startSec) / 5) * 100;
  // Exact raw frame from 1 to 481 (T = 4k + 1)
  const currentRawFrame = Math.min(481, Math.round((scrubTime / 30) * 480) + 1);
  // Latent frame from 1 to 121 (4x temporal VAE)
  const currentLatentFrame = Math.min(121, Math.floor((currentRawFrame - 1) / 4) + 1);

  // Speed: ~18s for full 30s timeline loop
  const LOOP_SPEED_MULTIPLIER = 30 / 18;

  useEffect(() => {
    // Respect user preference for reduced motion
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
      setIsPlaying(false);
    }

    const animate = (time: number) => {
      const deltaSec = (time - lastTimeRef.current) / 1000;
      lastTimeRef.current = time;

      if (isPlaying && !isUserScrubbing) {
        setScrubTime((prev) => {
          const next = prev + deltaSec * LOOP_SPEED_MULTIPLIER;
          if (next >= 30) {
            return 0;
          }
          return next;
        });
      }
      animationRef.current = requestAnimationFrame(animate);
    };

    lastTimeRef.current = performance.now();
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, isUserScrubbing]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setScrubTime(val);
  };

  const handleSvgPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    setIsUserScrubbing(true);
    updateScrubFromPointer(e);
  };

  const updateScrubFromPointer = (e: React.PointerEvent<SVGSVGElement> | React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const ratio = x / rect.width;
    const newTime = Math.max(0, Math.min(30, ratio * 30));
    setScrubTime(newTime);
  };

  const handleSvgPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (isUserScrubbing) {
      updateScrubFromPointer(e);
    }
  };

  const handleSvgPointerUp = () => {
    setIsUserScrubbing(false);
  };

  const handleChunkSelect = (idx: number) => {
    setScrubTime(idx * 5);
  };

  const handleTogglePlay = () => {
    setIsPlaying((prev) => !prev);
    lastTimeRef.current = performance.now();
  };

  const handleReset = () => {
    setScrubTime(0);
    lastTimeRef.current = performance.now();
  };

  const statTiles = [
    {
      icon: Film,
      value: '30 Seconds',
      label: 'Six 5-second chunks',
      detail: '481 raw frames at 16 FPS',
    },
    {
      icon: HardDrive,
      value: 'Flat Memory',
      label: 'Constant ~18.4 GB',
      detail: 'No OOM crashes over time',
    },
    {
      icon: Scissors,
      value: 'Zero Jump Cuts',
      label: 'Smooth cosine blend',
      detail: '1-latent boundary overlap',
    },
    {
      icon: Cpu,
      value: 'Decoupled DiT',
      label: 'Over 80% compute saved',
      detail: 'Split space & time attention',
    },
  ];

  // SVG coordinate dimensions
  const SVG_WIDTH = 900;
  const SVG_HEIGHT = 120;
  const playheadX = (scrubTime / 30) * SVG_WIDTH;

  return (
    <section className="relative overflow-hidden pt-6 pb-10 sm:pt-12 sm:pb-16 text-slate-100">
      {/* Background ambient accent glow */}
      <div 
        aria-hidden="true" 
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(90vw,700px)] h-[350px] bg-gradient-to-tr from-sky-500/10 via-cyan-500/10 to-teal-500/10 rounded-full blur-3xl pointer-events-none -z-10" 
      />

      <div className="max-w-[1100px] w-full mx-auto px-4 sm:px-6 lg:px-8 space-y-8 sm:space-y-10">
        {/* Main Headline & Description */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-300 font-medium">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>Vidgen Continuous Diffusion</span>
            <span className="text-slate-600">•</span>
            <span className="text-cyan-300">Long Video Architecture</span>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Long video generation with{' '}
            <span className="bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 bg-clip-text text-transparent">
              flat memory
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl mx-auto">
            Generate continuous, smooth 30-second videos in 5-second chunks. By keeping memory strictly constant, Vidgen avoids hardware crashes and eliminates visible cuts.
          </p>

          {/* Action CTAs with >=44px touch targets */}
          <div className="pt-2 flex flex-wrap items-center justify-center gap-3 sm:gap-4">
            <button
              onClick={onTryItClick}
              className="min-h-[44px] px-6 py-2.5 rounded-xl font-semibold text-slate-950 bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 hover:opacity-95 shadow-lg shadow-cyan-500/20 active:scale-[0.98] transition-all flex items-center gap-2 text-base cursor-pointer"
            >
              <Sparkles className="w-4 h-4 text-slate-950" />
              <span>Try Generator</span>
            </button>
            <button
              onClick={onExploreClick}
              className="min-h-[44px] px-5 py-2.5 rounded-xl font-medium text-slate-200 bg-slate-900 hover:bg-slate-800 border border-slate-800 active:scale-[0.98] transition-all flex items-center gap-2 text-base cursor-pointer"
            >
              <span>Explore 30s Architecture</span>
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Small Stat Tiles */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          {statTiles.map((tile, idx) => {
            const Icon = tile.icon;
            return (
              <div
                key={idx}
                className="p-3.5 sm:p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex flex-col justify-between space-y-1 hover:border-slate-700/80 transition-colors"
              >
                <div className="flex items-center gap-2 text-slate-400 mb-1">
                  <Icon className="w-4 h-4 text-cyan-400 shrink-0" />
                  <span className="text-xs font-medium text-slate-400 truncate">{tile.label}</span>
                </div>
                <div className="text-lg sm:text-xl font-bold text-white tracking-tight">
                  {tile.value}
                </div>
                <div className="text-[11px] text-slate-400 truncate">
                  {tile.detail}
                </div>
              </div>
            );
          })}
        </div>

        {/* Interactive SVG Timeline Component */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl shadow-slate-950/50 space-y-5">
          {/* Timeline Header Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
                <Clock className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-white">
                  Interactive 30-Second Generation Timeline
                </h2>
                <p className="text-xs text-slate-400">
                  Scrub anywhere on the SVG timeline or use the slider to inspect chunk rollout
                </p>
              </div>
            </div>

            {/* Play/Pause & Reset Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleTogglePlay}
                aria-label={isPlaying ? 'Pause timeline animation' : 'Play timeline animation'}
                className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                {isPlaying ? <Pause className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-cyan-400" />}
                <span className="hidden sm:inline">{isPlaying ? 'Pause' : 'Resume'}</span>
              </button>
              <button
                onClick={handleReset}
                aria-label="Restart generation loop"
                className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer"
              >
                <RotateCcw className="w-4 h-4 text-slate-400" />
                <span className="hidden sm:inline">Reset</span>
              </button>
            </div>
          </div>

          {/* Interactive SVG Canvas */}
          <div className="relative rounded-xl bg-slate-950 border border-slate-800 p-2 sm:p-3 overflow-hidden select-none">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
              className="w-full h-auto cursor-ew-resize touch-none block"
              onPointerDown={handleSvgPointerDown}
              onPointerMove={handleSvgPointerMove}
              onPointerUp={handleSvgPointerUp}
              onPointerCancel={handleSvgPointerUp}
              role="img"
              aria-label="Interactive 30-second video generation timeline"
            >
              <defs>
                {/* Accent wave gradient */}
                <linearGradient id="waveStrokeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#38bdf8" />
                  <stop offset="50%" stopColor="#22d3ee" />
                  <stop offset="100%" stopColor="#2dd4bf" />
                </linearGradient>

                {/* Shaded area gradient under active wave */}
                <linearGradient id="activeAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.25" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
                </linearGradient>

                {/* Seam glow gradient */}
                <linearGradient id="seamGlow" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.3" />
                  <stop offset="50%" stopColor="#22d3ee" stopOpacity="0.08" />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Background gridlines for 6 chunks */}
              {chunks.map((ch, idx) => {
                const xStart = (idx / 6) * SVG_WIDTH;
                const chunkW = SVG_WIDTH / 6;
                const isChunkActive = activeChunkIndex === idx;

                return (
                  <g key={ch.id}>
                    {/* Chunk boundary box */}
                    <rect
                      x={xStart + 2}
                      y={8}
                      width={chunkW - 4}
                      height={76}
                      rx={6}
                      fill={isChunkActive ? '#0f172a' : '#030712'}
                      stroke={isChunkActive ? '#0891b2' : '#1e293b'}
                      strokeWidth={isChunkActive ? 1.5 : 1}
                      opacity={isChunkActive ? 1 : 0.6}
                    />

                    {/* Chunk header text inside SVG */}
                    <text
                      x={xStart + 8}
                      y={24}
                      fill={isChunkActive ? '#38bdf8' : '#64748b'}
                      fontSize={11}
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      Chunk {idx + 1}
                    </text>

                    <text
                      x={xStart + chunkW - 8}
                      y={24}
                      textAnchor="end"
                      fill="#64748b"
                      fontSize={10}
                      fontFamily="monospace"
                    >
                      {ch.time}
                    </text>

                    {/* Frame count label */}
                    <text
                      x={xStart + 8}
                      y={76}
                      fill={isChunkActive ? '#e2e8f0' : '#475569'}
                      fontSize={10}
                      fontFamily="sans-serif"
                    >
                      {ch.frames}
                    </text>
                  </g>
                );
              })}

              {/* Seamless Seam Overlap Bands (1 latent frame cosine blend between adjacent chunks) */}
              {[1, 2, 3, 4, 5].map((idx) => {
                const seamX = (idx / 6) * SVG_WIDTH;
                return (
                  <g key={`seam-${idx}`}>
                    <rect
                      x={seamX - 6}
                      y={8}
                      width={12}
                      height={76}
                      fill="url(#seamGlow)"
                    />
                    <line
                      x1={seamX}
                      y1={8}
                      x2={seamX}
                      y2={84}
                      stroke="#22d3ee"
                      strokeWidth={1}
                      strokeDasharray="2 2"
                      opacity={0.6}
                    />
                  </g>
                );
              })}

              {/* Latent Denoising Wave Curve across 30 seconds */}
              {/* Baseline background wave */}
              <path
                d="M 10 50 Q 75 32, 150 50 T 300 50 T 450 50 T 600 50 T 750 50 T 890 50"
                fill="none"
                stroke="#1e293b"
                strokeWidth={3}
              />

              {/* Active Denoised Wave up to playhead position */}
              <clipPath id="playheadClip">
                <rect x={0} y={0} width={playheadX} height={SVG_HEIGHT} />
              </clipPath>

              {/* Active Wave with neon stroke */}
              <path
                d="M 10 50 Q 75 32, 150 50 T 300 50 T 450 50 T 600 50 T 750 50 T 890 50"
                fill="none"
                stroke="url(#waveStrokeGrad)"
                strokeWidth={3.5}
                clipPath="url(#playheadClip)"
              />

              {/* Shaded area under active wave */}
              <path
                d="M 10 50 Q 75 32, 150 50 T 300 50 T 450 50 T 600 50 T 750 50 T 890 50 L 890 84 L 10 84 Z"
                fill="url(#activeAreaGrad)"
                clipPath="url(#playheadClip)"
              />

              {/* Time ruler at bottom (y = 96) */}
              <line x1={8} y1={96} x2={SVG_WIDTH - 8} y2={96} stroke="#334155" strokeWidth={1} />
              {/* Ruler ticks & labels */}
              {[0, 5, 10, 15, 20, 25, 30].map((sec) => {
                const tickX = (sec / 30) * (SVG_WIDTH - 16) + 8;
                return (
                  <g key={`tick-${sec}`}>
                    <line x1={tickX} y1={92} x2={tickX} y2={100} stroke="#64748b" strokeWidth={1.5} />
                    <text
                      x={tickX}
                      y={112}
                      textAnchor="middle"
                      fill="#94a3b8"
                      fontSize={10}
                      fontFamily="monospace"
                      fontWeight="bold"
                    >
                      {sec}s
                    </text>
                  </g>
                );
              })}

              {/* Playhead Scrub Line & Indicator Node */}
              <g transform={`translate(${playheadX}, 0)`}>
                {/* Glow column */}
                <line
                  x1={0}
                  y1={4}
                  x2={0}
                  y2={96}
                  stroke="#38bdf8"
                  strokeWidth={2}
                />
                <line
                  x1={0}
                  y1={4}
                  x2={0}
                  y2={96}
                  stroke="#38bdf8"
                  strokeWidth={6}
                  opacity={0.3}
                />

                {/* Top Badge showing current scrub seconds */}
                <rect
                  x={-24}
                  y={4}
                  width={48}
                  height={18}
                  rx={4}
                  fill="#0284c7"
                  stroke="#38bdf8"
                  strokeWidth={1}
                />
                <text
                  x={0}
                  y={16}
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize={10}
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  {scrubTime.toFixed(1)}s
                </text>

                {/* Glowing Node at Wave Center */}
                <circle cx={0} cy={50} r={5} fill="#22d3ee" stroke="#ffffff" strokeWidth={1.5} />
                <circle cx={0} cy={50} r={9} fill="none" stroke="#38bdf8" strokeWidth={1.5} opacity={0.6} />

                {/* Bottom Pointer Diamond */}
                <polygon points="0,96 -4,103 4,103" fill="#38bdf8" />
              </g>
            </svg>
          </div>

          {/* Interactive Scrubbing Slider Control */}
          <div className="space-y-2 pt-1">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <label htmlFor="timeline-scrubber" className="flex items-center gap-1.5 text-slate-300 font-sans font-semibold">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                <span>Timeline Scrubber</span>
              </label>
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Position:</span>
                <strong className="text-cyan-300 font-bold">{scrubTime.toFixed(1)}s / 30.0s</strong>
                <span className="text-slate-600">•</span>
                <span className="text-slate-400">Frame:</span>
                <strong className="text-emerald-300 font-bold">{currentRawFrame} / 481f</strong>
              </div>
            </div>

            {/* Range input slider */}
            <input
              id="timeline-scrubber"
              type="range"
              min="0"
              max="30"
              step="0.1"
              value={scrubTime}
              onChange={handleSliderChange}
              onMouseDown={() => setIsUserScrubbing(true)}
              onMouseUp={() => setIsUserScrubbing(false)}
              onTouchStart={() => setIsUserScrubbing(true)}
              onTouchEnd={() => setIsUserScrubbing(false)}
              className="w-full h-3 bg-slate-950 rounded-xl appearance-none cursor-pointer accent-cyan-400 border border-slate-800"
              aria-valuemin={0}
              aria-valuemax={30}
              aria-valuenow={parseFloat(scrubTime.toFixed(1))}
              aria-label="Scrub 30-second video generation timeline"
            />

            <div className="flex justify-between text-[11px] font-mono text-slate-500">
              <span>0s (Start)</span>
              <span>10s</span>
              <span>20s</span>
              <span>30s (481 frames)</span>
            </div>
          </div>

          {/* 6 Filmstrip Chunks Cards synchronized with scrubber */}
          <div className="pt-2">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-3">
              {chunks.map((chunk, index) => {
                const isActive = activeChunkIndex === index;
                const isPast = activeChunkIndex > index;

                return (
                  <button
                    key={chunk.id}
                    onClick={() => handleChunkSelect(index)}
                    className={`relative p-3 rounded-xl text-left transition-all min-h-[96px] flex flex-col justify-between border cursor-pointer ${
                      isActive
                        ? 'bg-slate-800 border-cyan-400 shadow-md shadow-cyan-500/10 ring-1 ring-cyan-400/40'
                        : isPast
                        ? 'bg-slate-900 border-slate-800 hover:border-slate-700 text-slate-300'
                        : 'bg-slate-900/50 border-slate-800/60 hover:border-slate-700/80 text-slate-400'
                    }`}
                  >
                    {/* Top row: Chunk badge & Time */}
                    <div className="flex items-center justify-between w-full">
                      <span className={`text-[11px] font-mono px-1.5 py-0.5 rounded font-bold ${
                        isActive ? 'bg-cyan-400 text-slate-950' : 'bg-slate-800 text-slate-300'
                      }`}>
                        Chunk {index + 1}
                      </span>
                      <span className="text-[11px] font-mono text-slate-400">
                        {chunk.time}
                      </span>
                    </div>

                    {/* Middle: Simulated visual wave */}
                    <div className="my-2 h-7 rounded-md bg-slate-950 overflow-hidden relative border border-slate-800 flex items-center px-2">
                      <div className={`absolute inset-0 bg-gradient-to-r ${chunk.color} opacity-20`} />
                      
                      <div className="relative w-full flex items-center justify-between">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400" />
                        <div className="h-[2px] flex-1 mx-1.5 bg-gradient-to-r from-sky-400 via-cyan-400 to-teal-400" />
                        <div className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                      </div>

                      {/* Overlap Seam marker */}
                      {index > 0 && (
                        <div 
                          title="1-latent seamless overlap with cosine cross-fade"
                          className="absolute left-0 top-0 bottom-0 w-1.5 bg-cyan-400/40 border-r border-cyan-400/60" 
                        />
                      )}
                    </div>

                    {/* Bottom: Friendly status label */}
                    <div className="w-full">
                      <p className={`text-xs truncate font-medium ${isActive ? 'text-cyan-200' : 'text-slate-400'}`}>
                        {chunk.label}
                      </p>
                    </div>

                    {/* Active Progress Bar inside chunk */}
                    {isActive && (
                      <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800 rounded-b-xl overflow-hidden">
                        <div 
                          className="h-full bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 transition-all duration-75"
                          style={{ width: `${chunkProgress}%` }}
                        />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Active Status Callout with Plain Words */}
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong className="text-white">Active Step:</strong> Generating {currentChunk.time} ({currentChunk.frames}) • <span className="text-cyan-300 font-mono font-semibold">Latent {currentLatentFrame}/121</span>
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
              <span className="text-slate-400">
                Memory: <strong className="text-cyan-300">18.4 GB (Constant)</strong>
              </span>
              <span className="text-slate-400">
                Seam Blend: <strong className="text-emerald-300">Smooth Cosine</strong>
              </span>
              <span className="text-slate-400">
                Context: <strong className="text-sky-300">Rolling Window</strong>
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
