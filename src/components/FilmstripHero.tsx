import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Clock, Sparkles, ShieldCheck, ChevronRight, HardDrive, Film, Scissors, Cpu } from 'lucide-react';

interface FilmstripHeroProps {
  onExploreClick: () => void;
  onTryItClick: () => void;
}

export const FilmstripHero: React.FC<FilmstripHeroProps> = ({
  onExploreClick,
  onTryItClick,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [currentChunk, setCurrentChunk] = useState<number>(0);
  const [progress, setProgress] = useState<number>(0);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(performance.now());

  // 6 chunks = 30 seconds total (5s each) with friendly, plain words
  const chunks = [
    { id: 0, time: '0s - 5s', frames: 'Frames 1 - 81', label: 'Scene setup & anchor', color: 'from-sky-500 to-cyan-500' },
    { id: 1, time: '5s - 10s', frames: 'Frames 81 - 161', label: 'Subject motion begins', color: 'from-cyan-500 to-teal-500' },
    { id: 2, time: '10s - 15s', frames: 'Frames 161 - 241', label: 'Camera pan & flow', color: 'from-teal-500 to-emerald-500' },
    { id: 3, time: '15s - 20s', frames: 'Frames 241 - 321', label: 'Lighting & texture shift', color: 'from-emerald-500 to-teal-500' },
    { id: 4, time: '20s - 25s', frames: 'Frames 321 - 401', label: 'Peak action & detail', color: 'from-teal-500 to-cyan-500' },
    { id: 5, time: '25s - 30s', frames: 'Frames 401 - 481', label: 'Seamless final frames', color: 'from-cyan-500 to-sky-500' },
  ];

  // Slow, calm loop: ~2.4 seconds per chunk, total ~14.4s per complete cycle
  const CHUNK_DURATION_MS = 2400;

  useEffect(() => {
    // Respect user preference for reduced motion
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
      setIsPlaying(false);
    }

    const animate = (time: number) => {
      const delta = time - lastTimeRef.current;
      lastTimeRef.current = time;

      if (isPlaying) {
        setProgress((prev) => {
          const next = prev + (delta / CHUNK_DURATION_MS) * 100;
          if (next >= 100) {
            setCurrentChunk((c) => (c + 1) % chunks.length);
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
  }, [isPlaying, chunks.length]);

  const handleChunkClick = (idx: number) => {
    setCurrentChunk(idx);
    setProgress(0);
  };

  const handleTogglePlay = () => {
    setIsPlaying((prev) => !prev);
    lastTimeRef.current = performance.now();
  };

  const handleReset = () => {
    setCurrentChunk(0);
    setProgress(0);
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

        {/* Small Stat Tiles (Replaces jargon with plain words & numbers) */}
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

        {/* Creative Feature: Animated Filmstrip / Timeline */}
        <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl shadow-slate-950/50 space-y-4">
          {/* Filmstrip Header Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
                <Clock className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-white">
                  30-Second Generation Timeline
                </h2>
                <p className="text-xs text-slate-400">
                  Six 5-second chunks synthesized sequentially with continuous context
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

          {/* Filmstrip Chunks Grid / Timeline (Responsive: 2 col mobile, 3 col tablet, 6 col desktop) */}
          <div className="pt-2 pb-1">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-3">
              {chunks.map((chunk, index) => {
                const isActive = currentChunk === index;
                const isPast = currentChunk > index;

                return (
                  <button
                    key={chunk.id}
                    onClick={() => handleChunkClick(index)}
                    className={`relative p-3 rounded-xl text-left transition-all min-h-[96px] flex flex-col justify-between border cursor-pointer ${
                      isActive
                        ? 'bg-slate-800 border-cyan-400 shadow-md shadow-cyan-500/10'
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

                    {/* Middle: Keyframe simulated visual */}
                    <div className="my-2 h-7 rounded-md bg-slate-950 overflow-hidden relative border border-slate-800 flex items-center px-2">
                      <div className={`absolute inset-0 bg-gradient-to-r ${chunk.color} opacity-20`} />
                      
                      {/* Active wave line */}
                      <div className="relative w-full flex items-center justify-between">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400" />
                        <div className="h-[2px] flex-1 mx-1.5 bg-gradient-to-r from-sky-400 via-cyan-400 to-teal-400" />
                        <div className="w-1.5 h-1.5 rounded-full bg-teal-400" />
                      </div>

                      {/* Overlap Seam marker on chunk edges */}
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
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Active Chunk Callout info with Plain Words */}
          <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong className="text-white">Active Step:</strong> Generating {chunks[currentChunk].time} ({chunks[currentChunk].frames})
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
