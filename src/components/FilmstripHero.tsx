import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, Clock, Layers, Sparkles, ShieldCheck, ChevronRight } from 'lucide-react';

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

  // 6 chunks = 30 seconds total (5s each)
  const chunks = [
    { id: 0, time: '0s - 5s', frames: 'Frames 1 - 81', label: 'Scene setup & anchor', color: 'from-indigo-600 to-sky-600' },
    { id: 1, time: '5s - 10s', frames: 'Frames 81 - 161', label: 'Subject motion begins', color: 'from-sky-600 to-cyan-600' },
    { id: 2, time: '10s - 15s', frames: 'Frames 161 - 241', label: 'Camera pan & flow', color: 'from-cyan-600 to-teal-600' },
    { id: 3, time: '15s - 20s', frames: 'Frames 241 - 321', label: 'Dynamic lighting shift', color: 'from-teal-600 to-emerald-600' },
    { id: 4, time: '20s - 25s', frames: 'Frames 321 - 401', label: 'Climax & motion peak', color: 'from-emerald-600 to-sky-600' },
    { id: 5, time: '25s - 30s', frames: 'Frames 401 - 481', label: 'Seamless scene finish', color: 'from-sky-600 to-indigo-600' },
  ];

  // Slow loop: ~2.5 seconds per chunk, total ~15s per cycle
  const CHUNK_DURATION_MS = 2400;

  useEffect(() => {
    // Check user preference for reduced motion
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

  return (
    <section className="relative overflow-hidden pt-8 pb-12 sm:pt-14 sm:pb-16 text-slate-100">
      {/* Background soft ambient glow */}
      <div 
        aria-hidden="true" 
        className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(90vw,700px)] h-[350px] bg-gradient-to-tr from-indigo-500/10 via-sky-500/10 to-teal-500/10 rounded-full blur-3xl pointer-events-none -z-10" 
      />

      <div className="max-w-[1100px] mx-auto px-4 sm:px-6 lg:px-8">
        {/* Main Headline & Description */}
        <div className="text-center max-w-3xl mx-auto space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900/90 border border-slate-800 text-xs text-slate-300 font-medium">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span>Seedance &amp; Vidgen Architecture</span>
            <span className="text-slate-500">•</span>
            <span className="text-cyan-300">Continuous 30-Second Diffusion</span>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-white leading-tight">
            Effortless long video generation with{' '}
            <span className="bg-gradient-to-r from-indigo-400 via-sky-300 to-cyan-300 bg-clip-text text-transparent">
              flat memory
            </span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed max-w-2xl mx-auto">
            Generate continuous, smooth 30-second videos in 5-second chunks. By keeping memory strictly constant, Vidgen avoids crashes and eliminates visible cuts.
          </p>

          {/* Action CTAs */}
          <div className="pt-2 flex flex-wrap items-center justify-center gap-3 sm:gap-4">
            <button
              onClick={onTryItClick}
              className="min-h-[44px] px-6 py-2.5 rounded-xl font-semibold text-slate-950 bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 hover:opacity-95 shadow-lg shadow-cyan-500/20 active:scale-[0.98] transition-all flex items-center gap-2 text-base"
            >
              <Sparkles className="w-4 h-4 text-slate-950" />
              <span>Try Generator</span>
            </button>
            <button
              onClick={onExploreClick}
              className="min-h-[44px] px-5 py-2.5 rounded-xl font-medium text-slate-200 bg-slate-900 hover:bg-slate-800 border border-slate-700/80 active:scale-[0.98] transition-all flex items-center gap-2 text-base"
            >
              <span>Explore 30s Architecture</span>
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Creative Feature: Animated Filmstrip / Timeline */}
        <div className="mt-10 sm:mt-14 bg-slate-900/90 border border-slate-800 rounded-2xl p-4 sm:p-6 shadow-xl shadow-slate-950/50">
          {/* Filmstrip Header Controls */}
          <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-indigo-950/80 border border-indigo-800/60 flex items-center justify-center text-indigo-400">
                <Clock className="w-4 h-4" />
              </div>
              <div>
                <h2 className="text-sm sm:text-base font-semibold text-white">
                  30-Second Generation Timeline
                </h2>
                <p className="text-xs text-slate-400">
                  Six 5-second chunks (481 frames at 16 FPS) generated in sequence
                </p>
              </div>
            </div>

            {/* Play/Pause & Reset Controls */}
            <div className="flex items-center gap-2">
              <button
                onClick={handleTogglePlay}
                aria-label={isPlaying ? 'Pause timeline animation' : 'Play timeline animation'}
                className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs font-medium flex items-center gap-1.5 transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4 text-amber-400" /> : <Play className="w-4 h-4 text-cyan-400" />}
                <span className="hidden sm:inline">{isPlaying ? 'Pause' : 'Resume'}</span>
              </button>
              <button
                onClick={handleReset}
                aria-label="Restart generation loop"
                className="min-h-[44px] min-w-[44px] px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 text-xs font-medium flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-4 h-4 text-slate-400" />
                <span className="hidden sm:inline">Reset</span>
              </button>
            </div>
          </div>

          {/* Filmstrip Chunks Grid / Timeline */}
          <div className="pt-5 pb-2">
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 sm:gap-3">
              {chunks.map((chunk, index) => {
                const isActive = currentChunk === index;
                const isPast = currentChunk > index;

                return (
                  <button
                    key={chunk.id}
                    onClick={() => handleChunkClick(index)}
                    className={`relative p-3 rounded-xl text-left transition-all min-h-[96px] flex flex-col justify-between border ${
                      isActive
                        ? 'bg-slate-800 border-cyan-400/80 shadow-md shadow-cyan-500/10'
                        : isPast
                        ? 'bg-slate-900/90 border-slate-800 hover:border-slate-700 text-slate-300'
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
                        <div className="h-[2px] flex-1 mx-1.5 bg-gradient-to-r from-cyan-500/40 via-sky-400/60 to-indigo-500/40" />
                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                      </div>

                      {/* Overlap Seam marker on chunk edges */}
                      {index > 0 && (
                        <div 
                          title="1-latent seamless overlap with cosine cross-fade"
                          className="absolute left-0 top-0 bottom-0 w-2 bg-cyan-400/30 border-r border-cyan-400/50" 
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
                          className="h-full bg-gradient-to-r from-sky-400 to-cyan-300 transition-all duration-75"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Active Chunk Callout info */}
          <div className="mt-4 p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 flex flex-wrap items-center justify-between gap-3 text-xs sm:text-sm text-slate-300">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
              <span>
                <strong className="text-white">Active Status:</strong> Generating {chunks[currentChunk].time} ({chunks[currentChunk].frames})
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
              <span className="text-slate-400">
                Memory: <strong className="text-cyan-300">18.4 GB (Flat O(1))</strong>
              </span>
              <span className="text-slate-400">
                Overlap: <strong className="text-emerald-300">1 Latent Cosine Blend</strong>
              </span>
              <span className="text-slate-400">
                Cache: <strong className="text-sky-300">2 Anchors + 1 Window</strong>
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
