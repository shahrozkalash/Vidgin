import React from 'react';
import { Sparkles } from 'lucide-react';

interface FilmstripHeroProps {
  onTryItClick: () => void;
}

export const FilmstripHero: React.FC<FilmstripHeroProps> = ({
  onTryItClick,
}) => {
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
          </div>
        </div>

      </div>
    </section>
  );
};
