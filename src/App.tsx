import React from 'react';
import { PenLine, Clapperboard, Play, ArrowRight } from 'lucide-react';
import { Navbar } from './components/Navbar';
import { FilmstripHero } from './components/FilmstripHero';
import { TryItSection } from './components/TryItSection';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-[#f6f1fb] text-slate-800 flex flex-col font-sans selection:bg-violet-200 selection:text-violet-900 relative">
      {/* Simple Header */}
      <Navbar />

      {/* Simple Hero */}
      <FilmstripHero />

      {/* One simple card */}
      <main className="flex-1 w-full py-2 sm:py-4 flex">
        <TryItSection />
      </main>

      {/* How it works — one slim line */}
      <div className="w-full px-4 pb-3">
        <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 text-[13px] font-medium text-slate-500">
          <span className="flex items-center gap-1.5">
            <PenLine className="w-4 h-4 text-violet-500" /> Write it
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-300" />
          <span className="flex items-center gap-1.5">
            <Clapperboard className="w-4 h-4 text-violet-500" /> Vidgin films it
          </span>
          <ArrowRight className="w-3.5 h-3.5 text-slate-300" />
          <span className="flex items-center gap-1.5">
            <Play className="w-4 h-4 text-violet-500" /> Watch it
          </span>
        </div>
      </div>

      {/* Minimal Footer */}
      <Footer />
    </div>
  );
}
