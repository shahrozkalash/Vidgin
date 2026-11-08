import React from 'react';
import { Navbar } from './components/Navbar';
import { FilmstripHero } from './components/FilmstripHero';
import { TryItSection } from './components/TryItSection';
import { Footer } from './components/Footer';

export default function App() {
  return (
    <div className="min-h-screen bg-[#f3f8ec] text-slate-800 flex flex-col font-sans selection:bg-green-200 selection:text-green-900 relative">
      {/* Simple Header */}
      <Navbar />

      {/* Simple Hero */}
      <FilmstripHero />

      {/* One simple card */}
      <main className="flex-1 w-full py-2 sm:py-4 flex">
        <TryItSection />
      </main>

      {/* Minimal Footer */}
      <Footer />
    </div>
  );
}
