import React, { useState } from 'react';
import { PresetKey } from './types/vidgen';
import { Navbar } from './components/Navbar';
import { FilmstripHero } from './components/FilmstripHero';
import { OverviewSection } from './components/OverviewSection';
import { PresetsSection } from './components/PresetsSection';
import { LongVideoSection } from './components/LongVideoSection';
import { TryItSection } from './components/TryItSection';
import { Footer } from './components/Footer';

export default function App() {
  const [currentPreset, setCurrentPreset] = useState<PresetKey>('tiny');
  const [activeSection, setActiveSection] = useState<string>('overview');

  const scrollToSection = (sectionId: string) => {
    setActiveSection(sectionId);
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Streamlined Header */}
      <Navbar
        activeSection={activeSection}
        onSectionChange={scrollToSection}
      />

      {/* Hero with Creative Animated Filmstrip / Timeline */}
      <FilmstripHero
        onExploreClick={() => scrollToSection('long_video')}
        onTryItClick={() => scrollToSection('try_it')}
      />

      {/* Main 4 Sections in Centered Container (Max ~1100px) */}
      <main className="flex-1 max-w-[1100px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12 space-y-16 sm:space-y-24">
        {/* Section 1: Overview */}
        <OverviewSection />

        {/* Section 2: Presets */}
        <PresetsSection
          currentPreset={currentPreset}
          onPresetChange={setCurrentPreset}
        />

        {/* Section 3: Long Video (30 Seconds) */}
        <LongVideoSection currentPreset={currentPreset} />

        {/* Section 4: Try it */}
        <TryItSection
          currentPreset={currentPreset}
          onPresetChange={setCurrentPreset}
        />
      </main>

      {/* Minimal Footer */}
      <Footer />
    </div>
  );
}
