import React, { useState, useEffect } from 'react';
import { PresetKey } from './types/vidgen';
import { Navbar } from './components/Navbar';
import { FilmstripHero } from './components/FilmstripHero';
import { OverviewSection } from './components/OverviewSection';
import { PresetsSection } from './components/PresetsSection';
import { LongVideoSection } from './components/LongVideoSection';
import { TryItSection } from './components/TryItSection';
import { Footer } from './components/Footer';
import { EditorView } from './components/editor/EditorView';
import { Film, SlidersHorizontal } from 'lucide-react';

export default function App() {
  const [currentPreset, setCurrentPreset] = useState<PresetKey>('tiny');
  const [activeSection, setActiveSection] = useState<string>('overview');

  // Determine initial view: check URL query parameter ?view=editor | ?view=overview
  const getInitialView = (): 'editor' | 'overview' => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const viewParam = params.get('view');
      if (viewParam === 'editor') return 'editor';
      if (viewParam === 'overview') return 'overview';
      return window.innerWidth >= 900 ? 'editor' : 'overview';
    }
    return 'overview';
  };

  const [currentView, setCurrentView] = useState<'editor' | 'overview'>(getInitialView);
  const [hasUserOverridden, setHasUserOverridden] = useState(false);

  // Handle window resize dynamically if user hasn't explicitly overridden view
  useEffect(() => {
    const handleResize = () => {
      const params = new URLSearchParams(window.location.search);
      if (params.has('view')) return; // honor URL parameter strictly
      if (!hasUserOverridden) {
        setCurrentView(window.innerWidth >= 900 ? 'editor' : 'overview');
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [hasUserOverridden]);

  const handleToggleView = (view: 'editor' | 'overview') => {
    setCurrentView(view);
    setHasUserOverridden(true);
    // Update URL parameter without reload for easy testing on Chrome Desktop site
    const url = new URL(window.location.href);
    url.searchParams.set('view', view);
    window.history.replaceState({}, '', url.toString());
  };

  const scrollToSection = (sectionId: string) => {
    setActiveSection(sectionId);
    const el = document.getElementById(sectionId);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  // If in Desktop Editor mode
  if (currentView === 'editor') {
    return <EditorView onSwitchToOverview={() => handleToggleView('overview')} />;
  }

  // Existing mobile/tablet & classic layout (completely unchanged)
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200 relative">
      {/* Discreet floating View Toggle to switch to Desktop Editor anytime */}
      <div className="fixed bottom-4 right-4 z-50 shadow-xl">
        <button
          onClick={() => handleToggleView('editor')}
          className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-slate-900/90 hover:bg-slate-800 text-slate-200 border border-cyan-500/60 shadow-lg shadow-cyan-950/50 backdrop-blur-md text-xs font-medium transition-all hover:scale-105 active:scale-95"
          title="Switch to Desktop Timeline Video Editor"
        >
          <Film className="w-3.5 h-3.5 text-cyan-400" />
          <span>Switch to Editor</span>
        </button>
      </div>

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

