import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { FilmstripHero } from './components/FilmstripHero';
import { TryItSection } from './components/TryItSection';
import { Footer } from './components/Footer';
import { EditorView } from './components/editor/EditorView';
import { Film, SlidersHorizontal } from 'lucide-react';

export default function App() {

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

  // If in Desktop Editor mode
  if (currentView === 'editor') {
    return <EditorView onSwitchToOverview={() => handleToggleView('overview')} />;
  }

  // Existing mobile/tablet & classic layout (completely unchanged)
  return (
    <div className="min-h-screen bg-[#f3f8ec] text-slate-800 flex flex-col font-sans selection:bg-green-200 selection:text-green-900 relative">
      {/* Discreet floating View Toggle to switch to Desktop Editor anytime */}
      <div className="fixed bottom-4 right-4 z-50 shadow-xl">
        <button
          onClick={() => handleToggleView('editor')}
          className="flex items-center gap-2 px-3.5 py-2 rounded-full bg-white/90 hover:bg-white text-slate-700 border border-green-600/40 shadow-lg backdrop-blur-md text-xs font-medium transition-all hover:scale-105 active:scale-95"
          title="Switch to Desktop Timeline Video Editor"
        >
          <Film className="w-3.5 h-3.5 text-green-700" />
          <span>Switch to Editor</span>
        </button>
      </div>

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

