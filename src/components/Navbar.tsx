import React, { useState } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { Video, Menu, X } from 'lucide-react';

interface NavbarProps {
  currentPreset: PresetKey;
  onPresetChange: (preset: PresetKey) => void;
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentPreset,
  onPresetChange,
  activeSection,
  onSectionChange,
}) => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const sections = [
    { id: 'overview', label: 'Overview' },
    { id: 'presets', label: 'Presets' },
    { id: 'long_video', label: 'Long video' },
    { id: 'try_it', label: 'Try it' },
  ];

  const presetKeys: PresetKey[] = ['tiny', 'small', 'vidgen480p', 'vidgen14b'];

  const handleNavClick = (id: string) => {
    onSectionChange(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-slate-950/95 backdrop-blur-md border-b border-slate-800/80">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-3">
          {/* Logo & Brand Name 'vidgen' */}
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => handleNavClick('overview')}
              className="flex items-center gap-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-lg p-1 min-h-[44px]"
              aria-label="vidgen home"
            >
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 via-sky-500 to-cyan-400 p-0.5 shadow-md shadow-cyan-500/10 shrink-0">
                <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                  <Video className="w-4 h-4 text-cyan-400" />
                </div>
              </div>
              <div className="flex items-baseline gap-1.5">
                <span className="font-bold tracking-tight text-white text-xl lowercase font-sans">
                  vidgen
                </span>
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 inline-block mb-1" />
              </div>
            </button>
          </div>

          {/* Desktop Navigation Links (Centered / Calm) */}
          <nav className="hidden md:flex items-center gap-1" aria-label="Main navigation">
            {sections.map((sec) => {
              const isActive = activeSection === sec.id;
              return (
                <button
                  key={sec.id}
                  onClick={() => handleNavClick(sec.id)}
                  className={`min-h-[44px] px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-slate-900 text-cyan-300 font-semibold border border-slate-800 shadow-sm'
                      : 'text-slate-300 hover:text-white hover:bg-slate-900/60'
                  }`}
                >
                  {sec.label}
                </button>
              );
            })}
          </nav>

          {/* Preset Selector & Mobile Menu Button */}
          <div className="flex items-center gap-2 sm:gap-3">
            {/* Mobile Preset Selector (Native Dropdown for perfect touch & zero overflow) */}
            <div className="block sm:hidden">
              <label htmlFor="mobile-preset-select" className="sr-only">
                Select Model Preset
              </label>
              <select
                id="mobile-preset-select"
                value={currentPreset}
                onChange={(e) => onPresetChange(e.target.value as PresetKey)}
                className="min-h-[44px] bg-slate-900 border border-slate-800 text-xs font-medium text-slate-200 rounded-lg px-2.5 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
              >
                {presetKeys.map((pk) => {
                  const cfg = PRESETS[pk] || PRESETS.tiny;
                  return (
                    <option key={pk} value={pk}>
                      {cfg.name} ({cfg.params})
                    </option>
                  );
                })}
              </select>
            </div>

            {/* Tablet/Desktop Preset Pills (Flex-wrap safe) */}
            <div className="hidden sm:flex items-center bg-slate-900 p-1 rounded-xl border border-slate-800">
              {presetKeys.map((pk) => {
                const isSelected = currentPreset === pk;
                const cfg = PRESETS[pk] || PRESETS.tiny;
                return (
                  <button
                    key={pk}
                    onClick={() => onPresetChange(pk)}
                    className={`min-h-[36px] px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                      isSelected
                        ? 'bg-cyan-400 text-slate-950 font-bold shadow-sm'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    {cfg.name}
                  </button>
                );
              })}
            </div>

            {/* Mobile Hamburger / Menu Button (Touch Target >= 44px) */}
            <button
              onClick={() => setMobileMenuOpen((prev) => !prev)}
              aria-label={mobileMenuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={mobileMenuOpen}
              className="md:hidden min-h-[44px] min-w-[44px] p-2.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-900 border border-slate-800/80 flex items-center justify-center transition-colors"
            >
              {mobileMenuOpen ? <X className="w-5 h-5 text-slate-300" /> : <Menu className="w-5 h-5 text-slate-300" />}
            </button>
          </div>
        </div>

        {/* Mobile Dropdown Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-3 px-2 border-t border-slate-800/80 bg-slate-950/98 animate-in fade-in slide-in-from-top-2 duration-150">
            <nav className="flex flex-col space-y-1" aria-label="Mobile navigation">
              {sections.map((sec) => {
                const isActive = activeSection === sec.id;
                return (
                  <button
                    key={sec.id}
                    onClick={() => handleNavClick(sec.id)}
                    className={`min-h-[44px] w-full text-left px-4 py-2.5 rounded-xl text-base font-medium transition-colors ${
                      isActive
                        ? 'bg-slate-900 text-cyan-300 font-semibold border border-slate-800'
                        : 'text-slate-300 hover:text-white hover:bg-slate-900/50'
                    }`}
                  >
                    {sec.label}
                  </button>
                );
              })}
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};
