import React, { useState } from 'react';
import { Video, Menu, X, ChevronRight } from 'lucide-react';

interface NavbarProps {
  activeSection: string;
  onSectionChange: (section: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeSection,
  onSectionChange,
}) => {
  const [menuOpen, setMenuOpen] = useState(false);

  const sections = [
    { id: 'overview', label: 'Overview', desc: 'Continuous diffusion architecture' },
    { id: 'presets', label: 'Presets', desc: '4 calibrated model tiers' },
    { id: 'long_video', label: 'Long video (30s)', desc: 'Flat O(1) memory & seam blending' },
    { id: 'try_it', label: 'Try it', desc: 'Simulate generation & feedback' },
  ];

  const handleNavClick = (id: string) => {
    onSectionChange(id);
    setMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md border-b border-slate-800/80">
      <div className="max-w-[1100px] w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Name 'vidgen' */}
          <button
            onClick={() => handleNavClick('overview')}
            className="flex items-center gap-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400 rounded-xl p-1 min-h-[44px] cursor-pointer"
            aria-label="vidgen home"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-sky-400 via-cyan-300 to-teal-300 p-0.5 shadow-md shadow-cyan-500/20 shrink-0">
              <div className="w-full h-full bg-slate-950 rounded-[7px] flex items-center justify-center">
                <Video className="w-4 h-4 text-cyan-300" />
              </div>
            </div>
            <span className="font-bold tracking-tight text-white text-xl lowercase font-sans">
              vidgen
            </span>
          </button>

          {/* Clean Desktop Navigation Links */}
          <nav className="hidden md:flex items-center gap-1.5" aria-label="Main navigation">
            {sections.map((sec) => {
              const isActive = activeSection === sec.id;
              return (
                <button
                  key={sec.id}
                  onClick={() => handleNavClick(sec.id)}
                  className={`min-h-[44px] px-3.5 py-2 rounded-xl text-base font-medium transition-all cursor-pointer ${
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

          {/* Small Menu Button (Accessible >=44px touch target) */}
          <div className="flex items-center">
            <button
              onClick={() => setMenuOpen((prev) => !prev)}
              aria-label={menuOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={menuOpen}
              className="min-h-[44px] min-w-[44px] p-2.5 rounded-xl text-slate-300 hover:text-white hover:bg-slate-900 border border-slate-800 flex items-center justify-center gap-2 transition-colors cursor-pointer"
            >
              {menuOpen ? <X className="w-5 h-5 text-slate-300" /> : <Menu className="w-5 h-5 text-slate-300" />}
              <span className="text-xs font-semibold uppercase tracking-wider hidden sm:inline md:hidden">Menu</span>
            </button>
          </div>
        </div>

        {/* Dropdown Menu when menu button is clicked */}
        {menuOpen && (
          <div className="py-3 px-2 border-t border-slate-800/80 bg-slate-950/98 animate-in fade-in slide-in-from-top-2 duration-150">
            <nav className="flex flex-col space-y-1" aria-label="Expanded navigation">
              {sections.map((sec) => {
                const isActive = activeSection === sec.id;
                return (
                  <button
                    key={sec.id}
                    onClick={() => handleNavClick(sec.id)}
                    className={`min-h-[48px] w-full text-left px-4 py-3 rounded-xl flex items-center justify-between transition-colors cursor-pointer ${
                      isActive
                        ? 'bg-slate-900 text-cyan-300 font-semibold border border-slate-800'
                        : 'text-slate-300 hover:text-white hover:bg-slate-900/50'
                    }`}
                  >
                    <div>
                      <div className="text-base font-semibold">{sec.label}</div>
                      <div className="text-xs text-slate-400 mt-0.5">{sec.desc}</div>
                    </div>
                    <ChevronRight className={`w-4 h-4 ${isActive ? 'text-cyan-300' : 'text-slate-500'}`} />
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
