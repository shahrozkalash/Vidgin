import React from 'react';
import { Video } from 'lucide-react';

export const Navbar: React.FC = () => {
  const scrollTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <header className="sticky top-0 z-50 bg-[#f6f1fb]/85 backdrop-blur-md">
      <div className="max-w-[1100px] w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Name 'vidgen' */}
          <button
            onClick={scrollTop}
            className="flex items-center gap-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-violet-600 rounded-2xl p-1 min-h-[44px] cursor-pointer"
            aria-label="vidgen home"
          >
            <div className="w-9 h-9 rounded-2xl bg-violet-600 flex items-center justify-center shadow-sm shrink-0">
              <Video className="w-5 h-5 text-white" />
            </div>
            <span className="font-extrabold tracking-tight text-slate-900 text-[22px] lowercase font-sans">
              vidgen
            </span>
          </button>

          <span className="text-sm font-medium text-slate-500 hidden sm:block">
            Words in, video out
          </span>
        </div>
      </div>
    </header>
  );
};
