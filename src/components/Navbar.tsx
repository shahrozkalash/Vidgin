import React, { useEffect, useRef, useState } from 'react';
import { Video, FolderOpen, ChevronDown, Clapperboard, Share2, Check, RotateCcw } from 'lucide-react';
import { loadProjects, subscribeProjects, requestReuseProject, type Project } from '../lib/projects';

export const Navbar: React.FC = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [projects, setProjects] = useState<Project[]>(() => loadProjects());
  const menuRef = useRef<HTMLDivElement>(null);

  // Keep the list live: it updates the moment a new video is made.
  useEffect(() => subscribeProjects(() => setProjects(loadProjects())), []);

  // Close the dropdown when the user taps anywhere outside it.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [menuOpen]);

  const scrollTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const [copied, setCopied] = useState(false);
  const handleShare = async () => {
    const shareData = {
      title: 'vidgen',
      text: 'Describe it. Vidgen films it.',
      url: window.location.href,
    };
    if (navigator.share) {
      try {
        await navigator.share(shareData);
      } catch {
        // The user closed the share sheet — nothing to do.
      }
      return;
    }
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard blocked — nothing to do.
    }
  };

  const when = (ts: number) =>
    new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) +
    ', ' +
    new Date(ts).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });

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

          <div className="flex items-center gap-3 sm:gap-4">
            <span className="text-sm font-medium text-slate-500 hidden md:block">
              Words in, video out
            </span>

            {/* Share */}
            <button
              onClick={handleShare}
              className="flex items-center gap-2 px-3 py-2 rounded-full bg-white border border-violet-600/25 hover:border-violet-600/60 shadow-sm text-sm font-semibold text-slate-700 transition min-h-[44px] cursor-pointer"
              aria-label="Share Vidgin"
            >
              {copied ? (
                <Check className="w-4.5 h-4.5 text-violet-600" />
              ) : (
                <Share2 className="w-4.5 h-4.5 text-violet-600" />
              )}
              <span className="hidden sm:inline">{copied ? 'Link copied' : 'Share'}</span>
            </button>

            {/* Projects menu — top right */}
            <div className="relative" ref={menuRef}>
              <button
                onClick={() => setMenuOpen((o) => !o)}
                className="flex items-center gap-2 pl-2.5 pr-3 py-2 rounded-full bg-white border border-violet-600/25 hover:border-violet-600/60 shadow-sm text-sm font-semibold text-slate-700 transition min-h-[44px] cursor-pointer"
                aria-haspopup="true"
                aria-expanded={menuOpen}
              >
                <FolderOpen className="w-4.5 h-4.5 text-violet-600" />
                <span>Projects</span>
                {projects.length > 0 && (
                  <span className="min-w-5 h-5 px-1 rounded-full bg-violet-600 text-white text-[11px] font-bold flex items-center justify-center">
                    {projects.length}
                  </span>
                )}
                <ChevronDown
                  className={`w-4 h-4 text-slate-400 transition-transform ${menuOpen ? 'rotate-180' : ''}`}
                />
              </button>

              {menuOpen && (
                <div className="absolute right-0 top-[calc(100%+8px)] w-80 max-w-[85vw] bg-white rounded-3xl border border-violet-900/10 shadow-xl shadow-violet-900/10 overflow-hidden">
                  <div className="flex items-center justify-between px-4 pt-4 pb-2">
                    <span className="text-sm font-bold text-slate-800">Your projects</span>
                    <span className="text-xs font-medium text-slate-400">
                      {projects.length} total
                    </span>
                  </div>

                  {projects.length === 0 ? (
                    <p className="px-4 pb-5 pt-1 text-sm text-slate-400 text-center">
                      No projects yet.
                      <br />
                      Describe a video below and it will show up here.
                    </p>
                  ) : (
                    <>
                      <div className="max-h-72 overflow-y-auto divide-y divide-slate-100">
                        {projects.map((p, i) => (
                          <button
                            key={p.createdAt + '-' + i}
                            onClick={() => {
                              requestReuseProject(p);
                              setMenuOpen(false);
                            }}
                            className="w-full text-left px-4 py-3 hover:bg-violet-50 transition-colors cursor-pointer flex items-center gap-2"
                            title="Use this project again"
                          >
                            <span className="flex-1 min-w-0">
                              <span className="block text-sm font-medium text-slate-800 truncate">
                                {p.prompt}
                              </span>
                              <span className="mt-0.5 flex items-center gap-1.5 text-xs text-slate-400">
                                <Clapperboard className="w-3.5 h-3.5 text-violet-500" />
                                {p.duration}s video
                                <span aria-hidden="true">•</span>
                                {when(p.createdAt)}
                              </span>
                            </span>
                            <RotateCcw className="w-4 h-4 text-violet-400 shrink-0" />
                          </button>
                        ))}
                      </div>
                      <p className="px-4 py-2.5 bg-[#faf7ff] border-t border-violet-900/5 text-center text-xs text-slate-500">
                        You have worked on {projects.length}{' '}
                        {projects.length === 1 ? 'project' : 'projects'}
                      </p>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
