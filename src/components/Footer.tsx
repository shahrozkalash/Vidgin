import React from 'react';
import { Video, ShieldCheck, Heart } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="border-t border-slate-800/80 bg-slate-950 py-8 text-slate-400 text-xs sm:text-sm">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-tr from-indigo-500 to-cyan-400 p-0.5 flex items-center justify-center">
            <Video className="w-3.5 h-3.5 text-slate-950" />
          </div>
          <span className="font-bold text-white lowercase">vidgen</span>
          <span className="text-slate-500">•</span>
          <span>Continuous Latent Video Diffusion</span>
        </div>

        {/* Verification Status */}
        <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs">
          <ShieldCheck className="w-4 h-4 text-emerald-400" />
          <span>Parts 0-13 CPU Verified &amp; Passing</span>
        </div>

        {/* Quiet note */}
        <div className="text-slate-500 text-xs flex items-center gap-1">
          <span>Engineered for long-video continuity</span>
        </div>
      </div>
    </footer>
  );
};
