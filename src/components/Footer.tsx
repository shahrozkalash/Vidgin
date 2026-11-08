import React from 'react';
import { Video } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="py-8 text-slate-500 text-sm">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="w-6 h-6 rounded-lg bg-green-600 flex items-center justify-center">
            <Video className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="font-bold text-slate-800 lowercase">vidgen</span>
          <span className="text-slate-300">•</span>
          <span>Continuous Latent Video Diffusion</span>
        </div>

        {/* Quiet note */}
        <div className="text-slate-400 text-xs">
          Engineered for long-video continuity
        </div>
      </div>
    </footer>
  );
};
