import React, { useState } from 'react';
import { Sliders } from 'lucide-react';

export const LongVideoSection: React.FC = () => {
  const [targetDuration, setTargetDuration] = useState<number>(30); // 5 to 60s

  return (
    <section id="long_video" className="scroll-mt-20 space-y-6">
      {/* Interactive Duration Card */}
      <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6 shadow-lg shadow-slate-950/40">
        {/* Interactive Duration Scrubber */}
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <label htmlFor="duration-slider" className="text-base font-semibold text-slate-200 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Video Duration: <strong className="text-cyan-300 font-mono text-base">{targetDuration}s</strong></span>
            </label>
            <div className="flex flex-wrap gap-1.5">
              {[5, 15, 30, 45, 60].map((sec) => (
                <button
                  key={sec}
                  onClick={() => setTargetDuration(sec)}
                  className={`min-h-[44px] px-3 py-1.5 text-xs font-semibold rounded-xl transition-colors cursor-pointer ${
                    targetDuration === sec
                      ? 'bg-cyan-400 text-slate-950 font-bold shadow-sm shadow-cyan-400/20'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {sec}s
                </button>
              ))}
            </div>
          </div>

          <input
            id="duration-slider"
            type="range"
            min="5"
            max="60"
            step="5"
            value={targetDuration}
            onChange={(e) => setTargetDuration(Number(e.target.value))}
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />

          <div className="flex justify-between text-xs text-slate-400 font-mono">
            <span>5s (1 chunk)</span>
            <span>15s (3 chunks)</span>
            <span>30s (6 chunks)</span>
            <span>60s (12 chunks)</span>
          </div>
        </div>
      </div>
    </section>
  );
};
