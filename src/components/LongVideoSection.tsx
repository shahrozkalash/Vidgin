import React, { useState } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { calculateLongVideoSpecs } from '../utils/longVideoEngine';
import { MemoryTracker } from './MemoryTracker';
import { ChunkOverlapVisualizer } from './ChunkOverlapVisualizer';
import { Film, HardDrive, CheckCircle2, ChevronDown, Sliders } from 'lucide-react';

interface LongVideoSectionProps {
  currentPreset: PresetKey;
}

export const LongVideoSection: React.FC<LongVideoSectionProps> = ({ currentPreset }) => {
  const [targetDuration, setTargetDuration] = useState<number>(30); // 5 to 60s
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const specs = calculateLongVideoSpecs(targetDuration, 16, currentPreset);
  const cfg = PRESETS[currentPreset] || PRESETS.tiny;

  return (
    <section id="long_video" className="scroll-mt-20 space-y-6">
      {/* Section Header with the 30-Second Video information moved here */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 inline-flex items-center gap-1.5">
              <Film className="w-3.5 h-3.5" />
              <span>30-Second Video (481f, 16 FPS)</span>
            </span>
            <span className="text-xs text-slate-400 font-mono">
              Part 13 Verified
            </span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Long Video Architecture
          </h2>
          <p className="mt-1 text-base text-slate-300">
            How Vidgen generates continuous 30-second clips with constant O(1) memory.
          </p>
        </div>
      </div>

      {/* Interactive Duration & Memory Comparison Card */}
      <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6 shadow-lg shadow-slate-950/40">
        {/* Interactive Duration Scrubber */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label htmlFor="duration-slider" className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Video Duration: <strong className="text-cyan-300 font-mono text-base">{targetDuration} seconds</strong></span>
            </label>
            <div className="flex gap-1.5">
              {[5, 15, 30, 45, 60].map((sec) => (
                <button
                  key={sec}
                  onClick={() => setTargetDuration(sec)}
                  className={`min-h-[36px] px-2.5 py-1 text-xs font-medium rounded-lg transition-colors ${
                    targetDuration === sec
                      ? 'bg-cyan-400 text-slate-950 font-bold'
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

        {/* 4 Stat Highlights */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-2">
          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Total Frames</span>
            <strong className="text-lg font-bold text-white font-mono">
              {specs.totalFrames}f
            </strong>
            <span className="text-[11px] text-slate-400 block mt-0.5">T = 4k + 1</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Number of Chunks</span>
            <strong className="text-lg font-bold text-cyan-300 font-mono">
              {specs.numChunks} Chunks
            </strong>
            <span className="text-[11px] text-slate-400 block mt-0.5">5 seconds each</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Latent Frames</span>
            <strong className="text-lg font-bold text-slate-200 font-mono">
              {specs.latentFrames} Latents
            </strong>
            <span className="text-[11px] text-slate-400 block mt-0.5">4x temporal VAE</span>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
            <span className="text-xs text-slate-400 block mb-1">Attention Savings</span>
            <strong className="text-lg font-bold text-emerald-300 font-mono">
              {specs.attentionComplexityReduction > 1
                ? `${specs.attentionComplexityReduction.toFixed(1)}x Faster`
                : 'Baseline'}
            </strong>
            <span className="text-[11px] text-slate-400 block mt-0.5">vs Full Attention</span>
          </div>
        </div>

        {/* Memory Comparison Banner */}
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-3.5 rounded-lg bg-red-950/30 border border-red-900/40">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-red-300 uppercase tracking-wider">
                Unbounded Full Attention
              </span>
              <span className="text-xs font-mono font-bold text-red-400">O(N²) Explosion</span>
            </div>
            <p className="text-sm font-mono text-white mb-1">
              Required VRAM: <strong className="text-red-400">{specs.fullAttnVramGb.toFixed(1)} GB</strong>
            </p>
            <p className="text-xs text-slate-400">
              Quadratic token growth exceeds GPU memory capacity on clips longer than 5 seconds.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-emerald-950/30 border border-emerald-900/40">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
                Vidgen Chunked KV Cache
              </span>
              <span className="text-xs font-mono font-bold text-emerald-400">Flat O(1) Memory</span>
            </div>
            <p className="text-sm font-mono text-white mb-1">
              Required VRAM: <strong className="text-emerald-300">{specs.chunkedKvVramGb.toFixed(1)} GB</strong>
            </p>
            <p className="text-xs text-slate-400">
              Rolling window with anchor frames keeps activations strictly bounded indefinitely.
            </p>
          </div>
        </div>
      </div>

      {/* Accordions for Deep Technical Details (Interactive Memory Tracker & Overlap Visualizer) */}
      <div className="space-y-3 pt-2">
        {/* Accordion 1: Interactive Memory Tracker */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('memory_tool')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'memory_tool'}
          >
            <div className="flex items-center gap-2.5">
              <HardDrive className="w-4 h-4 text-cyan-400" />
              <span>Deep Technical Tool: Live Memory Tracker &amp; Eviction Simulator</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'memory_tool' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'memory_tool' && (
            <div className="p-4 sm:p-6 border-t border-slate-800/60 bg-slate-950/80">
              <MemoryTracker currentPreset={currentPreset} />
            </div>
          )}
        </div>

        {/* Accordion 2: 1-Latent Overlap & Seam Visualizer */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('overlap_tool')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'overlap_tool'}
          >
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              <span>Deep Technical Tool: 1-Latent Overlap &amp; Cosine Blending Visualizer</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'overlap_tool' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'overlap_tool' && (
            <div className="p-4 sm:p-6 border-t border-slate-800/60 bg-slate-950/80">
              <ChunkOverlapVisualizer currentPreset={currentPreset} />
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
