import React, { useState } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { calculateDiTParameters, DIT_PRESET_CONFIGS } from '../utils/ditEngine';
import { ChevronDown, Sliders, Check } from 'lucide-react';

interface PresetsSectionProps {
  currentPreset: PresetKey;
  onPresetChange: (preset: PresetKey) => void;
}

export const PresetsSection: React.FC<PresetsSectionProps> = ({
  currentPreset,
  onPresetChange,
}) => {
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const presetKeys: PresetKey[] = ['tiny', 'small', 'vidgen480p', 'vidgen14b'];

  const presetDetails: Record<
    PresetKey,
    {
      recommendedFor: string;
      vram: string;
      speed: string;
      description: string;
    }
  > = {
    tiny: {
      recommendedFor: 'Unit tests & CPU verification',
      vram: '< 1 GB',
      speed: 'Instant (<0.1s/step)',
      description: 'Ideal for local debugging and CI/CD pipelines without GPU requirements.',
    },
    small: {
      recommendedFor: 'Local development & prototyping',
      vram: '~4 GB',
      speed: 'Fast (~0.8s/step)',
      description: 'Balanced size for testing architectural modifications and reward loops.',
    },
    vidgen480p: {
      recommendedFor: 'Production high-fidelity standard',
      vram: '~18.4 GB',
      speed: 'Standard',
      description: 'Our primary 1.3B workhorse model capable of sustained 30-second clips.',
    },
    vidgen14b: {
      recommendedFor: 'Cinematic flagship studio grade',
      vram: '~48 GB (Multi-GPU)',
      speed: 'Studio grade',
      description: 'Large-scale 14B model for commercial photorealistic motion graphics.',
    },
  };

  const activeCfg = PRESETS[currentPreset] || PRESETS.tiny;
  const activeDitCfg = DIT_PRESET_CONFIGS[currentPreset] || DIT_PRESET_CONFIGS.tiny;
  const ditParams = calculateDiTParameters(activeDitCfg, currentPreset);

  return (
    <section id="presets" className="scroll-mt-20 space-y-6">
      {/* Section Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Model Presets
          </h2>
          <p className="mt-1 text-base text-slate-300">
            Four calibrated tiers from fast CPU testing to flagship cinematic synthesis.
          </p>
        </div>

        {/* Mobile dropdown selector */}
        <div className="w-full sm:hidden">
          <label htmlFor="presets-mobile-select" className="text-xs text-slate-400 block mb-1.5 font-medium">
            Active Preset:
          </label>
          <select
            id="presets-mobile-select"
            value={currentPreset}
            onChange={(e) => onPresetChange(e.target.value as PresetKey)}
            className="w-full min-h-[44px] bg-slate-900 border border-slate-800 text-sm font-medium text-slate-200 rounded-xl px-3 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-400"
          >
            {presetKeys.map((pk) => {
              const cfg = PRESETS[pk] || PRESETS.tiny;
              return (
                <option key={pk} value={pk}>
                  {cfg.name} • {cfg.params} • {cfg.resolution}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Preset Cards Grid (Responsive: 1 col on mobile, 2 col on tablet, 4 col on desktop) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {presetKeys.map((pk) => {
          const cfg = PRESETS[pk] || PRESETS.tiny;
          const meta = presetDetails[pk];
          const isSelected = currentPreset === pk;

          return (
            <button
              key={pk}
              onClick={() => onPresetChange(pk)}
              className={`min-h-[44px] p-5 rounded-2xl text-left transition-all flex flex-col justify-between border ${
                isSelected
                  ? 'bg-slate-900 border-cyan-400/90 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-400/50'
                  : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/90'
              }`}
            >
              <div className="w-full space-y-3">
                {/* Header: Name & Selection indicator */}
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    {cfg.params}
                  </span>
                  {isSelected ? (
                    <span className="w-6 h-6 rounded-full bg-cyan-400 text-slate-950 flex items-center justify-center">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </span>
                  ) : (
                    <span className="text-xs text-slate-500">Select</span>
                  )}
                </div>

                <div>
                  <h3 className="text-lg font-bold text-white">
                    {cfg.name}
                  </h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    {cfg.tagline}
                  </p>
                </div>

                {/* Specs List as friendly stacked badges */}
                <div className="space-y-1.5 pt-2 border-t border-slate-800/80 text-xs">
                  <div className="flex justify-between text-slate-400">
                    <span>Resolution:</span>
                    <strong className="text-slate-200 font-mono">{cfg.resolution}</strong>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Base frames:</span>
                    <strong className="text-slate-200 font-mono">{cfg.baseFrames}f</strong>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Tokens/chunk:</span>
                    <strong className="text-cyan-300 font-mono">{cfg.tokens.toLocaleString()}</strong>
                  </div>
                  <div className="flex justify-between text-slate-400">
                    <span>Target VRAM:</span>
                    <strong className="text-emerald-300 font-mono">{meta.vram}</strong>
                  </div>
                </div>
              </div>

              {/* Bottom footer description */}
              <div className="mt-4 pt-3 border-t border-slate-800/60 w-full">
                <p className="text-xs text-slate-400 line-clamp-2">
                  {meta.recommendedFor}
                </p>
              </div>
            </button>
          );
        })}
      </div>

      {/* Accordions for Deep Technical Details */}
      <div className="space-y-3 pt-2">
        {/* Accordion 1: DiT Parameter Breakdown */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('params')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'params'}
          >
            <div className="flex items-center gap-2.5">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Deep Technical Specs: {activeCfg.name} DiT Parameter Breakdown</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'params' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'params' && (
            <div className="px-5 pb-5 pt-2 border-t border-slate-800/60 text-sm text-slate-300 space-y-4">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block mb-1">Total Parameters</span>
                  <strong className="text-base font-bold text-white font-mono">
                    {(ditParams.totalParams / 1_000_000).toFixed(1)}M
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block mb-1">Layers &amp; Heads</span>
                  <strong className="text-base font-bold text-cyan-300 font-mono">
                    {activeDitCfg.numLayers}L / {activeDitCfg.numHeads}H
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block mb-1">Hidden Dim</span>
                  <strong className="text-base font-bold text-slate-200 font-mono">
                    {activeDitCfg.hiddenDim}D
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-slate-400 block mb-1">Modulation Savings</span>
                  <strong className="text-base font-bold text-emerald-300 font-mono">
                    &gt;80% (Shared MLP)
                  </strong>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Accordion 2: Spacetime Bucketing Geometry */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('buckets')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'buckets'}
          >
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              <span>Deep Technical Specs: Spacetime Bucketing Geometry</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'buckets' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'buckets' && (
            <div className="px-5 pb-5 pt-2 border-t border-slate-800/60 text-sm text-slate-300 space-y-3">
              <p className="text-xs text-slate-400">
                All training and inference buckets enforce both spatial divisibility by 16 and exact temporal congruence ($T \equiv 1 \pmod 4$).
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-2.5 text-xs">
                {['16:9 (Landscape)', '9:16 (Portrait)', '1:1 (Square)', '4:3 (Classic)', '3:4 (Tall)'].map((r, i) => (
                  <div key={i} className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-center">
                    <span className="font-semibold text-slate-200 block">{r.split(' ')[0]}</span>
                    <span className="text-[10px] text-slate-400">{r.split(' ')[1]}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
