import React, { useState } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { calculateDiTParameters, DIT_PRESET_CONFIGS } from '../utils/ditEngine';
import { ChevronDown, Sliders, Check, ArrowLeftRight, FileDown, Search, RotateCcw, AlertCircle } from 'lucide-react';
import { PresetComparisonTable } from './PresetComparisonTable';
import { exportPresetAsJsonFile } from '../utils/exportPresetJson';
import {
  PresetSearchFilterBar,
  PRESET_FILTER_METADATA,
  MemoryFilterTier,
  HardwareFilterEnv,
} from './PresetSearchFilterBar';
import { ParameterTooltip } from './ParameterTooltip';

interface PresetsSectionProps {
  currentPreset: PresetKey;
  onPresetChange: (preset: PresetKey) => void;
}

export const PresetsSection: React.FC<PresetsSectionProps> = ({
  currentPreset,
  onPresetChange,
}) => {
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);
  const [showComparison, setShowComparison] = useState<boolean>(false);
  const [exportedPreset, setExportedPreset] = useState<PresetKey | null>(null);

  // Search & Filter state
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [memoryFilter, setMemoryFilter] = useState<MemoryFilterTier>('all');
  const [hardwareFilter, setHardwareFilter] = useState<HardwareFilterEnv>('all');

  const handleResetFilters = () => {
    setSearchQuery('');
    setMemoryFilter('all');
    setHardwareFilter('all');
  };

  const isFiltered = searchQuery.trim().length > 0 || memoryFilter !== 'all' || hardwareFilter !== 'all';

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const handleExport = (presetKey: PresetKey = currentPreset) => {
    exportPresetAsJsonFile(presetKey);
    setExportedPreset(presetKey);
    setTimeout(() => {
      setExportedPreset((prev) => (prev === presetKey ? null : prev));
    }, 2500);
  };

  const presetKeys: PresetKey[] = ['tiny', 'small', 'vidgen480p', 'vidgen9b', 'vidgen14b'];

  const filteredPresetKeys = presetKeys.filter((pk) => {
    const meta = PRESET_FILTER_METADATA[pk];
    const cfg = PRESETS[pk] || PRESETS.tiny;

    if (memoryFilter !== 'all' && meta.memoryTier !== memoryFilter) {
      return false;
    }

    if (hardwareFilter !== 'all' && meta.hardwareEnv !== hardwareFilter) {
      return false;
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      const tokens = q.split(/\s+/).filter(Boolean);
      const searchableText = [
        meta.name,
        meta.tagline,
        meta.resolution,
        meta.params,
        meta.vram,
        meta.hardwareLabel,
        meta.recommendedFor,
        meta.description,
        cfg.ditConfig,
        ...meta.keywords,
      ]
        .join(' ')
        .toLowerCase();

      return tokens.every((token) => searchableText.includes(token));
    }

    return true;
  });

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
    vidgen9b: {
      recommendedFor: 'Big Brain high-capacity video generation (≥9B parameters)',
      vram: '~38 GB (A100/H100)',
      speed: 'High-Capacity (~5.8s/step)',
      description: 'Massive 9.2B parameter architecture for high-capacity visual intelligence, temporal stability, and zero-shot prompt comprehension.',
    },
    vidgen14b: {
      recommendedFor: 'Big Brain 14 billion parameter flagship studio grade',
      vram: '~48 GB (Multi-GPU)',
      speed: 'Studio grade (~7.5s/step)',
      description: 'Massive 14.3B parameter flagship model for commercial photorealistic video, intricate spatial physics, and rich cinematic lighting.',
    },
  };

  const activeCfg = PRESETS[currentPreset] || PRESETS.tiny;
  const activeDitCfg = DIT_PRESET_CONFIGS[currentPreset] || DIT_PRESET_CONFIGS.tiny;
  const ditParams = calculateDiTParameters(activeDitCfg, currentPreset);

  return (
    <section id="presets" className="scroll-mt-20 space-y-6">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            Model Presets
          </h2>
          <p className="mt-1 text-base text-slate-300">
            Four calibrated tiers from fast CPU testing to flagship cinematic synthesis.
          </p>
        </div>

        {/* Action Controls: Export JSON & Compare Presets */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => handleExport(currentPreset)}
            title={`Export ${activeCfg.name} architecture configuration and performance metrics as JSON`}
            className="min-h-[44px] px-3.5 py-2 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 border border-slate-700 bg-slate-900/90 text-slate-200 hover:text-white hover:border-cyan-500/60 hover:bg-slate-800/80 transition-all active:scale-[0.98] cursor-pointer shadow-sm"
          >
            {exportedPreset === currentPreset ? (
              <>
                <Check className="w-4 h-4 text-emerald-400" />
                <span className="text-emerald-300 font-medium">Exported JSON!</span>
              </>
            ) : (
              <>
                <FileDown className="w-4 h-4 text-cyan-400" />
                <span>Export JSON</span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                  {currentPreset}
                </span>
              </>
            )}
          </button>

          <button
            onClick={() => setShowComparison((prev) => !prev)}
            aria-expanded={showComparison}
            className={`min-h-[44px] px-4 py-2 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 border transition-all cursor-pointer ${
              showComparison
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-md shadow-cyan-500/10 ring-1 ring-cyan-400/40'
                : 'bg-slate-900/90 border-slate-800 text-slate-300 hover:text-white hover:border-slate-700 hover:bg-slate-800/80'
            }`}
          >
            <ArrowLeftRight className="w-4 h-4 text-cyan-400" />
            <span>{showComparison ? 'Hide Comparison' : 'Compare Presets'}</span>
            <span
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                showComparison ? 'bg-cyan-400/20 text-cyan-200' : 'bg-slate-800 text-slate-400'
              }`}
            >
              Side-by-Side
            </span>
          </button>
        </div>
      </div>

      {/* Interactive Search & Filter Bar */}
      <PresetSearchFilterBar
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        memoryFilter={memoryFilter}
        onMemoryFilterChange={setMemoryFilter}
        hardwareFilter={hardwareFilter}
        onHardwareFilterChange={setHardwareFilter}
        totalCount={presetKeys.length}
        filteredCount={filteredPresetKeys.length}
        onResetFilters={handleResetFilters}
      />

      {/* Notice if current active preset is hidden by filter */}
      {isFiltered && !filteredPresetKeys.includes(currentPreset) && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3.5 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-amber-400 shrink-0" />
            <span>
              The currently selected preset <strong className="text-white">"{activeCfg.name}"</strong> is hidden by your active filters.
            </span>
          </div>
          <button
            onClick={handleResetFilters}
            className="text-amber-300 hover:text-white font-semibold underline cursor-pointer shrink-0"
          >
            Reset filters to view it
          </button>
        </div>
      )}

      {/* Side-by-Side Unified Comparison Table */}
      {showComparison && (
        <PresetComparisonTable
          initialPresetA={currentPreset}
          initialPresetB={currentPreset === 'vidgen480p' ? 'small' : 'vidgen480p'}
          activePreset={currentPreset}
          onSelectActivePreset={onPresetChange}
        />
      )}

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
            const isMatch = filteredPresetKeys.includes(pk);
            return (
              <option key={pk} value={pk}>
                {cfg.name} • {cfg.params} • {cfg.resolution} {isFiltered ? (isMatch ? '✓' : '(hidden by filter)') : ''}
              </option>
            );
          })}
        </select>
      </div>

      {/* Preset Cards Grid or Empty State */}
      {filteredPresetKeys.length === 0 ? (
        <div className="p-8 sm:p-10 rounded-2xl bg-slate-900/60 border border-slate-800 text-center space-y-4">
          <div className="w-12 h-12 rounded-full bg-slate-800/80 flex items-center justify-center mx-auto text-slate-400 border border-slate-700">
            <Search className="w-5 h-5 text-slate-400" />
          </div>
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white">No presets match your filters</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              No model presets found matching {searchQuery ? `"${searchQuery}"` : 'the selected memory footprint and hardware environment'}. Try adjusting your search term or clearing the active filters.
            </p>
          </div>
          <button
            onClick={handleResetFilters}
            className="min-h-[40px] px-4 py-2 rounded-xl text-xs font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-400/50 hover:bg-cyan-500/30 transition-all cursor-pointer inline-flex items-center gap-2 shadow-xs"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Search &amp; Filters</span>
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {/* Subtle guide cue for interactive tooltips */}
          <div className="flex items-center justify-between text-xs text-slate-400 px-1">
            <div className="flex items-center gap-1.5">
              <span>Interactive Specs: Hover or tap any parameter</span>
              <span className="text-cyan-400 font-medium">with an info icon</span>
              <span>to view its video quality and generation time impact.</span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {filteredPresetKeys.map((pk, idx) => {
              const cfg = PRESETS[pk] || PRESETS.tiny;
              const meta = presetDetails[pk];
              const filterMeta = PRESET_FILTER_METADATA[pk];
              const isSelected = currentPreset === pk;
              const align = idx >= 2 ? 'right' : 'left';

              return (
                <button
                  key={pk}
                  onClick={() => onPresetChange(pk)}
                  className={`min-h-[44px] p-5 rounded-2xl text-left transition-all flex flex-col justify-between border cursor-pointer relative ${
                    isSelected
                      ? 'bg-slate-900 border-cyan-400/90 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-400/50'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/90'
                  }`}
                >
                  <div className="w-full space-y-3">
                    {/* Header: Name & Selection indicator */}
                    <div className="flex items-center justify-between">
                      <ParameterTooltip param="params" side="bottom" align={align} showIcon={false}>
                        <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {cfg.params}
                        </span>
                      </ParameterTooltip>

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

                    {/* Specs List with interactive parameter tooltips */}
                    <div className="space-y-1.5 pt-2 border-t border-slate-800/80 text-xs">
                      <div className="flex justify-between items-center text-slate-400">
                        <ParameterTooltip param="resolution" align={align}>
                          <span>Resolution:</span>
                        </ParameterTooltip>
                        <strong className="text-slate-200 font-mono">{cfg.resolution}</strong>
                      </div>
                      <div className="flex justify-between items-center text-slate-400">
                        <ParameterTooltip param="baseFrames" align={align}>
                          <span>Base frames:</span>
                        </ParameterTooltip>
                        <strong className="text-slate-200 font-mono">{cfg.baseFrames}f</strong>
                      </div>
                      <div className="flex justify-between items-center text-slate-400">
                        <ParameterTooltip param="tokens" align={align}>
                          <span>Tokens/chunk:</span>
                        </ParameterTooltip>
                        <strong className="text-cyan-300 font-mono">{cfg.tokens.toLocaleString()}</strong>
                      </div>
                      <div className="flex justify-between items-center text-slate-400">
                        <ParameterTooltip param="vram" align={align}>
                          <span>Target VRAM:</span>
                        </ParameterTooltip>
                        <strong className="text-emerald-300 font-mono">{meta.vram}</strong>
                      </div>
                      <div className="flex justify-between items-center text-slate-400">
                        <ParameterTooltip param="hardware" align={align}>
                          <span>Hardware:</span>
                        </ParameterTooltip>
                        <strong
                          className="text-indigo-300 font-mono text-[11px] truncate max-w-[130px]"
                          title={filterMeta.hardwareLabel}
                        >
                          {filterMeta.hardwareLabel.split('(')[0].trim()}
                        </strong>
                      </div>
                    </div>
                  </div>

                  {/* Bottom footer description */}
                  <div className="mt-4 pt-3 border-t border-slate-800/60 w-full space-y-1">
                    <p className="text-[11px] font-semibold text-slate-300 truncate" title={filterMeta.hardwareLabel}>
                      {filterMeta.hardwareLabel}
                    </p>
                    <p className="text-xs text-slate-400 line-clamp-2">
                      {meta.recommendedFor}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}

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
                  <div className="mb-1">
                    <ParameterTooltip param="totalParams" align="left">
                      <span className="text-slate-400">Total Parameters</span>
                    </ParameterTooltip>
                  </div>
                  <strong className="text-base font-bold text-white font-mono block">
                    {(ditParams.totalParams / 1_000_000).toFixed(1)}M
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="mb-1">
                    <ParameterTooltip param="layersHeads" align="left">
                      <span className="text-slate-400">Layers &amp; Heads</span>
                    </ParameterTooltip>
                  </div>
                  <strong className="text-base font-bold text-cyan-300 font-mono block">
                    {activeDitCfg.numLayers}L / {activeDitCfg.numHeads}H
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="mb-1">
                    <ParameterTooltip param="hiddenDim" align="center">
                      <span className="text-slate-400">Hidden Dim</span>
                    </ParameterTooltip>
                  </div>
                  <strong className="text-base font-bold text-slate-200 font-mono block">
                    {activeDitCfg.hiddenDim}D
                  </strong>
                </div>
                <div className="p-3 rounded-xl bg-slate-950/70 border border-slate-800">
                  <div className="mb-1">
                    <ParameterTooltip param="modulation" align="right">
                      <span className="text-slate-400">Modulation Savings</span>
                    </ParameterTooltip>
                  </div>
                  <strong className="text-base font-bold text-emerald-300 font-mono block">
                    &gt;80% (Shared MLP)
                  </strong>
                </div>
              </div>

              {/* Export CTA within deep specs accordion */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-3 border-t border-slate-800/80">
                <span className="text-xs text-slate-400">
                  Export complete architecture hyperparameters, parameter breakdowns, and memory/FLOPs metrics:
                </span>
                <button
                  onClick={() => handleExport(currentPreset)}
                  className="min-h-[44px] px-3.5 py-2 rounded-xl text-xs font-semibold flex items-center justify-center gap-2 bg-slate-950 hover:bg-slate-800 text-cyan-300 border border-slate-700 hover:border-cyan-400 transition-colors cursor-pointer shrink-0"
                >
                  <FileDown className="w-3.5 h-3.5" />
                  <span>Download {activeCfg.name} JSON Specification</span>
                </button>
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
