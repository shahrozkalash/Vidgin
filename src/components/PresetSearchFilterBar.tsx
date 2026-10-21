import React from 'react';
import { Search, X, RotateCcw, HardDrive, Cpu, Filter, Check } from 'lucide-react';
import { PresetKey } from '../types/vidgen';

export type MemoryFilterTier = 'all' | 'under_8gb' | '8_to_24gb' | 'over_24gb';
export type HardwareFilterEnv = 'all' | 'cpu' | 'consumer_gpu' | 'pro_gpu' | 'multi_gpu';

export interface PresetFilterPresetMeta {
  key: PresetKey;
  name: string;
  tagline: string;
  resolution: string;
  params: string;
  vram: string;
  vramNumberGb: number;
  memoryTier: MemoryFilterTier;
  hardwareEnv: HardwareFilterEnv;
  hardwareLabel: string;
  recommendedFor: string;
  description: string;
  keywords: string[];
}

export const PRESET_FILTER_METADATA: Record<PresetKey, PresetFilterPresetMeta> = {
  tiny: {
    key: 'tiny',
    name: 'Tiny',
    tagline: 'Track A Prototyping & Verification',
    resolution: '64x64',
    params: '~20M',
    vram: '< 1 GB',
    vramNumberGb: 0.3,
    memoryTier: 'under_8gb',
    hardwareEnv: 'cpu',
    hardwareLabel: 'CPU / Edge / CI/CD',
    recommendedFor: 'Unit tests & CPU verification',
    description: 'Ideal for local debugging and CI/CD pipelines without GPU requirements.',
    keywords: ['cpu', 'ci/cd', 'test', 'verification', 'prototype', 'edge', 'minimal', 'tiny', 'local', 'debug', 'free'],
  },
  small: {
    key: 'small',
    name: 'Small',
    tagline: 'Medium Scale Verification',
    resolution: '256x256',
    params: '~270M',
    vram: '~4 GB',
    vramNumberGb: 4.1,
    memoryTier: 'under_8gb',
    hardwareEnv: 'consumer_gpu',
    hardwareLabel: 'Consumer GPU / Colab T4',
    recommendedFor: 'Local development & prototyping',
    description: 'Balanced size for testing architectural modifications and reward loops.',
    keywords: ['colab', 't4', 'gpu', 'consumer', 'rtx', 'apple silicon', 'macbook', 'medium', 'small', '4gb', 'development'],
  },
  vidgen480p: {
    key: 'vidgen480p',
    name: 'Vidgen 480p',
    tagline: 'Standard Production DiT',
    resolution: '480x832',
    params: '~1.3B',
    vram: '~18.4 GB',
    vramNumberGb: 18.4,
    memoryTier: '8_to_24gb',
    hardwareEnv: 'pro_gpu',
    hardwareLabel: 'Pro GPU (A100 / RTX 4090)',
    recommendedFor: 'Production high-fidelity standard',
    description: 'Our primary 1.3B workhorse model capable of sustained 30-second clips.',
    keywords: ['480p', '1.3b', 'production', 'workhorse', 'pro gpu', 'a100', 'a10g', '4090', 'workstation', 'standard', '18gb'],
  },
  vidgen9b: {
    key: 'vidgen9b',
    name: 'Vidgen 9B (Big Brain)',
    tagline: 'High-Capacity 9.2B Spacetime DiT',
    resolution: '720x1280',
    params: '~9.2B',
    vram: '~38 GB (A100/H100)',
    vramNumberGb: 38,
    memoryTier: 'over_24gb',
    hardwareEnv: 'multi_gpu',
    hardwareLabel: 'Datacenter GPU (A100 / H100)',
    recommendedFor: 'Big Brain high-capacity video generation (≥9B parameters)',
    description: 'High-capacity 9.2B parameter architecture for deep spatio-temporal reasoning, fluid narrative stability, and zero-shot prompt comprehension.',
    keywords: ['9b', 'big brain', '9 billion', 'flagship', 'a100', 'h100', 'cluster', 'photorealistic', '720p', 'cinema'],
  },
  vidgen14b: {
    key: 'vidgen14b',
    name: 'Vidgen 14B (Big Brain Flagship)',
    tagline: 'Flagship 14.3B Spacetime DiT',
    resolution: '720x1280',
    params: '~14.3B',
    vram: '~48 GB (Multi-GPU)',
    vramNumberGb: 48,
    memoryTier: 'over_24gb',
    hardwareEnv: 'multi_gpu',
    hardwareLabel: 'Multi-GPU Cluster (H100 / DGX)',
    recommendedFor: 'Big Brain flagship 14 billion parameter studio grade synthesis',
    description: 'Massive 14.3B parameter flagship architecture delivering photorealistic motion graphics, complex cinematic physics, and multi-actor interaction.',
    keywords: ['14b', '14 billion', 'big brain', 'cinematic', 'flagship', 'cluster', 'multi-gpu', 'h100', 'dgx', 'studio', '48gb', 'photorealistic', '720p'],
  },
};

interface PresetSearchFilterBarProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  memoryFilter: MemoryFilterTier;
  onMemoryFilterChange: (tier: MemoryFilterTier) => void;
  hardwareFilter: HardwareFilterEnv;
  onHardwareFilterChange: (env: HardwareFilterEnv) => void;
  totalCount: number;
  filteredCount: number;
  onResetFilters: () => void;
}

export const PresetSearchFilterBar: React.FC<PresetSearchFilterBarProps> = ({
  searchQuery,
  onSearchChange,
  memoryFilter,
  onMemoryFilterChange,
  hardwareFilter,
  onHardwareFilterChange,
  totalCount,
  filteredCount,
  onResetFilters,
}) => {
  const isFiltered = searchQuery.trim().length > 0 || memoryFilter !== 'all' || hardwareFilter !== 'all';

  const memoryOptions: { value: MemoryFilterTier; label: string; badge: string }[] = [
    { value: 'all', label: 'All Memory', badge: 'Any VRAM' },
    { value: 'under_8gb', label: '< 8 GB', badge: 'Tiny & Small' },
    { value: '8_to_24gb', label: '8 – 24 GB', badge: 'Vidgen 480p' },
    { value: 'over_24gb', label: '24 GB+', badge: 'Vidgen 14B Multi-GPU' },
  ];

  const hardwareOptions: { value: HardwareFilterEnv; label: string; badge: string }[] = [
    { value: 'all', label: 'All Environments', badge: 'Any HW' },
    { value: 'cpu', label: 'CPU / Edge / CI', badge: 'Tiny' },
    { value: 'consumer_gpu', label: 'Consumer / T4', badge: 'Small' },
    { value: 'pro_gpu', label: 'Pro Single-GPU', badge: 'Vidgen 480p' },
    { value: 'multi_gpu', label: 'Multi-GPU Cluster', badge: 'Vidgen 14B' },
  ];

  return (
    <div className="p-4 sm:p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-sm space-y-4">
      {/* Top row: Search input + Results counter + Reset button */}
      <div className="flex flex-col md:flex-row md:items-center gap-3 justify-between">
        {/* Search input field */}
        <div className="relative flex-1">
          <label htmlFor="preset-search-input" className="sr-only">
            Search presets
          </label>
          <div className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
            <Search className="w-4 h-4" />
          </div>
          <input
            id="preset-search-input"
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search by name, resolution, params (e.g. 1.3B), hardware (T4, CPU, A100), or VRAM..."
            className="w-full min-h-[44px] pl-10 pr-10 py-2.5 rounded-xl bg-slate-950 border border-slate-800 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/60 focus:border-cyan-400/80 transition-all"
          />
          {searchQuery && (
            <button
              onClick={() => onSearchChange('')}
              aria-label="Clear search input"
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-white transition-colors cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Counter & Reset control */}
        <div className="flex items-center justify-between md:justify-end gap-3 shrink-0">
          <div className="text-xs text-slate-400 font-mono">
            Showing <span className="text-cyan-300 font-bold">{filteredCount}</span> of{' '}
            <span className="text-slate-200">{totalCount}</span> presets
          </div>

          {isFiltered && (
            <button
              onClick={onResetFilters}
              className="min-h-[36px] px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-cyan-300 hover:text-white border border-slate-700 transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Filters</span>
            </button>
          )}
        </div>
      </div>

      {/* Filter selectors: Memory Footprint & Hardware Environment */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
        {/* Filter 1: Memory Footprint */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 font-semibold text-slate-300">
              <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
              <span>Memory Footprint (VRAM)</span>
            </span>
            {memoryFilter !== 'all' && (
              <button
                onClick={() => onMemoryFilterChange('all')}
                className="text-[11px] text-cyan-400 hover:underline cursor-pointer"
              >
                Clear
              </button>
            )}
          </div>

          {/* Segmented options buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800/90">
            {memoryOptions.map((opt) => {
              const isActive = memoryFilter === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => onMemoryFilterChange(opt.value)}
                  className={`min-h-[36px] px-2 py-1.5 rounded-lg text-xs font-medium transition-all text-center flex flex-col items-center justify-center cursor-pointer ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-200 border border-cyan-400/50 shadow-xs'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                  }`}
                  title={`${opt.label} (${opt.badge})`}
                >
                  <span className="font-semibold">{opt.label}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Filter 2: Intended Hardware Environment */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="flex items-center gap-1.5 font-semibold text-slate-300">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
              <span>Intended Hardware Environment</span>
            </span>
            {hardwareFilter !== 'all' && (
              <button
                onClick={() => onHardwareFilterChange('all')}
                className="text-[11px] text-indigo-400 hover:underline cursor-pointer"
              >
                Clear
              </button>
            )}
          </div>

          {/* Segmented options buttons */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800/90">
            {hardwareOptions.map((opt) => {
              const isActive = hardwareFilter === opt.value;
              return (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => onHardwareFilterChange(opt.value)}
                  className={`min-h-[36px] px-1.5 py-1.5 rounded-lg text-xs font-medium transition-all text-center flex flex-col items-center justify-center cursor-pointer ${
                    isActive
                      ? 'bg-indigo-500/20 text-indigo-200 border border-indigo-400/50 shadow-xs'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                  }`}
                  title={`${opt.label} (${opt.badge})`}
                >
                  <span className="font-semibold truncate w-full text-center">{opt.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Quick Filter Shortcuts row */}
      <div className="pt-2 border-t border-slate-800/60 flex flex-wrap items-center gap-2 text-xs">
        <span className="text-slate-400 flex items-center gap-1 text-[11px]">
          <Filter className="w-3 h-3 text-slate-400" />
          <span>Quick Filters:</span>
        </span>

        <button
          onClick={() => {
            onSearchChange('');
            onMemoryFilterChange('under_8gb');
            onHardwareFilterChange('consumer_gpu');
          }}
          className="min-h-[30px] px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-cyan-300 transition-colors cursor-pointer"
        >
          Free Colab T4 (Small)
        </button>

        <button
          onClick={() => {
            onSearchChange('');
            onMemoryFilterChange('under_8gb');
            onHardwareFilterChange('cpu');
          }}
          className="min-h-[30px] px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-emerald-300 transition-colors cursor-pointer"
        >
          CPU Prototyping (Tiny)
        </button>

        <button
          onClick={() => {
            onSearchChange('');
            onMemoryFilterChange('8_to_24gb');
            onHardwareFilterChange('pro_gpu');
          }}
          className="min-h-[30px] px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-indigo-300 transition-colors cursor-pointer"
        >
          Production DiT (Vidgen 480p)
        </button>

        <button
          onClick={() => {
            onSearchChange('');
            onMemoryFilterChange('over_24gb');
            onHardwareFilterChange('multi_gpu');
          }}
          className="min-h-[30px] px-2.5 py-1 rounded-md text-[11px] font-medium bg-slate-950 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 hover:text-purple-300 transition-colors cursor-pointer"
        >
          Cinematic Cluster (Vidgen 14B)
        </button>
      </div>
    </div>
  );
};
