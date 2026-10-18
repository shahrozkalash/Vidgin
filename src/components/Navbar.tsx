import React from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { Layers, Video, Cpu, Activity, Database, CheckCircle2, Sliders, PlayCircle } from 'lucide-react';

interface NavbarProps {
  currentPreset: PresetKey;
  onPresetChange: (preset: PresetKey) => void;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentPreset,
  onPresetChange,
  activeTab,
  onTabChange,
}) => {
  const preset = PRESETS[currentPreset];

  const navItems = [
    { id: 'batch_inspector', label: 'Visual Batch Dump', icon: PlayCircle, badge: 'Acceptance' },
    { id: 'bucketing', label: 'Spacetime Bucketing', icon: Layers, badge: '5 Ratios' },
    { id: 'sampler', label: 'Distributed Sampler', icon: Cpu, badge: 'Resumable' },
    { id: 'precompute', label: 'Precompute & Latents', icon: Database, badge: '16-Ch fp16' },
    { id: 'throughput', label: 'Throughput Benchmark', icon: Activity, badge: '>50 clips/s' },
    { id: 'specs_tests', label: 'Specs & Test Suite', icon: CheckCircle2, badge: 'Part 2' },
  ];

  return (
    <header className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Branding */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-sky-600 to-indigo-600 p-0.5 shadow-lg shadow-cyan-500/20">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Video className="w-5 h-5 text-cyan-400" />
              </div>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold tracking-tight text-white text-base">VidGen Studio</span>
                <span className="px-2 py-0.5 text-xs font-semibold rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
                  Part 2 Active
                </span>
                <span className="px-1.5 py-0.2 text-[10px] font-mono rounded bg-slate-800 text-slate-300">
                  Wan Latent Diffusion
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Multi-Aspect Bucketing • Causal 3D VAE Precompute • Deterministic Distributed Sampler
              </p>
            </div>
          </div>

          {/* Preset Selector */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 bg-slate-900/90 p-1 rounded-xl border border-slate-800">
              <Sliders className="w-3.5 h-3.5 text-slate-400 ml-1.5 hidden md:block" />
              {(['tiny', 'small', 'wan480p', 'wan14b'] as PresetKey[]).map((pKey) => {
                const isSelected = currentPreset === pKey;
                return (
                  <button
                    key={pKey}
                    onClick={() => onPresetChange(pKey)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-all ${
                      isSelected
                        ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
                    }`}
                  >
                    {PRESETS[pKey].name}
                  </button>
                );
              })}
            </div>

            {/* Quick Wan Verification Badges */}
            <div className="hidden lg:flex items-center gap-2">
              <div className="text-[11px] font-mono bg-slate-900 border border-emerald-500/30 px-2.5 py-1 rounded-lg text-emerald-400 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                T = 4k + 1 ({preset.baseFrames}f)
              </div>
              <div className="text-[11px] font-mono bg-slate-900 border border-sky-500/30 px-2 py-1 rounded-lg text-sky-400">
                (H,W)%16==0
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex overflow-x-auto space-x-1 py-1.5 no-scrollbar border-t border-slate-800/60">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onTabChange(item.id)}
                className={`flex items-center gap-2 px-3 py-2 text-xs font-medium rounded-lg whitespace-nowrap transition-all ${
                  isActive
                    ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/80'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
                {item.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                      isActive
                        ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/20'
                        : 'bg-slate-800/80 text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
