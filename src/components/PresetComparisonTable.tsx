import React, { useState } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import { calculateDiTParameters, DIT_PRESET_CONFIGS } from '../utils/ditEngine';
import { estimateAttentionAnalytical, calculateLongVideoSpecs } from '../utils/longVideoEngine';
import { ArrowLeftRight, Check, AlertTriangle, ShieldCheck, Cpu, HardDrive, Zap, FileDown } from 'lucide-react';
import { exportPresetAsJsonFile } from '../utils/exportPresetJson';

interface PresetComparisonTableProps {
  initialPresetA?: PresetKey;
  initialPresetB?: PresetKey;
  onSelectActivePreset?: (preset: PresetKey) => void;
  activePreset?: PresetKey;
}

export const PresetComparisonTable: React.FC<PresetComparisonTableProps> = ({
  initialPresetA = 'small',
  initialPresetB = 'vidgen480p',
  onSelectActivePreset,
  activePreset,
}) => {
  const [presetA, setPresetA] = useState<PresetKey>(initialPresetA);
  const [presetB, setPresetB] = useState<PresetKey>(
    initialPresetB === initialPresetA ? (initialPresetA === 'vidgen480p' ? 'small' : 'vidgen480p') : initialPresetB
  );
  const [exportedKey, setExportedKey] = useState<PresetKey | null>(null);

  const handleExport = (key: PresetKey) => {
    exportPresetAsJsonFile(key);
    setExportedKey(key);
    setTimeout(() => {
      setExportedKey((prev) => (prev === key ? null : prev));
    }, 2500);
  };

  const presetKeys: PresetKey[] = ['tiny', 'small', 'vidgen480p', 'vidgen9b', 'vidgen14b'];

  const presetMeta: Record<
    PresetKey,
    {
      recommendedFor: string;
      vram: string;
      speed: string;
      tierLabel: string;
      targetHardware: string;
    }
  > = {
    tiny: {
      recommendedFor: 'Unit tests & CPU verification',
      vram: '< 1 GB',
      speed: '< 0.1s/step',
      tierLabel: 'CPU / CI Verification',
      targetHardware: 'Any Laptop CPU or Free T4',
    },
    small: {
      recommendedFor: 'Local prototyping & experimentation',
      vram: '~4 GB',
      speed: '~0.8s/step',
      tierLabel: 'Local Prototyping',
      targetHardware: 'Free Colab T4 (16GB) or RTX 3060',
    },
    vidgen480p: {
      recommendedFor: 'Production standard 30s synthesis',
      vram: '~18.4 GB',
      speed: '~2.4s/step',
      tierLabel: 'Production Standard',
      targetHardware: 'RTX 3090 / 4090 (24GB) or A10G',
    },
    vidgen9b: {
      recommendedFor: 'Big Brain high-capacity video generation (≥9B parameters)',
      vram: '~38 GB',
      speed: '~5.8s/step',
      tierLabel: 'Big Brain (9.2B)',
      targetHardware: 'A100 (80GB) or H100 GPU',
    },
    vidgen14b: {
      recommendedFor: 'Cinematic flagship photorealism',
      vram: '~48 GB',
      speed: '~7.5s/step',
      tierLabel: 'Cinematic Flagship',
      targetHardware: 'A100 (80GB) or H100 Cluster',
    },
  };

  const getStats = (key: PresetKey) => {
    const config = PRESETS[key] || PRESETS.tiny;
    const ditCfg = DIT_PRESET_CONFIGS[key] || DIT_PRESET_CONFIGS.tiny;
    const dit = calculateDiTParameters(ditCfg, key);
    const analytical = estimateAttentionAnalytical(30, 16, key);
    const chunks = calculateLongVideoSpecs(30, 16, key);
    const meta = presetMeta[key];

    return { key, config, ditCfg, dit, analytical, chunks, meta };
  };

  const statsA = getStats(presetA);
  const statsB = getStats(presetB);

  const swapPresets = () => {
    const temp = presetA;
    setPresetA(presetB);
    setPresetB(temp);
  };

  const formatFlops = (tflops: number) => {
    if (tflops >= 1000) {
      return `${(tflops / 1000).toFixed(2)} PFLOPs`;
    }
    if (tflops >= 1) {
      return `${tflops.toFixed(2)} TFLOPs`;
    }
    if (tflops >= 0.001) {
      return `${(tflops * 1000).toFixed(1)} GFLOPs`;
    }
    return `${(tflops * 1_000_000).toFixed(0)} MFLOPs`;
  };

  const formatMemoryGb = (gb: number) => {
    if (gb >= 1024) {
      return `${(gb / 1024).toFixed(2)} TB`;
    }
    if (gb >= 1) {
      return `${gb.toFixed(2)} GB`;
    }
    const mb = gb * 1024;
    if (mb >= 1) {
      return `${mb.toFixed(1)} MB`;
    }
    return `${(mb * 1024).toFixed(0)} KB`;
  };

  const calculateRatio = (valA: number, valB: number) => {
    if (valA === 0 || valB === 0) return '—';
    if (valA === valB) return '1.0x (Identical)';
    if (valB > valA) {
      const ratio = valB / valA;
      return ratio >= 100 ? `+${ratio.toFixed(0)}x` : `+${ratio.toFixed(1)}x`;
    } else {
      const ratio = valA / valB;
      return ratio >= 100 ? `-${ratio.toFixed(0)}x` : `-${ratio.toFixed(1)}x`;
    }
  };

  return (
    <div className="rounded-2xl bg-slate-900/90 border border-slate-800 p-4 sm:p-6 space-y-6 shadow-xl">
      {/* Top Header & Presets Selector Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-slate-800/80">
        <div>
          <div className="flex items-center gap-2">
            <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
              <ArrowLeftRight className="w-4 h-4" />
            </span>
            <h3 className="text-lg font-bold text-white tracking-tight">
              Preset Side-by-Side Comparison
            </h3>
          </div>
          <p className="mt-1 text-xs sm:text-sm text-slate-400">
            Compare analytical memory footprints, FLOPs compute, and scaling requirements for a 30-second video (481 frames @ 16 FPS).
          </p>
        </div>

        {/* Quick pair presets */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500 font-medium mr-1">Quick Pairs:</span>
          {[
            { label: 'Small vs 480p', a: 'small' as PresetKey, b: 'vidgen480p' as PresetKey },
            { label: '480p vs 9B (Big Brain)', a: 'vidgen480p' as PresetKey, b: 'vidgen9b' as PresetKey },
            { label: '9B vs 14B', a: 'vidgen9b' as PresetKey, b: 'vidgen14b' as PresetKey },
            { label: 'Tiny vs Small', a: 'tiny' as PresetKey, b: 'small' as PresetKey },
          ].map((pair, idx) => (
            <button
              key={idx}
              onClick={() => {
                setPresetA(pair.a);
                setPresetB(pair.b);
              }}
              className={`min-h-[36px] px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                presetA === pair.a && presetB === pair.b
                  ? 'bg-cyan-500/20 border-cyan-400/60 text-cyan-300'
                  : 'bg-slate-950/70 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
              }`}
            >
              {pair.label}
            </button>
          ))}
        </div>
      </div>

      {/* Selectors for Preset A and Preset B */}
      <div className="grid grid-cols-1 md:grid-cols-11 gap-3 items-center">
        {/* Preset A Picker */}
        <div className="md:col-span-5 p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-cyan-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              Preset A (Baseline)
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleExport(presetA)}
                title={`Export ${statsA.config.name} JSON architecture & metrics`}
                className="flex items-center gap-1 text-[11px] font-mono text-cyan-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-700/80 px-2 py-0.5 rounded transition-colors cursor-pointer"
              >
                {exportedKey === presetA ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-300">Exported</span>
                  </>
                ) : (
                  <>
                    <FileDown className="w-3 h-3 text-cyan-400" />
                    <span>JSON</span>
                  </>
                )}
              </button>
              {activePreset === presetA && (
                <span className="text-[11px] text-cyan-300 font-mono hidden sm:inline">Active</span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
            {presetKeys.map((pk) => (
              <button
                key={pk}
                onClick={() => setPresetA(pk)}
                className={`min-h-[40px] px-2 py-1.5 rounded-lg text-xs font-medium border text-center transition-all ${
                  presetA === pk
                    ? 'bg-cyan-950/80 border-cyan-400 text-white font-semibold shadow-sm shadow-cyan-500/20'
                    : 'bg-slate-900/50 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {PRESETS[pk]?.name}
              </button>
            ))}
          </div>
        </div>

        {/* Swap button */}
        <div className="md:col-span-1 flex justify-center">
          <button
            onClick={swapPresets}
            title="Swap Preset A and Preset B"
            aria-label="Swap Preset A and Preset B"
            className="min-h-[44px] min-w-[44px] p-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 transition-colors flex items-center justify-center cursor-pointer"
          >
            <ArrowLeftRight className="w-4 h-4" />
          </button>
        </div>

        {/* Preset B Picker */}
        <div className="md:col-span-5 p-3.5 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-indigo-400 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              Preset B (Comparison)
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleExport(presetB)}
                title={`Export ${statsB.config.name} JSON architecture & metrics`}
                className="flex items-center gap-1 text-[11px] font-mono text-indigo-300 hover:text-white bg-slate-900 hover:bg-slate-800 border border-slate-700/80 px-2 py-0.5 rounded transition-colors cursor-pointer"
              >
                {exportedKey === presetB ? (
                  <>
                    <Check className="w-3 h-3 text-emerald-400" />
                    <span className="text-emerald-300">Exported</span>
                  </>
                ) : (
                  <>
                    <FileDown className="w-3 h-3 text-indigo-400" />
                    <span>JSON</span>
                  </>
                )}
              </button>
              {activePreset === presetB && (
                <span className="text-[11px] text-indigo-300 font-mono hidden sm:inline">Active</span>
              )}
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
            {presetKeys.map((pk) => (
              <button
                key={pk}
                onClick={() => setPresetB(pk)}
                className={`min-h-[40px] px-2 py-1.5 rounded-lg text-xs font-medium border text-center transition-all ${
                  presetB === pk
                    ? 'bg-indigo-950/80 border-indigo-400 text-white font-semibold shadow-sm shadow-indigo-500/20'
                    : 'bg-slate-900/50 border-slate-800/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {PRESETS[pk]?.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Unified Table Container */}
      <div className="rounded-xl border border-slate-800 overflow-hidden bg-slate-950/60">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs sm:text-sm border-collapse min-w-[620px]">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-900/80 text-slate-300">
                <th scope="col" className="py-3.5 px-4 font-semibold w-2/5">
                  Specification &amp; Metric
                </th>
                <th scope="col" className="py-3.5 px-4 font-semibold text-cyan-300 w-1/4">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400" />
                    <span>{statsA.config.name}</span>
                    <span className="text-[11px] text-slate-400 font-normal">({statsA.config.params})</span>
                  </div>
                </th>
                <th scope="col" className="py-3.5 px-4 font-semibold text-indigo-300 w-1/4">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-indigo-400" />
                    <span>{statsB.config.name}</span>
                    <span className="text-[11px] text-slate-400 font-normal">({statsB.config.params})</span>
                  </div>
                </th>
                <th scope="col" className="py-3.5 px-4 font-semibold text-slate-400 w-[15%] text-right">
                  Ratio (B / A)
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-xs sm:text-sm">
              {/* Category 1: Architecture & Model Geometry */}
              <tr className="bg-slate-900/40 text-[11px] font-sans font-bold tracking-wider text-slate-400 uppercase">
                <td colSpan={4} className="py-2.5 px-4 flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                  <span>1. Architecture &amp; Spatial Geometry</span>
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  Native Resolution
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsA.config.resolution} ({statsA.config.height}×{statsA.config.width})
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsB.config.resolution} ({statsB.config.height}×{statsB.config.width})
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.config.height * statsA.config.width,
                    statsB.config.height * statsB.config.width
                  )} px
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  DiT Parameter Count
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {(statsA.dit.totalParams / 1_000_000).toFixed(1)}M
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {(statsB.dit.totalParams / 1_000_000).toFixed(1)}M
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(statsA.dit.totalParams, statsB.dit.totalParams)}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  Layers / Hidden Dim / Heads
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsA.ditCfg.numLayers}L / {statsA.ditCfg.hiddenDim}D / {statsA.ditCfg.numHeads}H
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsB.ditCfg.numLayers}L / {statsB.ditCfg.hiddenDim}D / {statsB.ditCfg.numHeads}H
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right font-sans text-xs">
                  {statsB.ditCfg.numLayers > statsA.ditCfg.numLayers
                    ? `+${statsB.ditCfg.numLayers - statsA.ditCfg.numLayers}L`
                    : statsB.ditCfg.numLayers === statsA.ditCfg.numLayers
                    ? 'Same depth'
                    : `-${statsA.ditCfg.numLayers - statsB.ditCfg.numLayers}L`}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  Spatial Tokens per Latent Frame
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsA.analytical.spatialTokensPerFrame.toLocaleString()}
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {statsB.analytical.spatialTokensPerFrame.toLocaleString()}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.spatialTokensPerFrame,
                    statsB.analytical.spatialTokensPerFrame
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  30s Sequence Length (Visual + Text)
                </td>
                <td className="py-2.5 px-4 text-cyan-300">
                  {statsA.analytical.totalSeqLen.toLocaleString()}
                </td>
                <td className="py-2.5 px-4 text-indigo-300">
                  {statsB.analytical.totalSeqLen.toLocaleString()}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(statsA.analytical.totalSeqLen, statsB.analytical.totalSeqLen)}
                </td>
              </tr>

              {/* Category 2: Memory Footprint Statistics */}
              <tr className="bg-slate-900/40 text-[11px] font-sans font-bold tracking-wider text-slate-400 uppercase">
                <td colSpan={4} className="py-2.5 px-4 flex items-center gap-2">
                  <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
                  <span>2. Memory Footprint (30-Second Sequence)</span>
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Full Unbounded Attention Matrix</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    Peak O(N²) attention matrix if fully materialized
                  </span>
                </td>
                <td className="py-2.5 px-4">
                  <div className="text-slate-200">
                    {formatMemoryGb(statsA.analytical.fullAttnMatrixGb)}
                  </div>
                  <div className="mt-1">
                    {statsA.analytical.exceedsT4Memory ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-rose-400 font-sans">
                        <AlertTriangle className="w-3 h-3" /> Exceeds 16GB T4
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-sans">
                        <ShieldCheck className="w-3 h-3" /> Fits on free T4
                      </span>
                    )}
                  </div>
                </td>
                <td className="py-2.5 px-4">
                  <div className="text-slate-200">
                    {formatMemoryGb(statsB.analytical.fullAttnMatrixGb)}
                  </div>
                  <div className="mt-1">
                    {statsB.analytical.exceedsT4Memory ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-rose-400 font-sans">
                        <AlertTriangle className="w-3 h-3" /> Exceeds 16GB T4
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-400 font-sans">
                        <ShieldCheck className="w-3 h-3" /> Fits on free T4
                      </span>
                    )}
                  </div>
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.fullAttnMatrixGb,
                    statsB.analytical.fullAttnMatrixGb
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Decoupled Attention Peak Matrix</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    Spatial + temporal separated attention
                  </span>
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatMemoryGb(statsA.analytical.decoupledAttnMatrixGb)}
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatMemoryGb(statsB.analytical.decoupledAttnMatrixGb)}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.decoupledAttnMatrixGb,
                    statsB.analytical.decoupledAttnMatrixGb
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Chunked Streaming Working Memory</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    Bounded 5-second overlapping chunk working VRAM
                  </span>
                </td>
                <td className="py-2.5 px-4 text-emerald-300 font-semibold">
                  {statsA.chunks.chunkedKvVramGb} GB
                </td>
                <td className="py-2.5 px-4 text-emerald-300 font-semibold">
                  {statsB.chunks.chunkedKvVramGb} GB
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.chunks.chunkedKvVramGb,
                    statsB.chunks.chunkedKvVramGb
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  Target Hardware Specification
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-300">
                  {statsA.meta.targetHardware}
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-300">
                  {statsB.meta.targetHardware}
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-500 text-right">
                  {statsA.meta.vram} vs {statsB.meta.vram}
                </td>
              </tr>

              {/* Category 3: FLOPs & Compute Complexity Statistics */}
              <tr className="bg-slate-900/40 text-[11px] font-sans font-bold tracking-wider text-slate-400 uppercase">
                <td colSpan={4} className="py-2.5 px-4 flex items-center gap-2">
                  <Zap className="w-3.5 h-3.5 text-amber-400" />
                  <span>3. Compute Complexity &amp; FLOPs (30-Second Sequence)</span>
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Full Unbounded Attention FLOPs</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    Formula: 4 × N² × D × L
                  </span>
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatFlops(statsA.analytical.fullAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatFlops(statsB.analytical.fullAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.fullAttnFlopsTFlops,
                    statsB.analytical.fullAttnFlopsTFlops
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Decoupled Spatial+Temporal FLOPs</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    Separated 2D space + 1D temporal passes
                  </span>
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatFlops(statsA.analytical.decoupledAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-slate-200">
                  {formatFlops(statsB.analytical.decoupledAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.decoupledAttnFlopsTFlops,
                    statsB.analytical.decoupledAttnFlopsTFlops
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  <div>Chunked Streaming FLOPs</div>
                  <span className="text-[11px] font-sans text-slate-500 block">
                    6 sliding chunks with overlapping latents
                  </span>
                </td>
                <td className="py-2.5 px-4 text-cyan-300 font-semibold">
                  {formatFlops(statsA.analytical.chunkedAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-indigo-300 font-semibold">
                  {formatFlops(statsB.analytical.chunkedAttnFlopsTFlops)}
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right">
                  {calculateRatio(
                    statsA.analytical.chunkedAttnFlopsTFlops,
                    statsB.analytical.chunkedAttnFlopsTFlops
                  )}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  FLOPs Complexity Reduction
                </td>
                <td className="py-2.5 px-4 text-emerald-400 font-semibold">
                  {statsA.chunks.attentionComplexityReduction}x reduction
                </td>
                <td className="py-2.5 px-4 text-emerald-400 font-semibold">
                  {statsB.chunks.attentionComplexityReduction}x reduction
                </td>
                <td className="py-2.5 px-4 text-slate-400 text-right font-sans text-xs">
                  {statsB.chunks.attentionComplexityReduction > statsA.chunks.attentionComplexityReduction
                    ? `+${(statsB.chunks.attentionComplexityReduction - statsA.chunks.attentionComplexityReduction).toFixed(1)}x greater savings`
                    : 'Balanced'}
                </td>
              </tr>
              <tr className="hover:bg-slate-900/30 transition-colors">
                <td className="py-2.5 px-4 font-sans text-slate-300 font-medium">
                  Sampling Speed / Step Latency
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-300">
                  {statsA.meta.speed}
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-300">
                  {statsB.meta.speed}
                </td>
                <td className="py-2.5 px-4 font-sans text-xs text-slate-500 text-right">
                  {statsA.config.name} is faster
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Action / Activate Preset Buttons */}
      {onSelectActivePreset && (
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 text-xs">
          <span className="text-slate-400">
            Set active studio preset for the rest of the dashboard:
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => onSelectActivePreset(presetA)}
              className={`min-h-[40px] px-3.5 py-2 rounded-xl font-medium border flex items-center gap-1.5 transition-all ${
                activePreset === presetA
                  ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300'
                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
              }`}
            >
              {activePreset === presetA && <Check className="w-3.5 h-3.5" />}
              <span>Use {statsA.config.name} as Active</span>
            </button>
            <button
              onClick={() => onSelectActivePreset(presetB)}
              className={`min-h-[40px] px-3.5 py-2 rounded-xl font-medium border flex items-center gap-1.5 transition-all ${
                activePreset === presetB
                  ? 'bg-indigo-500/20 border-indigo-400 text-indigo-300'
                  : 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
              }`}
            >
              {activePreset === presetB && <Check className="w-3.5 h-3.5" />}
              <span>Use {statsB.config.name} as Active</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
