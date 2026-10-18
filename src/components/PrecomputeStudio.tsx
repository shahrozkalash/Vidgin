import React, { useState, useMemo } from 'react';
import { PresetKey, LatentStats, PrecomputeShard } from '../types/vidgen';
import { PRESETS, getStandardBuckets, calculateChannelStats } from '../utils/bucketEngine';
import {
  Database,
  BarChart3,
  Layers,
  Sparkles,
  FileCode,
  CheckCircle2,
  Download,
  Flame,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';

interface PrecomputeStudioProps {
  currentPreset: PresetKey;
}

export const PrecomputeStudio: React.FC<PrecomputeStudioProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const [stats, setStats] = useState<LatentStats>(() => calculateChannelStats(16));
  const [activeVariant, setActiveVariant] = useState<'latents_orig' | 'latents_hflip' | 'latents_crop2'>('latents_orig');
  const [simulatedDropout, setSimulatedDropout] = useState<boolean>(false);
  const [isExportingTar, setIsExportingTar] = useState<boolean>(false);

  // Generate 8 sample shards
  const shards: PrecomputeShard[] = useMemo(() => {
    const t = Math.floor((preset.baseFrames - 1) / 4) + 1;
    const h = preset.key === 'tiny' ? 8 : preset.key === 'small' ? 32 : 60;
    const w = preset.key === 'tiny' ? 8 : preset.key === 'small' ? 32 : 104;
    const textDim = preset.key === 'tiny' ? 384 : preset.key === 'small' ? 1024 : 1536;

    return Array.from({ length: 6 }).map((_, idx) => ({
      id: `clip_${String(idx + 1).padStart(4, '0')}`,
      latentShape: [16, t, h, w],
      dtypeLatent: 'float16',
      dtypeText: 'bfloat16',
      textEmbDim: textDim,
      variants: ['latents_orig', 'latents_hflip', 'latents_crop2'],
      shardSizeKb: Math.round((16 * t * h * w * 2 * 3) / 1024 + 48),
    }));
  }, [preset]);

  const testPrompt = 'A vibrant glowing neon digit drifting gracefully across coordinate space.';

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-amber-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-amber-500/10 text-amber-400">
                <Database className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Part 2 Precompute Engine & 16-Channel Normalization
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Eliminate CPU VAE and text encoder bottlenecks by precomputing latents in{' '}
              <code className="text-amber-300 font-mono">fp16</code> and text embeddings in{' '}
              <code className="text-cyan-300 font-mono">bf16</code>. Sharded into memory-mapped{' '}
              <code className="text-cyan-300 font-mono">.safetensors</code> or WebDataset tars with stored
              flip & crop variants and per-channel normalization $(\mu, \sigma)$.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono bg-slate-950 text-slate-300 px-3 py-1.5 rounded-xl border border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Memory-Mapped Zero-Copy
            </span>
          </div>
        </div>
      </div>

      {/* Grid: 16-Channel Stats Normalizer & Shard Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: 16-Channel Latent Statistics Accumulator */}
        <div className="lg:col-span-7 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-white">
                Per-Channel Latent Normalization (16 Channels)
              </h3>
            </div>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
              Sampled: {stats.totalElementsSampled.toLocaleString()} elements
            </span>
          </div>

          {/* Normalization Formula */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center justify-between text-xs font-mono">
            <span className="text-slate-400">Normalization Law:</span>
            <span className="text-cyan-300 font-bold">
              z_norm = (z_raw - mean[c]) / std[c]
            </span>
            <span className="text-[10px] text-emerald-400">Zero-mean, Unit-variance</span>
          </div>

          {/* Channel Means & Stds Bar Chart */}
          <div className="space-y-1.5 pt-2">
            <div className="grid grid-cols-16 text-[9px] font-mono text-slate-500 text-center mb-1">
              {stats.mean.map((_, c) => (
                <span key={c}>c{c}</span>
              ))}
            </div>

            {/* Mean Distribution Bars */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800/80 space-y-3">
              <div>
                <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                  <span>Channel Means (μ):</span>
                  <span className="text-slate-500">Center at ~0.0</span>
                </div>
                <div className="grid grid-cols-16 gap-1 h-12 items-center">
                  {stats.mean.map((m, c) => {
                    const heightPercent = Math.min(100, Math.abs(m) * 400 + 10);
                    const isPositive = m >= 0;
                    return (
                      <div
                        key={c}
                        className="flex flex-col items-center justify-center h-full group relative"
                      >
                        <div
                          className={`w-full rounded-sm transition-all ${
                            isPositive ? 'bg-cyan-400' : 'bg-rose-400'
                          }`}
                          style={{ height: `${heightPercent}%` }}
                        ></div>
                        <div className="absolute -top-6 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-[8px] font-mono text-white opacity-0 group-hover:opacity-100 pointer-events-none z-10">
                          {m.toFixed(3)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Std Distribution Bars */}
              <div>
                <div className="flex justify-between text-[11px] font-mono text-slate-400 mb-1">
                  <span>Channel Standard Deviations (σ):</span>
                  <span className="text-slate-500">Normalizes to 1.0</span>
                </div>
                <div className="grid grid-cols-16 gap-1 h-12 items-end">
                  {stats.std.map((s, c) => {
                    const heightPercent = Math.min(100, (s / 1.5) * 100);
                    return (
                      <div
                        key={c}
                        className="flex flex-col items-center justify-end h-full group relative"
                      >
                        <div
                          className="w-full bg-emerald-400 rounded-sm transition-all"
                          style={{ height: `${heightPercent}%` }}
                        ></div>
                        <div className="absolute -top-6 bg-slate-900 border border-slate-700 px-1 py-0.5 rounded text-[8px] font-mono text-white opacity-0 group-hover:opacity-100 pointer-events-none z-10">
                          {s.toFixed(3)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Export metadata json preview */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-[11px] font-mono text-slate-400 flex items-center justify-between">
            <span>Saved To Disk:</span>
            <span className="text-cyan-300">checkpoints/{preset.key}/latent_stats.json</span>
            <span className="text-emerald-400 font-bold">16 Channels Saved</span>
          </div>
        </div>

        {/* Right: Shards, Stored Variants & 10% CFG Dropout Simulation */}
        <div className="lg:col-span-5 space-y-4">
          {/* Precomputed Shard & Variants Card */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <FileCode className="w-4 h-4 text-amber-400" />
                <h3 className="text-sm font-bold text-white">
                  .safetensors Precomputed Shard
                </h3>
              </div>
              <span className="text-[10px] font-mono text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/40">
                fp16 + bf16
              </span>
            </div>

            {/* Stored Augmentation Variants in safetensors */}
            <div className="space-y-2">
              <div className="text-xs font-semibold text-slate-300">
                Stored Pixel Augmentation Variants (Zero Encode Time at Training):
              </div>
              <div className="grid grid-cols-3 gap-2">
                {(['latents_orig', 'latents_hflip', 'latents_crop2'] as const).map((v) => (
                  <button
                    key={v}
                    onClick={() => setActiveVariant(v)}
                    className={`p-2 rounded-xl text-center text-xs font-mono transition-all ${
                      activeVariant === v
                        ? 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/20'
                        : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-white border border-slate-800'
                    }`}
                  >
                    {v === 'latents_orig' ? 'Original' : v === 'latents_hflip' ? 'H-Flip' : 'Crop #2'}
                  </button>
                ))}
              </div>
            </div>

            {/* Tensor Shapes in Shard */}
            <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80 space-y-2 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">{activeVariant}:</span>
                <span className="text-amber-300 font-bold">
                  [16, {Math.floor((preset.baseFrames - 1) / 4) + 1}, {preset.key === 'tiny' ? 8 : 32}, {preset.key === 'tiny' ? 8 : 32}] (fp16)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">text_emb:</span>
                <span className="text-cyan-300 font-bold">
                  [16, {preset.key === 'tiny' ? 384 : 1024}] (bf16)
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">text_null_emb:</span>
                <span className="text-slate-300 font-bold">
                  [16, {preset.key === 'tiny' ? 384 : 1024}] (bf16 Empty)
                </span>
              </div>
            </div>

            {/* 10% CFG Dropout Simulation */}
            <div className="pt-2 border-t border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300">
                  10% CFG Null-Prompt Embedding Drop:
                </span>
                <button
                  onClick={() => setSimulatedDropout(!simulatedDropout)}
                  className={`px-2.5 py-1 text-xs font-mono font-bold rounded-lg transition-all ${
                    simulatedDropout
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                  }`}
                >
                  {simulatedDropout ? 'Swap Active (Empty)' : 'Normal Prompt'}
                </button>
              </div>

              <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800 text-xs font-mono">
                <div className="text-[10px] text-slate-500 mb-0.5">Input to Cross-Attention:</div>
                <div className={simulatedDropout ? 'text-amber-400 font-bold' : 'text-slate-300'}>
                  {simulatedDropout
                    ? '⇄ Swapped in precomputed text_null_emb (Empty string vector)'
                    : `"${testPrompt}"`}
                </div>
              </div>
            </div>

            {/* WebDataset TAR Pack Button */}
            <div className="pt-2">
              <button
                onClick={() => {
                  setIsExportingTar(true);
                  setTimeout(() => setIsExportingTar(false), 800);
                }}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-xl border border-slate-700 transition-all cursor-pointer"
              >
                <Download className="w-3.5 h-3.5 text-cyan-400" />
                <span>
                  {isExportingTar ? 'Packing Shards to WebDataset TAR...' : 'Export as WebDataset TAR Archive'}
                </span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
