import React, { useState, useMemo } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import {
  calculateLongVideoSpecs,
  estimateAttentionAnalytical,
  AttentionAnalyticalEstimate,
} from '../utils/longVideoEngine';
import { MemoryTracker } from './MemoryTracker';
import { ChunkOverlapVisualizer } from './ChunkOverlapVisualizer';
import {
  Film,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  Sliders,
  HelpCircle,
  Cpu,
} from 'lucide-react';

interface LongVideoSectionProps {
  currentPreset: PresetKey;
}

export const LongVideoSection: React.FC<LongVideoSectionProps> = ({ currentPreset }) => {
  const [targetDuration, setTargetDuration] = useState<number>(30); // 5 to 60s
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);
  const [showFormula, setShowFormula] = useState<boolean>(false);

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const specs = calculateLongVideoSpecs(targetDuration, 16, currentPreset);
  const cfg = PRESETS[currentPreset] || PRESETS.tiny;

  // Real-time analytical attention calculation matching python estimate_attention_flops_and_memory
  const attnEst: AttentionAnalyticalEstimate = useMemo(() => {
    return estimateAttentionAnalytical(targetDuration, 16, currentPreset);
  }, [targetDuration, currentPreset]);

  // Format helpers for readable units
  const formatMem = (gb: number, mb: number) => {
    if (gb >= 1.0) {
      return `${gb.toFixed(1)} GB`;
    }
    return `${mb.toFixed(1)} MB (${gb.toFixed(3)} GB)`;
  };

  const formatFlops = (tflops: number) => {
    if (tflops >= 1.0) {
      return `${tflops.toFixed(1)} TFLOPs`;
    }
    const gflops = tflops * 1000;
    return `${gflops.toFixed(1)} GFLOPs`;
  };

  return (
    <section id="long_video" className="scroll-mt-20 space-y-6">
      {/* Section Header with 30-Second Video badge moved here from Navbar */}
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

        {/* Dynamic Memory & Compute Comparison Banner */}
        <div className="p-4 sm:p-5 rounded-2xl bg-slate-950 border border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
          {/* Card 1: Unbounded Full Attention (Computed dynamically from Python estimator) */}
          <div
            className={`p-4 sm:p-5 rounded-xl border flex flex-col justify-between space-y-3.5 transition-all ${
              attnEst.exceedsT4Memory
                ? 'bg-rose-950/20 border-rose-900/50'
                : 'bg-slate-900/90 border-slate-800'
            }`}
          >
            <div className="space-y-3">
              {/* Header with Title, Preset tag, and T4 Warning/Fit Status */}
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                    Unbounded Full Attention
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-cyan-300 border border-slate-700">
                    {attnEst.presetName}
                  </span>
                </div>

                {/* Warning message only if > 16 GB, otherwise neutral green message */}
                {attnEst.exceedsT4Memory ? (
                  <span className="text-xs font-mono font-bold text-rose-300 bg-rose-950/80 border border-rose-800/80 px-2.5 py-1 rounded-full flex items-center gap-1.5 shadow-sm">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                    <span>Exceeds GPU memory (&gt; 16 GB Colab T4)</span>
                  </span>
                ) : (
                  <span className="text-xs font-mono font-semibold text-emerald-300 bg-emerald-950/80 border border-emerald-800/80 px-2.5 py-1 rounded-full flex items-center gap-1.5 shadow-sm">
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    <span>Fits on a free T4 (≤ 16 GB)</span>
                  </span>
                )}
              </div>

              {/* Requirement 2: Two computed lines per preset */}
              <div className="space-y-2 pt-1 font-mono text-xs">
                {/* Line 1: Attention memory if the full matrix is stored */}
                <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 flex flex-wrap items-baseline justify-between gap-1">
                  <span className="text-slate-400 font-sans text-xs">Attention Matrix Memory (stored full matrix):</span>
                  <strong
                    className={`text-sm font-bold ${
                      attnEst.exceedsT4Memory ? 'text-rose-400' : 'text-emerald-300'
                    }`}
                  >
                    {formatMem(attnEst.fullAttnMatrixGb, attnEst.fullAttnMatrixMb)}
                  </strong>
                </div>

                {/* Line 2: Compute (FLOPs) for full vs decoupled vs chunked attention */}
                <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 space-y-1">
                  <span className="text-slate-400 font-sans text-xs block">Compute (FLOPs):</span>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                    <span className="text-slate-300">
                      Full: <strong className="text-rose-400">{formatFlops(attnEst.fullAttnFlopsTFlops)}</strong>
                    </span>
                    <span className="text-slate-500">vs</span>
                    <span className="text-slate-300">
                      Decoupled: <strong className="text-sky-300">{formatFlops(attnEst.decoupledAttnFlopsTFlops)}</strong>
                    </span>
                    <span className="text-slate-500">vs</span>
                    <span className="text-slate-300">
                      Chunked: <strong className="text-emerald-300">{formatFlops(attnEst.chunkedAttnFlopsTFlops)}</strong>
                    </span>
                  </div>
                </div>
              </div>

              {/* Requirement 4: Fixed explanatory wording */}
              <p className="text-sm text-slate-300 leading-relaxed pt-1">
                Quadratic growth makes long clips expensive in time and, without memory-efficient attention, in memory.
              </p>
            </div>

            {/* Requirement 1: "How this is calculated" compact accordion */}
            <div className="pt-2 border-t border-slate-800/80">
              <button
                onClick={() => setShowFormula((prev) => !prev)}
                className="min-h-[44px] w-full flex items-center justify-between text-xs font-medium text-slate-400 hover:text-cyan-300 transition-colors cursor-pointer py-1"
                aria-expanded={showFormula}
              >
                <div className="flex items-center gap-1.5">
                  <HelpCircle className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
                  <span>How this is calculated (Python Estimator formulas)</span>
                </div>
                <ChevronDown
                  className={`w-3.5 h-3.5 transition-transform duration-200 ${
                    showFormula ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {showFormula && (
                <div className="mt-2.5 p-3 rounded-xl bg-slate-950 border border-slate-800/90 text-[11px] font-mono text-slate-300 space-y-2 animate-in fade-in duration-150">
                  <div className="text-slate-400 font-sans">
                    Computed dynamically from <code className="text-cyan-300">vidgen.attention:estimate_attention_flops_and_memory</code>:
                  </div>
                  <div className="space-y-1 bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/60">
                    <div>
                      <span className="text-slate-400">Total Tokens:</span>{' '}
                      <span className="text-white">N = {attnEst.totalSeqLen.toLocaleString()}</span>{' '}
                      <span className="text-slate-500">
                        ({attnEst.latentFrames}t × {attnEst.spatialTokensPerFrame}s + {attnEst.textTokens} text)
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400">Matrix Memory:</span>{' '}
                      <span className="text-cyan-300">num_heads × N² × 2 bytes (fp16)</span> ={' '}
                      <span className="text-white">{formatMem(attnEst.fullAttnMatrixGb, attnEst.fullAttnMatrixMb)}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Full Compute:</span>{' '}
                      <span className="text-cyan-300">4 × N² × dim × depth</span> ={' '}
                      <span className="text-white">{formatFlops(attnEst.fullAttnFlopsTFlops)}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Decoupled Compute:</span>{' '}
                      <span className="text-sky-300">[T×4(S+L)²d + S×4T²d] × depth</span> ={' '}
                      <span className="text-white">{formatFlops(attnEst.decoupledAttnFlopsTFlops)}</span>
                    </div>
                    <div>
                      <span className="text-slate-400">Chunked Compute:</span>{' '}
                      <span className="text-emerald-300">{attnEst.numChunks} chunks × 4 × (21·S + L)²d × depth</span> ={' '}
                      <span className="text-white">{formatFlops(attnEst.chunkedAttnFlopsTFlops)}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Card 2: Vidgen Chunked KV Cache (Constant bounded memory) */}
          <div className="p-4 sm:p-5 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between space-y-3.5">
            <div className="space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
                    Vidgen Chunked KV Cache
                  </span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/80">
                    O(1) Memory
                  </span>
                </div>
                <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/80 border border-emerald-800/80 px-2.5 py-1 rounded-full flex items-center gap-1.5 shadow-sm">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                  <span>Bounded indefinitely</span>
                </span>
              </div>

              {/* Two lines for chunked system */}
              <div className="space-y-2 pt-1 font-mono text-xs">
                <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 flex flex-wrap items-baseline justify-between gap-1">
                  <span className="text-slate-400 font-sans text-xs">Peak Activation &amp; KV Memory:</span>
                  <strong className="text-sm font-bold text-emerald-300">
                    {specs.chunkedKvVramGb.toFixed(1)} GB (Constant)
                  </strong>
                </div>

                <div className="p-2.5 rounded-lg bg-slate-950/80 border border-slate-800/80 space-y-1">
                  <span className="text-slate-400 font-sans text-xs block">Chunk Architecture Efficiency:</span>
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                    <span className="text-slate-300">
                      Chunks: <strong className="text-cyan-300">{attnEst.numChunks} (5s each)</strong>
                    </span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-300">
                      Tokens/chunk: <strong className="text-white">{attnEst.chunkSeqLen.toLocaleString()}</strong>
                    </span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-300">
                      Savings: <strong className="text-emerald-300">{specs.attentionComplexityReduction > 1 ? `${specs.attentionComplexityReduction.toFixed(1)}x speedup` : '1.0x'}</strong>
                    </span>
                  </div>
                </div>
              </div>

              <p className="text-sm text-slate-300 leading-relaxed pt-1">
                Rolling sliding window with anchor frames keeps memory strictly bounded regardless of clip duration.
              </p>
            </div>

            <div className="pt-2 border-t border-slate-800/80">
              <div className="min-h-[44px] flex items-center justify-between text-xs text-slate-400 py-1 font-mono">
                <span>Context: 1 Window Chunk + 2 Anchors</span>
                <span className="text-emerald-400 font-semibold">1-Latent Cosine Blend</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Accordions for Deep Technical Details (Interactive Memory Tracker & Overlap Visualizer) */}
      <div className="space-y-3 pt-2">
        {/* Accordion 1: Interactive Memory Tracker */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('memory_tool')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors cursor-pointer"
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
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors cursor-pointer"
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
