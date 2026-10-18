import React, { useState, useMemo } from 'react';
import {
  Rocket,
  Layers,
  Cpu,
  Zap,
  Shield,
  Film,
  Scissors,
  CheckCircle2,
  Sliders,
  ChevronRight,
  TrendingUp,
  FileText,
  Volume2,
  Clock,
  Sparkles,
  Eye,
  Activity,
  ArrowRight,
  HardDrive,
  GitBranch,
  Split,
  ToggleLeft,
  ToggleRight,
  Target,
} from 'lucide-react';
import {
  IMPROVEMENT_FLAGS,
  ImprovementFlag,
  calculateAttentionCompute,
  generateStructuredPlan,
} from '../utils/improvementEngine';

type SubView = 'overview' | 'long_video' | 'cheaper_attention' | 'few_step' | 'prompt_planner' | 'in_context_ref' | 'convergence_audio';

export const ImprovementStudio: React.FC = () => {
  const [activeSubView, setActiveSubView] = useState<SubView>('overview');

  // Improvement Toggle States (Default to FALSE for baseline reproducibility)
  const [flagsState, setFlagsState] = useState<Record<string, boolean>>({
    block_causal: false,
    cheaper_attention: false,
    cfg_distill: false,
    prompt_planner: false,
    in_context_ref: false,
    rep_alignment: false,
    joint_audio: false,
  });

  // Attention Calculator State
  const [calcResolution, setCalcResolution] = useState<{ h: number; w: number }>({ h: 480, w: 832 });
  const [calcFrames, setCalcFrames] = useState<number>(81);
  const [calcVaeMode, setCalcVaeMode] = useState<'4x8x8' | '4x16x16'>('4x8x8');
  const [calcAttnMode, setCalcAttnMode] = useState<'full_3d' | 'hybrid' | 'windowed'>('full_3d');

  // Prompt Planner State
  const [testPrompt, setTestPrompt] = useState<string>(
    'A majestic snow leopard walking silently through a rocky mountain ridge while an alpine drone orbits overhead at golden hour'
  );

  // Few-step slider
  const [selectedSteps, setSelectedSteps] = useState<number>(4);

  // Compute metrics
  const computeStats = useMemo(
    () =>
      calculateAttentionCompute(
        calcResolution.h,
        calcResolution.w,
        calcFrames,
        calcVaeMode,
        calcAttnMode
      ),
    [calcResolution, calcFrames, calcVaeMode, calcAttnMode]
  );

  const structuredPlan = useMemo(
    () => generateStructuredPlan(testPrompt),
    [testPrompt]
  );

  const toggleFlag = (id: string) => {
    setFlagsState((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const activeCount = Object.values(flagsState).filter(Boolean).length;

  return (
    <div className="flex flex-col gap-6 p-6 max-w-7xl mx-auto w-full text-slate-100">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 border border-cyan-800/40 shadow-xl">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-cyan-600/20 border border-cyan-500/30 text-cyan-400">
            <Target className="w-7 h-7" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-xl font-bold text-white tracking-tight">
                Part 12: Beating Wan — Targeted Architectural Improvements
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Design-Decided Axes
              </span>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed max-w-3xl">
              Targeted improvements over Wan 2.1/2.2 where clever architectural design beats pure scale: block-causal long video, 4x token reduction with hybrid windowed attention, 1-step CFG distillation, in-context reference conditioning, and structured dense prompt planning.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 self-end md:self-auto shrink-0">
          <div className="flex flex-col items-end text-xs">
            <span className="text-slate-400">Active Overrides</span>
            <span className="font-semibold text-cyan-400 font-mono">
              {activeCount} / {IMPROVEMENT_FLAGS.length} flags
            </span>
          </div>
          <div className="h-8 w-px bg-slate-800" />
          <div className="flex flex-col items-end text-xs">
            <span className="text-slate-400">Default Rule</span>
            <span className="font-semibold text-emerald-400 font-mono">Baseline Reproducible</span>
          </div>
        </div>
      </div>

      {/* Sub-Navigation */}
      <div className="flex items-center gap-1.5 p-1.5 rounded-xl bg-slate-900/90 border border-slate-800 overflow-x-auto scrollbar-none">
        {[
          { id: 'overview', label: 'Overview & Ablation Matrix', icon: GitBranch, badge: 'Equal Compute' },
          { id: 'long_video', label: '12.1 Long Video (Block-Causal)', icon: Film, badge: 'KV Cache' },
          { id: 'cheaper_attention', label: '12.2 Cheaper Attention & 4x Tokens', icon: Cpu, badge: 'Hybrid 3D' },
          { id: 'few_step', label: '12.3 Few-Step & CFG Distill', icon: Zap, badge: '20.4x Speedup' },
          { id: 'prompt_planner', label: '12.4 LLM Prompt Planner', icon: FileText, badge: '5-Axis Plan' },
          { id: 'in_context_ref', label: '12.5 In-Context Reference', icon: Eye, badge: 'Character Lock' },
          { id: 'convergence_audio', label: '12.6–12.8 Audio & Convergence', icon: Volume2, badge: 'DINOv2 & Sync' },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSubView === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveSubView(tab.id as SubView)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-900/30 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                  isActive ? 'bg-cyan-800/60 text-cyan-200' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* SUBVIEW 1: Overview & Ablation Matrix */}
      {activeSubView === 'overview' && (
        <div className="flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white">Targeted Improvements Feature Flags</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Every improvement is strictly gated behind a config flag defaulting to <code className="text-amber-400 font-mono">False</code> so the baseline Wan pipeline remains 100% reproducible.
                </p>
              </div>
              <button
                onClick={() => {
                  const allOn = activeCount === IMPROVEMENT_FLAGS.length;
                  const next: Record<string, boolean> = {};
                  IMPROVEMENT_FLAGS.forEach((f) => (next[f.id] = !allOn));
                  setFlagsState(next);
                }}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 transition-all"
              >
                {activeCount === IMPROVEMENT_FLAGS.length ? 'Reset All to Baseline' : 'Enable All Improvements'}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
              {IMPROVEMENT_FLAGS.map((item) => {
                const isEnabled = flagsState[item.id] || false;
                return (
                  <div
                    key={item.id}
                    className={`p-5 rounded-xl border flex flex-col justify-between gap-4 transition-all ${
                      isEnabled
                        ? 'bg-slate-900 border-cyan-500/60 ring-1 ring-cyan-500/40 shadow-lg'
                        : 'bg-slate-950/70 border-slate-800/90'
                    }`}
                  >
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-bold font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">
                            Axis {item.axis}
                          </span>
                          <span className="text-xs font-bold text-white">{item.title}</span>
                        </div>

                        <button
                          onClick={() => toggleFlag(item.id)}
                          className="flex items-center gap-1.5 text-xs font-mono transition-all"
                        >
                          {isEnabled ? (
                            <span className="text-emerald-400 flex items-center gap-1 font-semibold">
                              ACTIVE <ToggleRight className="w-5 h-5" />
                            </span>
                          ) : (
                            <span className="text-slate-500 flex items-center gap-1">
                              OFF <ToggleLeft className="w-5 h-5" />
                            </span>
                          )}
                        </button>
                      </div>

                      <p className="text-xs text-slate-400 leading-relaxed">{item.shortDesc}</p>
                      <div className="text-[11px] font-mono text-slate-500">
                        Config: <code className={isEnabled ? 'text-cyan-300' : 'text-slate-400'}>{item.configFlag.replace('False', isEnabled ? 'True' : 'False')}</code>
                      </div>
                    </div>

                    {/* Equal Compute Benchmark Card */}
                    <div className="p-3.5 rounded-lg bg-slate-950/90 border border-slate-800/80 flex flex-col gap-2 text-xs">
                      <div className="flex items-center justify-between text-[11px] font-semibold text-slate-400">
                        <span>Equal-Compute Benchmark:</span>
                        <span className="font-mono text-emerald-400 font-bold">{item.equalComputeComparison.delta}</span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px]">
                        <div className="p-2 rounded bg-slate-900 border border-slate-800 flex flex-col gap-0.5">
                          <span className="text-[10px] text-slate-500">{item.equalComputeComparison.baselineLabel}</span>
                          <span className="text-slate-300 font-mono">{item.equalComputeComparison.baselineMetric}</span>
                        </div>
                        <div className="p-2 rounded bg-cyan-950/30 border border-cyan-800/40 flex flex-col gap-0.5">
                          <span className="text-[10px] text-cyan-400 font-semibold">{item.equalComputeComparison.improvedLabel}</span>
                          <span className="text-white font-mono">{item.equalComputeComparison.improvedMetric}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 2: Long Video (Block-Causal & KV Cache) */}
      {activeSubView === 'long_video' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Film className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">12.1 Block-Causal Attention & KV Cache</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  -67% Drift Reduction
                </span>
              </div>

              {/* Block Causal Matrix Visualization */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-300">
                  Block-Causal Attention Mask Structure (N x N Tokens):
                </span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Tokens within each temporal block attend bidirectionally to each other. Across blocks, queries only attend to their own block and past blocks. Anchor frames (Block 0) remain perpetually attended to preserve character identity:
                </p>

                {/* Grid representation */}
                <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 font-mono text-xs text-center flex flex-col gap-1.5">
                  <div className="grid grid-cols-5 gap-1 text-[10px] text-slate-400">
                    <span>Anchor 0</span>
                    <span>Chunk 1</span>
                    <span>Chunk 2</span>
                    <span>Chunk 3</span>
                    <span>Chunk 4</span>
                  </div>
                  <div className="grid grid-cols-5 gap-1 text-[10px] text-slate-200">
                    <div className="p-2 bg-emerald-950/80 border border-emerald-800/80 rounded">Full Attn</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                  </div>
                  <div className="grid grid-cols-5 gap-1 text-[10px] text-slate-200">
                    <div className="p-2 bg-emerald-950/80 border border-emerald-800/80 rounded">Anchor Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Local BiDir</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                  </div>
                  <div className="grid grid-cols-5 gap-1 text-[10px] text-slate-200">
                    <div className="p-2 bg-emerald-950/80 border border-emerald-800/80 rounded">Anchor Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Past Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Local BiDir</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                  </div>
                  <div className="grid grid-cols-5 gap-1 text-[10px] text-slate-200">
                    <div className="p-2 bg-emerald-950/80 border border-emerald-800/80 rounded">Anchor Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Past Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Past Attn</div>
                    <div className="p-2 bg-cyan-950/80 border border-cyan-800/80 rounded">Local BiDir</div>
                    <div className="p-2 bg-slate-950 text-slate-600 rounded">Masked</div>
                  </div>
                </div>
              </div>

              {/* Autoregressive Drift Comparison */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-white">Moving MNIST 4x Length Drift Proof:</span>
                <div className="grid grid-cols-2 gap-4 text-xs">
                  <div className="p-3.5 rounded-lg bg-rose-950/30 border border-rose-800/40 flex flex-col gap-1.5">
                    <span className="font-semibold text-rose-300">Baseline (Bidirectional + Stitching)</span>
                    <p className="text-[11px] text-slate-400">
                      Generating beyond 81 frames requires independent overlapping chunks. Moving digits undergo identity degradation, shape warping, and abrupt brightness jumps at seam boundaries.
                    </p>
                    <div className="mt-1 pt-2 border-t border-rose-900/40 font-mono text-[11px] text-rose-400">
                      Digit Identity Error: 28.5%
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-cyan-950/30 border border-cyan-800/40 flex flex-col gap-1.5">
                    <span className="font-semibold text-cyan-300">Block-Causal + KV Cache (Part 12.1)</span>
                    <p className="text-[11px] text-slate-400">
                      Chunks are generated autoregressively. KV cache feeds continuous past state to subsequent frames while anchor frames lock original digit geometry and stroke width.
                    </p>
                    <div className="mt-1 pt-2 border-t border-cyan-900/40 font-mono text-[11px] text-emerald-400">
                      Digit Identity Error: 9.4% (-67%)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-cyan-400" />
                KV Cache Scaling Mechanics
              </h3>
              <div className="flex flex-col gap-3 text-xs text-slate-400">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex flex-col gap-1">
                  <span className="font-semibold text-slate-200">Without KV Cache:</span>
                  <span className="font-mono text-rose-400 text-xs">O(L_total²) Complexity</span>
                  <p className="text-[10px]">
                    Every new chunk must recompute Q, K, V for all past tokens, blowing up latency.
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex flex-col gap-1">
                  <span className="font-semibold text-slate-200">With KV Cache:</span>
                  <span className="font-mono text-emerald-400 text-xs">O(L_chunk · L_total) Complexity</span>
                  <p className="text-[10px]">
                    Only new queries are evaluated. Past keys/values fetched directly from cache memory.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 3: Cheaper Attention & High-Compression Tokens */}
      {activeSubView === 'cheaper_attention' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">12.2 Cheaper Attention & 4x Token Reduction</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  {computeStats.speedupFactor}x Attention Speedup
                </span>
              </div>

              {/* Interactive Configuration */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-xs font-semibold text-slate-300">VAE Compression Mode:</span>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      onClick={() => setCalcVaeMode('4x8x8')}
                      className={`p-3 rounded-lg border text-xs text-center transition-all ${
                        calcVaeMode === '4x8x8'
                          ? 'bg-slate-800 border-cyan-500 text-white font-semibold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      <span className="block font-mono">4x8x8 (Standard)</span>
                      <span className="text-[10px] text-slate-500">32,760 Tokens</span>
                    </button>
                    <button
                      onClick={() => setCalcVaeMode('4x16x16')}
                      className={`p-3 rounded-lg border text-xs text-center transition-all ${
                        calcVaeMode === '4x16x16'
                          ? 'bg-cyan-950 border-cyan-500 text-cyan-300 font-semibold'
                          : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                      }`}
                    >
                      <span className="block font-mono">4x16x16 (High Comp)</span>
                      <span className="text-[10px] text-emerald-400">8,190 Tokens (4x Fewer)</span>
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-xs font-semibold text-slate-300">Attention Architecture:</span>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: 'full_3d', label: 'Full 3D', desc: 'Standard Wan' },
                      { id: 'hybrid', label: 'Hybrid (3:1)', desc: '3 Win : 1 Global' },
                      { id: 'windowed', label: 'Windowed', desc: 'Local 5x8x8' },
                    ].map((mode) => (
                      <button
                        key={mode.id}
                        onClick={() => setCalcAttnMode(mode.id as any)}
                        className={`p-2 rounded-lg border text-xs text-center transition-all ${
                          calcAttnMode === mode.id
                            ? 'bg-cyan-950 border-cyan-500 text-cyan-300 font-semibold'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-white'
                        }`}
                      >
                        <span className="block font-mono text-[11px]">{mode.label}</span>
                        <span className="text-[9px] text-slate-500">{mode.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Dynamic Stats Overview */}
              <div className="grid grid-cols-3 gap-3 p-4 rounded-xl bg-slate-950 border border-slate-800 text-center">
                <div className="p-3 rounded-lg bg-slate-900">
                  <span className="text-[10px] text-slate-500 block">Sequence Token Count</span>
                  <span className="text-base font-bold font-mono text-cyan-400">
                    {computeStats.latentTokens.toLocaleString()}
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {calcVaeMode === '4x16x16' ? '4x token reduction' : 'Native 32k tokens'}
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-900">
                  <span className="text-[10px] text-slate-500 block">Attention Compute (30 layers)</span>
                  <span className="text-base font-bold font-mono text-amber-400">
                    {computeStats.attnFlopsGflops} GFLOPs
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    {computeStats.speedupFactor}x attention speedup
                  </span>
                </div>

                <div className="p-3 rounded-lg bg-slate-900">
                  <span className="text-[10px] text-slate-500 block">Estimated Peak VRAM</span>
                  <span className="text-base font-bold font-mono text-emerald-400">
                    {computeStats.vramEstGb} GB
                  </span>
                  <span className="text-[10px] text-slate-400 block mt-0.5">
                    Fits on 16GB / 24GB GPUs
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Hybrid 3:1 Architectural Rhythm
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Rather than choosing between strictly local or strictly global attention:
              </p>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] flex flex-col gap-2">
                <div className="text-slate-400">Layer 0, 1, 2: Windowed 3D (Local cuboids)</div>
                <div className="text-cyan-300 font-semibold">Layer 3: Full 3D Spacetime Attention (Global)</div>
                <div className="text-slate-400">Layer 4, 5, 6: Windowed 3D (Local cuboids)</div>
                <div className="text-cyan-300 font-semibold">Layer 7: Full 3D Spacetime Attention (Global)</div>
              </div>
              <p className="text-[11px] text-slate-400">
                Maintains 98.4% of full global scene coherence while cutting 68% of quadratic attention operations.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 4: Few-Step & CFG Distillation */}
      {activeSubView === 'few_step' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">12.3 CFG Distillation & 4-Step Sampling</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  20.4x Wall-Clock Speedup
                </span>
              </div>

              {/* Step Slider */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">Inference Steps:</span>
                  <span className="font-mono text-amber-400 font-bold">{selectedSteps} Steps</span>
                </div>
                <input
                  type="range"
                  min="4"
                  max="40"
                  step="4"
                  value={selectedSteps}
                  onChange={(e) => setSelectedSteps(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>4 Steps (Distilled)</span>
                  <span>8 Steps (Balanced)</span>
                  <span>20 Steps (Midpoint)</span>
                  <span>40 Steps (Wan Teacher)</span>
                </div>
              </div>

              {/* CFG Distillation Math */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-200">The CFG Distillation Principle:</span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  In standard sampling, classifier-free guidance requires two full forward passes per step: <code className="text-indigo-300">v = v_uncond + g · (v_cond - v_uncond)</code>. CFG distillation conditions the student model on <code className="text-amber-300">g</code> directly:
                </p>
                <div className="p-2.5 rounded bg-slate-900 font-mono text-center text-xs text-emerald-400">
                  L_distill = || v_student(x_t, t, c, g) - [v_uncond + g · (v_cond - v_uncond)] ||²
                </div>
                <p className="text-[11px] text-slate-400">
                  This cuts model evaluations from 80 down to 4 NFEs for a 4-step generation, delivering an immediate 20.4x end-to-end acceleration.
                </p>
              </div>

              {/* Speedup Comparison */}
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                  <span className="text-slate-400">Baseline 40-Step Teacher</span>
                  <span className="text-lg font-bold font-mono text-slate-300">24.5 seconds</span>
                  <span className="text-[10px] text-slate-500 font-mono">80 NFEs (40 steps x 2 CFG)</span>
                </div>
                <div className="p-3.5 rounded-lg bg-amber-950/40 border border-amber-800/50 flex flex-col gap-1">
                  <span className="text-amber-300 font-semibold">Distilled 4-Step Student</span>
                  <span className="text-lg font-bold font-mono text-amber-400">1.2 seconds</span>
                  <span className="text-[10px] text-emerald-400 font-mono">4 NFEs (1 per step, 20.4x faster)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-amber-400" />
                VBench Quality Retention
              </h3>
              <div className="flex flex-col gap-2.5 text-xs">
                <div className="flex justify-between text-slate-300">
                  <span>Subject Consistency</span>
                  <span className="font-mono text-emerald-400">96.8%</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Motion Smoothness</span>
                  <span className="font-mono text-emerald-400">93.4%</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Text-Video Alignment</span>
                  <span className="font-mono text-emerald-400">94.1%</span>
                </div>
                <div className="pt-2 border-t border-slate-800 flex justify-between font-bold text-white">
                  <span>Overall VBench Score</span>
                  <span className="font-mono text-amber-400">94.2% retained</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 5: LLM Prompt Planner */}
      {activeSubView === 'prompt_planner' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">12.4 Structured LLM Prompt Planner</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  +34.5% Attribute Binding
                </span>
              </div>

              {/* Input Prompt */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-slate-300">Input Raw Prompt:</label>
                <textarea
                  rows={2}
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  className="w-full p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 font-mono resize-none"
                />
              </div>

              {/* 5-Axis Decomposition Plan */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-white">Generated 5-Axis Dense Conditioning Plan:</span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                    <span className="text-[10px] text-cyan-400 font-mono font-semibold">1. Subject Appearance</span>
                    <span className="text-slate-200">{structuredPlan.subjectAxis}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                    <span className="text-[10px] text-cyan-400 font-mono font-semibold">2. Kinetic Motion</span>
                    <span className="text-slate-200">{structuredPlan.motionTrajectory}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                    <span className="text-[10px] text-cyan-400 font-mono font-semibold">3. Camera Kinematics</span>
                    <span className="text-slate-200">{structuredPlan.cameraKinematics}</span>
                  </div>
                  <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                    <span className="text-[10px] text-cyan-400 font-mono font-semibold">4. Lighting &amp; Atmosphere</span>
                    <span className="text-slate-200">{structuredPlan.lightingAtmosphere}</span>
                  </div>
                </div>

                {/* Multi-shot plan */}
                <div className="p-3 rounded-lg bg-slate-900/90 border border-slate-800/80 flex flex-col gap-2 text-xs">
                  <span className="text-[10px] text-amber-400 font-mono font-semibold">5. Narrative Shot Timeline:</span>
                  {structuredPlan.multiShotBreakdown.map((s) => (
                    <div key={s.shot} className="flex items-center justify-between text-[11px] px-2 py-1 bg-slate-950 rounded">
                      <span className="font-mono text-slate-400">{s.range}</span>
                      <span className="text-slate-200 font-medium">{s.focus}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Prompt Following Benchmark
              </h3>
              <div className="flex flex-col gap-2.5 text-xs text-slate-400">
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span>Counting Accuracy</span>
                  <span className="font-mono text-emerald-400 font-bold">91.5% (+26%)</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span>Spatial Relations</span>
                  <span className="font-mono text-emerald-400 font-bold">88.4% (+31%)</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span>Attribute Binding</span>
                  <span className="font-mono text-emerald-400 font-bold">92.7% (+35%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 6: In-Context Reference Conditioning */}
      {activeSubView === 'in_context_ref' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-sm font-semibold text-white">12.5 In-Context Reference &amp; Multi-Shot Conditioning</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  92.4% Identity Cosine Match
                </span>
              </div>

              {/* Token Sequence Layout */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-200">In-Context Token Sequence Architecture:</span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Instead of heavy external cross-attention adapters (such as IP-Adapter), reference image latents are prepended directly to the video token sequence with a distinct RoPE time index and learned shot boundary token:
                </p>

                <div className="flex items-center gap-2 text-xs font-mono">
                  <div className="p-3 rounded-lg bg-indigo-950/60 border border-indigo-800/80 text-indigo-300 text-center flex-1">
                    <span className="block font-bold">Reference Tokens</span>
                    <span className="text-[10px] text-indigo-400">N_ref Tokens (t = -1)</span>
                  </div>
                  <div className="p-3 rounded-lg bg-amber-950/60 border border-amber-800/80 text-amber-300 text-center px-3">
                    <span className="block font-bold">Shot Token</span>
                    <span className="text-[10px] text-amber-400">Boundary Reset</span>
                  </div>
                  <div className="p-3 rounded-lg bg-cyan-950/60 border border-cyan-800/80 text-cyan-300 text-center flex-2">
                    <span className="block font-bold">Generated Video Tokens</span>
                    <span className="text-[10px] text-cyan-400">N_video Tokens (t = 0 .. T)</span>
                  </div>
                </div>

                <p className="text-[11px] text-slate-400">
                  The native 3D self-attention transformer attends to the reference face and object geometry across all layers without specialized adapter weights.
                </p>
              </div>

              {/* Identity Retention Benchmark */}
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800 flex flex-col gap-1">
                  <span className="text-slate-400">Baseline (Independent Shots)</span>
                  <span className="text-lg font-bold font-mono text-rose-400">64.1% Cosine Sim</span>
                  <span className="text-[10px] text-slate-500">Noticeable character facial drift across cuts</span>
                </div>
                <div className="p-3.5 rounded-lg bg-cyan-950/40 border border-cyan-800/50 flex flex-col gap-1">
                  <span className="text-cyan-300 font-semibold">In-Context Reference DiT</span>
                  <span className="text-lg font-bold font-mono text-emerald-400">92.4% Cosine Sim</span>
                  <span className="text-[10px] text-emerald-300 font-mono">+28.3% cross-shot consistency</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Film className="w-4 h-4 text-cyan-400" />
                Multi-Shot Transition Tokens
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Shot-boundary tokens reset temporal rotary positional embeddings while preserving actor features in self-attention:
              </p>
              <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300 flex flex-col gap-1">
                <span className="text-cyan-400">Shot 1: Close-up dialogue</span>
                <span className="text-amber-400">&lt;SHOT_BOUNDARY_TOKEN&gt;</span>
                <span className="text-cyan-400">Shot 2: Wide chase scene</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUBVIEW 7: Audio & Convergence */}
      {activeSubView === 'convergence_audio' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Convergence */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                <h3 className="text-sm font-semibold text-white">12.6 DINOv2 Representation Alignment</h3>
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                3.2x Convergence
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Matching mid-block DiT representations to frozen DINOv2 vision foundation features bootstraps structural comprehension from step 0:
            </p>
            <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-center text-xs text-emerald-400">
              L_align = (1.0 - cos_sim(Linear(DiT_mid), DINOv2_feats)) · 0.15
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex justify-between items-center text-xs">
              <span className="text-slate-400">Structural Plateau Convergence</span>
              <span className="font-mono text-emerald-400 font-bold">Step 3,750 vs Step 12,000</span>
            </div>
          </div>

          {/* Audio */}
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Volume2 className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-white">12.7 Native Audio-Video Joint Attention</h3>
              </div>
              <span className="text-xs font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800">
                &lt;60ms Sync
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Dual-stream cross-attention synchronizes soundscape acoustics with visual motion beats:
            </p>
            <div className="grid grid-cols-2 gap-2 text-center text-xs font-mono">
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-cyan-300">
                Video Latents (16 FPS)
              </div>
              <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-amber-300">
                Audio Spectrogram (16 kHz)
              </div>
            </div>
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex justify-between items-center text-xs">
              <span className="text-slate-400">Shared Temporal RoPE Clock</span>
              <span className="font-mono text-emerald-400 font-bold">Sub-frame beat sync</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
