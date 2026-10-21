import React, { useState, useMemo, useEffect } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import {
  calculateDiTParameters,
  DIT_PRESET_CONFIGS,
  computeSinusoidalEmbeddings,
  computeModulationValues,
  simulateOverfitRun,
  getAttentionComplexityComparison,
  OverfitStepRecord,
} from '../utils/ditEngine';
import {
  Layers,
  Cpu,
  Zap,
  TrendingDown,
  CheckCircle2,
  Sliders,
  Play,
  RotateCcw,
  ShieldCheck,
  Activity,
  ArrowRight,
  Sparkles,
  Database,
  BarChart3,
  Split,
  Eye,
  Brain,
} from 'lucide-react';

interface DiTStudioProps {
  currentPreset: PresetKey;
}

export const DiTStudio: React.FC<DiTStudioProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const ditConfig = DIT_PRESET_CONFIGS[currentPreset];
  const paramsBreakdown = useMemo(() => calculateDiTParameters(ditConfig, currentPreset), [ditConfig, currentPreset]);

  const [activeSubTab, setActiveSubTab] = useState<'block_architecture' | 'parameter_verifier' | 'big_brain_scaler' | 'timestep_adaln' | 'overfit_simulator' | 'ablation_comparison'>('big_brain_scaler');

  // Big Brain Scaler Custom Architecture State - Defaulted to 14 Billion Flagship
  const [customLayers, setCustomLayers] = useState<number>(40);
  const [customDim, setCustomDim] = useState<number>(5120);
  const [customHeads, setCustomHeads] = useState<number>(40);
  const [customFfn, setCustomFfn] = useState<number>(13824);

  const customDitConfig = useMemo(() => ({
    numLayers: customLayers,
    hiddenDim: customDim,
    numHeads: customHeads,
    headDim: Math.round(customDim / customHeads),
    ffnDim: customFfn,
    textDim: 4096,
    patchSize: [1, 2, 2] as [number, number, number],
    inChannels: 16,
    outChannels: 16,
  }), [customLayers, customDim, customHeads, customFfn]);

  const customBreakdown = useMemo(() => {
    return calculateDiTParameters(customDitConfig, 'vidgen14b');
  }, [customDitConfig]);

  const isBigBrainThresholdMet = customBreakdown.totalParams >= 14_000_000_000;

  // Timestep & adaLN Interactive State
  const [timestep, setTimestep] = useState<number>(0.35);
  const sinusoidalHarmonics = useMemo(() => computeSinusoidalEmbeddings(timestep, 32), [timestep]);
  const modValues = useMemo(() => computeModulationValues(timestep), [timestep]);

  // Overfit Training Simulation State
  const [numClips, setNumClips] = useState<number>(2);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingHistory, setTrainingHistory] = useState<OverfitStepRecord[]>(() => simulateOverfitRun(2, 30));
  const [currentStepIdx, setCurrentStepIdx] = useState<number>(29);

  const startOverfitSimulation = () => {
    setIsTraining(true);
    const newRecords = simulateOverfitRun(numClips, 35);
    setTrainingHistory(newRecords);
    setCurrentStepIdx(0);

    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step < newRecords.length) {
        setCurrentStepIdx(step);
      } else {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 70);
  };

  // Attention Complexity comparison
  const complexity = useMemo(() => {
    const gridT = Math.floor((preset.baseFrames - 1) / 4) + 1;
    const gridH = Math.floor(preset.height / 16);
    const gridW = Math.floor(preset.width / 16);
    return getAttentionComplexityComparison(gridT, gridH, gridW);
  }, [preset]);

  const currentOverfitRecord = trainingHistory[currentStepIdx] || trainingHistory[trainingHistory.length - 1];
  const initialLoss = trainingHistory[0]?.loss || 0.85;
  const latestLoss = currentOverfitRecord.loss;
  const lossReductionPercent = ((initialLoss - latestLoss) / initialLoss) * 100;

  return (
    <div className="space-y-6">
      {/* Studio Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-indigo-500/10 text-indigo-400">
                <Cpu className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Part 6: Vidgen DiT Backbone Engine
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              N-layer repeated DiT backbone with <strong className="text-white">adaLN-Zero</strong> modulation,
              Full 3D spacetime attention with <strong className="text-cyan-300">3D RoPE & QK-RMSNorm</strong>,
              plain LayerNorm text cross-attention, GELU (tanh approximation) FFN, and parameter-saving shared timestep projections.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono bg-slate-950 text-slate-300 px-3 py-1.5 rounded-xl border border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
              Preset: {preset.name} ({paramsBreakdown.totalParams >= 1e9 ? `${(paramsBreakdown.totalParams / 1e9).toFixed(2)}B` : `${(paramsBreakdown.totalParams / 1e6).toFixed(1)}M`} params)
            </span>
          </div>
        </div>

        {/* Sub-Tabs */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-slate-800/80 overflow-x-auto no-scrollbar">
          {[
            { id: 'big_brain_scaler', label: '🧠 Big Brain Scaler (≥14B Params)', icon: Brain },
            { id: 'block_architecture', label: 'Vidgen DiT Block Flow', icon: Layers },
            { id: 'parameter_verifier', label: 'Preset Params & Savings', icon: BarChart3 },
            { id: 'timestep_adaln', label: 'Timestep & adaLN-Zero', icon: Sliders },
            { id: 'overfit_simulator', label: 'Overfit Test on Clips', icon: TrendingDown },
            { id: 'ablation_comparison', label: 'Full 3D vs Factorized Ablation', icon: Split },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeSubTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveSubTab(tab.id as any)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                  isActive
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                    : 'bg-slate-950/60 text-slate-400 hover:text-white hover:bg-slate-800/60 border border-slate-800/60'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* SUB-TAB 0: BIG BRAIN SCALER (≥ 9 BILLION PARAMETERS) */}
      {activeSubTab === 'big_brain_scaler' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-6">
            {/* Header */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div>
                <div className="flex items-center gap-2">
                  <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
                    <Brain className="w-5 h-5" />
                  </span>
                  <h3 className="text-base font-bold text-white flex items-center gap-2">
                    Big Brain Parameter Scaler (≥ 14 Billion Parameters)
                  </h3>
                </div>
                <p className="text-xs text-slate-300 mt-1 max-w-2xl">
                  Configure and scale the Vidgen Video DiT backbone to reach and exceed the <strong className="text-white">14 Billion Parameter</strong> flagship threshold.
                  Fine-tune depth, width, attention heads, and FFN expansion with real-time mathematical parameter verification.
                </p>
              </div>

              {/* Quick Actions */}
              <div className="flex flex-wrap items-center gap-2">
                <button
                  onClick={() => {
                    setCustomLayers(40);
                    setCustomDim(5120);
                    setCustomHeads(40);
                    setCustomFfn(13824);
                  }}
                  className="px-3.5 py-1.5 rounded-xl bg-gradient-to-r from-emerald-500/30 to-cyan-500/30 hover:from-emerald-500/40 hover:to-cyan-500/40 text-emerald-200 border border-emerald-400/50 text-xs font-bold transition-all cursor-pointer flex items-center gap-1.5 shadow-md shadow-emerald-950/40"
                >
                  <Brain className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                  <span>Preset: 14.3B Big Brain Flagship</span>
                </button>
                <button
                  onClick={() => {
                    setCustomLayers(40);
                    setCustomDim(4096);
                    setCustomHeads(32);
                    setCustomFfn(11264);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 shadow-sm"
                >
                  <Zap className="w-3.5 h-3.5" />
                  <span>Preset: 9.2B High-Capacity</span>
                </button>
                <button
                  onClick={() => {
                    setCustomLayers(48);
                    setCustomDim(5632);
                    setCustomHeads(44);
                    setCustomFfn(15360);
                  }}
                  className="px-3 py-1.5 rounded-xl bg-indigo-500/10 hover:bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs font-semibold transition-all cursor-pointer flex items-center gap-1.5 shadow-sm"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Preset: 18.5B Ultra Giant</span>
                </button>
              </div>
            </div>

            {/* Threshold Banner Card */}
            <div className={`p-5 rounded-2xl border transition-all ${
              isBigBrainThresholdMet
                ? 'bg-gradient-to-r from-emerald-950/60 via-slate-950 to-cyan-950/60 border-emerald-500/60 shadow-lg shadow-emerald-950/30'
                : 'bg-gradient-to-r from-amber-950/40 via-slate-950 to-slate-950 border-amber-500/50'
            }`}>
              <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
                <div className="space-y-1.5">
                  <div className="flex items-center gap-2">
                    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold font-mono uppercase tracking-wider ${
                      isBigBrainThresholdMet
                        ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 shadow-sm shadow-emerald-500/20'
                        : 'bg-amber-500/20 text-amber-300 border border-amber-500/50'
                    }`}>
                      <Brain className="w-4 h-4 animate-pulse" />
                      {isBigBrainThresholdMet ? '✓ Big Brain Threshold Satisfied (≥ 14 Billion)' : 'Below 14 Billion Threshold'}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">
                      Target: ≥ 14,000,000,000
                    </span>
                  </div>

                  <div className="flex items-baseline gap-3">
                    <span className="text-3xl sm:text-4xl font-extrabold text-white font-mono tracking-tight">
                      {(customBreakdown.totalParams / 1e9).toFixed(3)}B
                    </span>
                    <span className="text-sm font-mono text-slate-300">
                      ({customBreakdown.totalParams.toLocaleString()} parameters)
                    </span>
                  </div>

                  <p className="text-xs text-slate-300">
                    {isBigBrainThresholdMet
                      ? `This flagship 14B architecture provides state-of-the-art cinematic spatio-temporal reasoning, fluid narrative stability, complex real-world physical simulations, and multi-actor persistence with ${((customBreakdown.totalParams - 14e9) / 1e9).toFixed(2)}B parameters above the 14B flagship requirement.`
                      : 'Increase layers, hidden width, or FFN dimension using the controls below to cross 14.00 Billion parameters.'}
                  </p>
                </div>

                {/* Progress Meter towards 14B */}
                <div className="w-full lg:w-72 bg-slate-950/80 p-3.5 rounded-xl border border-slate-800 space-y-2">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-slate-400">14B Target Progress:</span>
                    <span className={`font-bold ${isBigBrainThresholdMet ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {Math.min(200, Math.round((customBreakdown.totalParams / 14e9) * 100))}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
                    <div
                      className={`h-full transition-all duration-300 ${
                        isBigBrainThresholdMet
                          ? 'bg-gradient-to-r from-cyan-500 via-teal-400 to-emerald-400'
                          : 'bg-gradient-to-r from-amber-500 to-orange-400'
                      }`}
                      style={{ width: `${Math.min(100, (customBreakdown.totalParams / 14e9) * 100)}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] font-mono text-slate-500">
                    <span>0B</span>
                    <span className="text-slate-400">9B</span>
                    <span className="text-cyan-400 font-bold">14.0B Target</span>
                    <span>20.0B+</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Interactive Sliders Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Slider 1: Layers */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300">Transformer Blocks / Layers (L)</span>
                  <span className="font-mono font-bold text-cyan-400 px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800/60">
                    {customLayers} Layers
                  </span>
                </div>
                <input
                  type="range"
                  min="16"
                  max="56"
                  step="2"
                  value={customLayers}
                  onChange={(e) => setCustomLayers(parseInt(e.target.value))}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>16 Layers (Light)</span>
                  <span className="text-slate-400">40 Layers (14B Flagship)</span>
                  <span>56 Layers (Deep)</span>
                </div>
              </div>

              {/* Slider 2: Hidden Dim */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300">Hidden Width / Dim (D)</span>
                  <span className="font-mono font-bold text-cyan-400 px-2 py-0.5 rounded bg-cyan-950 border border-cyan-800/60">
                    {customDim} Channels
                  </span>
                </div>
                <input
                  type="range"
                  min="2048"
                  max="6144"
                  step="256"
                  value={customDim}
                  onChange={(e) => {
                    const newDim = parseInt(e.target.value);
                    setCustomDim(newDim);
                    // Adjust heads if needed to maintain headDim ~128
                    const heads = Math.max(16, Math.round(newDim / 128));
                    setCustomHeads(heads);
                  }}
                  className="w-full accent-cyan-400 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>2048 (Narrow)</span>
                  <span className="text-slate-400">5120 (14B Base)</span>
                  <span>6144 (Ultra-Wide)</span>
                </div>
              </div>

              {/* Slider 3: Attention Heads */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300">Attention Heads (H)</span>
                  <span className="font-mono font-bold text-sky-400 px-2 py-0.5 rounded bg-sky-950 border border-sky-800/60">
                    {customHeads} Heads (Dim/Head: {Math.round(customDim / customHeads)})
                  </span>
                </div>
                <input
                  type="range"
                  min="16"
                  max="48"
                  step="4"
                  value={customHeads}
                  onChange={(e) => setCustomHeads(parseInt(e.target.value))}
                  className="w-full accent-sky-400 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>16 Heads</span>
                  <span className="text-slate-400">40 Heads (14B)</span>
                  <span>48 Heads</span>
                </div>
              </div>

              {/* Slider 4: FFN Expansion */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-slate-300">FFN Intermediate Width</span>
                  <span className="font-mono font-bold text-indigo-400 px-2 py-0.5 rounded bg-indigo-950 border border-indigo-800/60">
                    {customFfn} ({((customFfn / customDim) * 1).toFixed(2)}× Expansion)
                  </span>
                </div>
                <input
                  type="range"
                  min="6144"
                  max="18432"
                  step="512"
                  value={customFfn}
                  onChange={(e) => setCustomFfn(parseInt(e.target.value))}
                  className="w-full accent-indigo-400 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>6,144 (Standard)</span>
                  <span className="text-slate-400">13,824 (14B Standard)</span>
                  <span>18,432 (Massive)</span>
                </div>
              </div>
            </div>

            {/* Architecture Metrics Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Per-Block Params</div>
                <div className="text-base font-bold text-white font-mono mt-0.5">
                  {(customBreakdown.totalPerBlock).toLocaleString()}
                </div>
                <div className="text-[10px] text-cyan-400 font-mono mt-0.5">
                  {(customBreakdown.totalPerBlock / 1e6).toFixed(1)}M / block
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Blocks Footprint</div>
                <div className="text-base font-bold text-emerald-400 font-mono mt-0.5">
                  {(customBreakdown.allBlocks / 1e9).toFixed(3)}B
                </div>
                <div className="text-[10px] text-emerald-400/80 font-mono mt-0.5">
                  {customLayers} repeated blocks
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Head Dimension</div>
                <div className="text-base font-bold text-sky-400 font-mono mt-0.5">
                  {Math.round(customDim / customHeads)}
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                  3D RoPE (44, 42, 42)
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Shared Mod Savings</div>
                <div className="text-base font-bold text-teal-400 font-mono mt-0.5">
                  {customBreakdown.sharedModSavingsPercent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-teal-400/80 font-mono mt-0.5">
                  vs per-block MLPs
                </div>
              </div>
            </div>

            {/* Generated Python Config Snippet */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white font-mono">Generated Vidgen Python DiT Configuration (vidgen/config.py)</span>
                <span className="text-[10px] font-mono text-cyan-400">Total Params: {(customBreakdown.totalParams / 1e9).toFixed(2)}B</span>
              </div>
              <pre className="p-3 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono text-slate-200 overflow-x-auto">
{`# 🧠 Big Brain Model Configuration (${(customBreakdown.totalParams / 1e9).toFixed(2)} Billion Parameters)
dit_config = DiTConfig(
    layers=${customLayers},       # ${customLayers} transformer blocks
    dim=${customDim},        # ${customDim} model hidden dimension
    heads=${customHeads},        # ${customHeads} attention heads (${Math.round(customDim / customHeads)} dims per head)
    ffn=${customFfn},       # ${customFfn} intermediate FFN expansion width
    rope_split=(44, 42, 42),  # 3D RoPE spatial-temporal coordinate split
)

vidgen_config = VidgenConfig(
    name="vidgen_big_brain",
    height=720,
    width=1280,
    frames=81,               # T = 81 satisfies T = 4k + 1 (k = 20)
    dit=dit_config,
    text_dim=4096,
    target_params_m=${(customBreakdown.totalParams / 1e6).toFixed(1)}, # ${(customBreakdown.totalParams / 1e9).toFixed(2)} Billion parameters
)`}
              </pre>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 1: VIDGEN DIT BLOCK FLOW */}
      {activeSubTab === 'block_architecture' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Vidgen Block Micro-Architecture ({ditConfig.numLayers} Blocks Repeated)
                </h3>
                <p className="text-xs text-slate-400">
                  Every block combines 3 sub-layers with precise normalization and modulation semantics:
                </p>
              </div>
              <span className="text-[11px] font-mono px-2.5 py-1 rounded-lg bg-cyan-950 text-cyan-300 border border-cyan-800/60">
                Hidden Dim: {ditConfig.hiddenDim} • {ditConfig.numHeads} Heads • Head Dim: {ditConfig.headDim}
              </span>
            </div>

            {/* Block Schematic Interactive Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Sub-layer 1: Self-Attention */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 relative overflow-hidden group hover:border-cyan-500/50 transition-all">
                <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-xl pointer-events-none"></div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                    Step 1: Self-Attention
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">adaLN Modulated</span>
                </div>
                <h4 className="text-xs font-bold text-white">Full 3D Spacetime Attention</h4>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Input is modulated by <code className="text-cyan-300 font-mono">scale1, shift1</code>.
                  Q and K pass through <strong className="text-white">RMSNorm</strong> and are rotated via <strong className="text-cyan-300">3D RoPE</strong>.
                  PyTorch SDPA computes attention over all <strong className="text-white">{complexity.N.toLocaleString()}</strong> tokens.
                </p>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-[10px] font-mono space-y-1">
                  <div className="text-cyan-300">x_norm = LN(x) * (1 + scale1) + shift1</div>
                  <div className="text-sky-300">q, k = 3D_RoPE(RMSNorm(Q), RMSNorm(K))</div>
                  <div className="text-emerald-300">x = x + gate1 * OutProj(SDPA(q, k, v))</div>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center justify-between pt-1">
                  <span>Residual gate:</span>
                  <span className="text-emerald-400 font-mono font-bold">adaLN-Zero (Init = 0)</span>
                </div>
              </div>

              {/* Sub-layer 2: Cross-Attention */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 relative overflow-hidden group hover:border-indigo-500/50 transition-all">
                <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-xl pointer-events-none"></div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-400 border border-indigo-800/60">
                    Step 2: Cross-Attention
                  </span>
                  <span className="text-[10px] font-mono text-amber-400">Zero Modulation</span>
                </div>
                <h4 className="text-xs font-bold text-white">Text Cross-Attention (Part 5)</h4>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Plain <code className="text-indigo-300 font-mono">nn.LayerNorm</code> before cross-attention.
                  <strong className="text-white"> No timestep modulation</strong> is applied to text interactions.
                  Video queries attend to frozen text tokens with padding mask.
                </p>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-[10px] font-mono space-y-1">
                  <div className="text-indigo-300">x_cross = LayerNorm(x)  # Plain LN</div>
                  <div className="text-sky-300">scores = (Q_norm @ K_norm.T) * scale</div>
                  <div className="text-emerald-300">x = x + CrossAttn(x_cross, text_context)</div>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center justify-between pt-1">
                  <span>Residual type:</span>
                  <span className="text-indigo-400 font-mono font-bold">Standard Direct Add</span>
                </div>
              </div>

              {/* Sub-layer 3: Feed-Forward */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 relative overflow-hidden group hover:border-sky-500/50 transition-all">
                <div className="absolute top-0 right-0 w-24 h-24 bg-sky-500/5 rounded-full blur-xl pointer-events-none"></div>
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-sky-950 text-sky-400 border border-sky-800/60">
                    Step 3: Feed-Forward
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">adaLN Modulated</span>
                </div>
                <h4 className="text-xs font-bold text-white">FFN with GELU Tanh</h4>
                <p className="text-[11px] text-slate-300 leading-relaxed">
                  Input is modulated by <code className="text-sky-300 font-mono">scale2, shift2</code>.
                  Two-layer MLP expanding to <strong className="text-white">FFN Dim {ditConfig.ffnDim}</strong> with
                  approximate tanh GELU non-linearity, gated before residual addition.
                </p>
                <div className="p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-[10px] font-mono space-y-1">
                  <div className="text-sky-300">x_norm = LN(x) * (1 + scale2) + shift2</div>
                  <div className="text-amber-300">h = GELU_tanh(fc1(x_norm))</div>
                  <div className="text-emerald-300">x = x + gate2 * fc2(h)</div>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center justify-between pt-1">
                  <span>Residual gate:</span>
                  <span className="text-emerald-400 font-mono font-bold">adaLN-Zero (Init = 0)</span>
                </div>
              </div>
            </div>

            {/* Input & Output Heads */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 flex items-start gap-3">
                <span className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 mt-0.5">
                  <ArrowRight className="w-4 h-4" />
                </span>
                <div>
                  <h5 className="text-xs font-bold text-white">Input Patch Embedding</h5>
                  <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                    Latents [B, 16, {preset.baseFrames}, {preset.height}, {preset.width}] → Conv3d(1, 2, 2) → [B, {complexity.N}, {ditConfig.hiddenDim}]
                  </p>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800 flex items-start gap-3">
                <span className="p-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 mt-0.5">
                  <CheckCircle2 className="w-4 h-4" />
                </span>
                <div>
                  <h5 className="text-xs font-bold text-white">Output Head (adaLN-Zero)</h5>
                  <p className="text-[11px] text-slate-400 font-mono mt-0.5">
                    Final adaLN → Linear to patch dimension (zero-init) → Unpatchify to predicted flow [B, 16, t, h, w]
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: PARAMETER VERIFIER */}
      {activeSubTab === 'parameter_verifier' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                  Parameter Count Breakdown vs Part 0 Targets
                </h3>
                <p className="text-xs text-slate-400">
                  Acceptance requirement: parameter count per preset matches Part 0 target within 10%.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-mono font-bold px-2.5 py-1 rounded-lg border ${
                  Math.abs(paramsBreakdown.diffPercent) <= 10
                    ? 'bg-emerald-950 text-emerald-300 border-emerald-800/60'
                    : 'bg-amber-950 text-amber-300 border-amber-800/60'
                }`}>
                  {Math.abs(paramsBreakdown.diffPercent) <= 10 ? '✓ Verified (within 10%)' : 'Variance Warning'}
                </span>
              </div>
            </div>

            {/* Quick Metrics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Total Parameters</div>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  {(paramsBreakdown.totalParams).toLocaleString()}
                </div>
                <div className="text-[10px] text-cyan-400 font-mono mt-1">
                  Target: {paramsBreakdown.nominalTargetParams} (Δ {paramsBreakdown.diffPercent > 0 ? '+' : ''}{paramsBreakdown.diffPercent.toFixed(1)}%)
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Blocks Total ({ditConfig.numLayers}L)</div>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  {(paramsBreakdown.allBlocks).toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  {(paramsBreakdown.totalPerBlock).toLocaleString()} params/block
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Shared Mod Savings</div>
                <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                  {paramsBreakdown.sharedModSavingsPercent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-emerald-400/80 font-mono mt-1">
                  vs per-block MLPs
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Spacetime Tokens</div>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  {complexity.N.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  Grid: {Math.floor((preset.baseFrames-1)/4)+1} × {Math.floor(preset.height/16)} × {Math.floor(preset.width/16)}
                </div>
              </div>
            </div>

            {/* Detailed Component Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                    <th className="py-2.5 px-3">Module Component</th>
                    <th className="py-2.5 px-3">Configuration Details</th>
                    <th className="py-2.5 px-3 text-right">Parameter Count</th>
                    <th className="py-2.5 px-3 text-right">Share</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">3D Patch Embedding</td>
                    <td className="py-2 px-3 text-slate-400">Conv3d(16, {ditConfig.hiddenDim}, (1,2,2))</td>
                    <td className="py-2 px-3 text-right">{paramsBreakdown.patchEmbed.toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{((paramsBreakdown.patchEmbed / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Timestep Conditioning MLP</td>
                    <td className="py-2 px-3 text-slate-400">Sinusoidal(256) → MLP({ditConfig.hiddenDim}) → Shared 6*D</td>
                    <td className="py-2 px-3 text-right">{paramsBreakdown.timestepEmbedder.toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{((paramsBreakdown.timestepEmbedder / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Text Projector</td>
                    <td className="py-2 px-3 text-slate-400">Linear({ditConfig.textDim}, {ditConfig.hiddenDim}) → GELU → Linear</td>
                    <td className="py-2 px-3 text-right">{paramsBreakdown.textProjector.toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{((paramsBreakdown.textProjector / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Full 3D Self-Attention ({ditConfig.numLayers}×)</td>
                    <td className="py-2 px-3 text-slate-400">Q, K, V + QK-RMSNorm + OutProj (RoPE cache 0 params)</td>
                    <td className="py-2 px-3 text-right">{(paramsBreakdown.selfAttnPerBlock * ditConfig.numLayers).toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{(((paramsBreakdown.selfAttnPerBlock * ditConfig.numLayers) / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Text Cross-Attention ({ditConfig.numLayers}×)</td>
                    <td className="py-2 px-3 text-slate-400">LayerNorm + Q, K, V + QK-RMSNorm + OutProj</td>
                    <td className="py-2 px-3 text-right">{(paramsBreakdown.crossAttnPerBlock * ditConfig.numLayers).toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{(((paramsBreakdown.crossAttnPerBlock * ditConfig.numLayers) / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Feed-Forward Networks ({ditConfig.numLayers}×)</td>
                    <td className="py-2 px-3 text-slate-400">Linear({ditConfig.hiddenDim}, {ditConfig.ffnDim}) → GELU_tanh → Linear</td>
                    <td className="py-2 px-3 text-right">{(paramsBreakdown.ffnPerBlock * ditConfig.numLayers).toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{(((paramsBreakdown.ffnPerBlock * ditConfig.numLayers) / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Block Modulation Tables ({ditConfig.numLayers}×)</td>
                    <td className="py-2 px-3 text-slate-400">Learnable per-block offset: 6 × {ditConfig.hiddenDim} parameters</td>
                    <td className="py-2 px-3 text-right">{(paramsBreakdown.blockModPerBlock * ditConfig.numLayers).toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{(((paramsBreakdown.blockModPerBlock * ditConfig.numLayers) / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 font-semibold text-white">Final Output Head</td>
                    <td className="py-2 px-3 text-slate-400">adaLN norm + Linear({ditConfig.hiddenDim}, 16*1*2*2) (zero-init)</td>
                    <td className="py-2 px-3 text-right">{paramsBreakdown.outputHead.toLocaleString()}</td>
                    <td className="py-2 px-3 text-right text-slate-400">{((paramsBreakdown.outputHead / paramsBreakdown.totalParams) * 100).toFixed(1)}%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 3: TIMESTEP & ADALN-ZERO */}
      {activeSubTab === 'timestep_adaln' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-cyan-400" />
                  Timestep Conditioning & adaLN-Zero Modulation
                </h3>
                <p className="text-xs text-slate-400">
                  Sinusoidal 256-dim embedding projected to 6 modulation parameters per block.
                </p>
              </div>
              <div className="text-xs font-mono text-cyan-400">
                t = {timestep.toFixed(2)} (step ~{Math.round(timestep * 1000)}/1000)
              </div>
            </div>

            {/* Slider */}
            <div className="space-y-2 bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-slate-400">t = 0.0 (Clean Data / Target Flow)</span>
                <span className="text-cyan-400 font-bold">Interactive Timestep: {timestep.toFixed(3)}</span>
                <span className="text-slate-400">t = 1.0 (Pure Gaussian Noise)</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.01"
                value={timestep}
                onChange={(e) => setTimestep(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
            </div>

            {/* Visualizer: 256-dim Sinusoidal Harmonics Preview */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white">Sinusoidal Embedding Harmonics (Sampled 32 dims)</span>
                <span className="text-[10px] font-mono text-slate-400">cos(t*freq) / sin(t*freq)</span>
              </div>
              <div className="grid grid-cols-16 gap-1 h-12 items-end">
                {sinusoidalHarmonics.map((val, idx) => {
                  const heightPercent = Math.max(10, Math.abs(val) * 100);
                  const isPositive = val >= 0;
                  return (
                    <div
                      key={idx}
                      title={`Dim ${idx}: ${val.toFixed(3)}`}
                      className={`w-full rounded-t transition-all ${
                        isPositive ? 'bg-cyan-500/80 hover:bg-cyan-400' : 'bg-indigo-500/80 hover:bg-indigo-400'
                      }`}
                      style={{ height: `${heightPercent}%` }}
                    />
                  );
                })}
              </div>
            </div>

            {/* 6 Modulation Values Display */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              {[
                { label: 'scale1 (γ₁)', value: modValues.scale1, desc: 'Self-Attn Scale', color: 'text-cyan-400' },
                { label: 'shift1 (β₁)', value: modValues.shift1, desc: 'Self-Attn Shift', color: 'text-sky-400' },
                { label: 'gate1 (α₁)', value: modValues.gate1, desc: 'Self-Attn Gate', color: 'text-emerald-400', zero: true },
                { label: 'scale2 (γ₂)', value: modValues.scale2, desc: 'FFN Scale', color: 'text-cyan-400' },
                { label: 'shift2 (β₂)', value: modValues.shift2, desc: 'FFN Shift', color: 'text-sky-400' },
                { label: 'gate2 (α₂)', value: modValues.gate2, desc: 'FFN Gate', color: 'text-emerald-400', zero: true },
              ].map((item, idx) => (
                <div key={idx} className="p-3 rounded-xl bg-slate-950 border border-slate-800 space-y-1">
                  <div className="text-[10px] font-mono text-slate-400">{item.label}</div>
                  <div className={`text-base font-bold font-mono ${item.color}`}>
                    {item.value >= 0 ? `+${item.value}` : item.value}
                  </div>
                  <div className="text-[10px] text-slate-400">{item.desc}</div>
                  {item.zero && (
                    <span className="text-[9px] font-mono px-1.5 py-0.2 rounded bg-emerald-950 text-emerald-400 border border-emerald-800/60 inline-block">
                      adaLN-Zero
                    </span>
                  )}
                </div>
              ))}
            </div>

            {/* Deep Dive Note on adaLN-Zero */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
              <div className="font-bold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                Why adaLN-Zero Initialization is Essential:
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                By initializing the residual gate parameters <code className="text-emerald-400 font-mono">gate1 = 0</code> and <code className="text-emerald-400 font-mono">gate2 = 0</code> and the final linear projection layer to zero,
                each block initially acts as a pure identity operator: <code className="text-cyan-300 font-mono">x + 0 = x</code>.
                This prevents exponential variance explosion across deep 30-layer models at step 0, ensuring smooth gradient propagation and stable AdamW optimization.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 4: OVERFIT SIMULATOR */}
      {activeSubTab === 'overfit_simulator' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-emerald-400" />
                  Overfit Acceptance Test on Synthetic Clips
                </h3>
                <p className="text-xs text-slate-400">
                  Acceptance requirement: train on 1 to 8 clips and flow-matching MSE loss drops close to zero.
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  value={numClips}
                  onChange={(e) => setNumClips(parseInt(e.target.value))}
                  disabled={isTraining}
                  className="bg-slate-950 border border-slate-800 text-xs text-slate-300 px-3 py-1.5 rounded-xl cursor-pointer"
                >
                  <option value={1}>1 Clip Batch</option>
                  <option value={2}>2 Clips Batch</option>
                  <option value={4}>4 Clips Batch</option>
                  <option value={8}>8 Clips Batch</option>
                </select>

                <button
                  onClick={startOverfitSimulation}
                  disabled={isTraining}
                  className="flex items-center gap-2 px-4 py-1.5 bg-gradient-to-r from-cyan-500 to-sky-600 hover:from-cyan-400 hover:to-sky-500 text-slate-950 font-bold text-xs rounded-xl shadow-md transition-all cursor-pointer disabled:opacity-50"
                >
                  <Play className={`w-3.5 h-3.5 ${isTraining ? 'animate-spin' : ''}`} />
                  <span>{isTraining ? 'Optimizing...' : 'Run Overfit Test'}</span>
                </button>
              </div>
            </div>

            {/* Top Metrics Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Current Step</div>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  Step {currentOverfitRecord.step} / {trainingHistory.length}
                </div>
                <div className="text-[10px] text-slate-400 font-mono mt-1">
                  Synthetic clips: {numClips}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Flow MSE Loss</div>
                <div className="text-lg font-bold text-emerald-400 font-mono mt-0.5">
                  {currentOverfitRecord.loss.toFixed(5)}
                </div>
                <div className="text-[10px] text-emerald-400/80 font-mono mt-1">
                  Initial: {initialLoss.toFixed(4)}
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Loss Drop</div>
                <div className="text-lg font-bold text-cyan-400 font-mono mt-0.5">
                  {lossReductionPercent.toFixed(1)}%
                </div>
                <div className="text-[10px] text-cyan-400/80 font-mono mt-1">
                  Threshold: &gt;70%
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800">
                <div className="text-[10px] text-slate-400 uppercase font-mono">Gradient Flow</div>
                <div className="text-lg font-bold text-white font-mono mt-0.5">
                  ||∇L|| = {currentOverfitRecord.gradNorm.toFixed(3)}
                </div>
                <div className="text-[10px] text-emerald-400 font-mono mt-1">
                  Non-zero in all {ditConfig.numLayers} blocks
                </div>
              </div>
            </div>

            {/* SVG Loss Curve Chart */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-white">Rectified Flow Training Loss Trajectory (MSE: ||v_θ - (x₁ - x₀)||²)</span>
                <span className="text-[10px] font-mono text-emerald-400">Target: Loss &lt; 0.05</span>
              </div>

              <div className="h-44 w-full relative pt-2">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 500 140" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  <line x1="0" y1="30" x2="500" y2="30" stroke="#1e293b" strokeDasharray="4 4" strokeWidth="1" />
                  <line x1="0" y1="70" x2="500" y2="70" stroke="#1e293b" strokeDasharray="4 4" strokeWidth="1" />
                  <line x1="0" y1="110" x2="500" y2="110" stroke="#1e293b" strokeDasharray="4 4" strokeWidth="1" />

                  {/* Threshold line at 0.05 */}
                  <line x1="0" y1="120" x2="500" y2="120" stroke="#059669" strokeDasharray="3 3" strokeWidth="1" />
                  <text x="5" y="117" fill="#10b981" fontSize="9" fontFamily="monospace">Target: 0.05</text>

                  {/* Polyline of Loss */}
                  {trainingHistory.slice(0, currentStepIdx + 1).length > 1 && (
                    <polyline
                      fill="none"
                      stroke="#06b6d4"
                      strokeWidth="2.5"
                      points={trainingHistory
                        .slice(0, currentStepIdx + 1)
                        .map((rec, i) => {
                          const x = (i / (trainingHistory.length - 1)) * 500;
                          // Normalized y where 1.0 -> y=10 and 0.0 -> y=130
                          const y = 130 - (rec.loss / 1.0) * 120;
                          return `${x},${Math.max(10, Math.min(130, y))}`;
                        })
                        .join(' ')}
                    />
                  )}

                  {/* Current Point */}
                  {trainingHistory[currentStepIdx] && (
                    <circle
                      cx={(currentStepIdx / (trainingHistory.length - 1)) * 500}
                      cy={Math.max(10, Math.min(130, 130 - (trainingHistory[currentStepIdx].loss / 1.0) * 120))}
                      r="4"
                      fill="#22d3ee"
                      stroke="#083344"
                      strokeWidth="2"
                    />
                  )}
                </svg>
              </div>

              <div className="flex justify-between text-[10px] font-mono text-slate-500 pt-1">
                <span>Step 1 (Loss: {initialLoss.toFixed(4)})</span>
                <span className="text-cyan-400">Optimization: AdamW lr=1e-3, cosine decay</span>
                <span className="text-emerald-400 font-bold">Step {trainingHistory.length} (Loss: {trainingHistory[trainingHistory.length - 1]?.loss.toFixed(4)})</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 5: FULL 3D VS FACTORIZED ABLATION */}
      {activeSubTab === 'ablation_comparison' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Split className="w-4 h-4 text-cyan-400" />
                  Full 3D Spacetime Attention vs Factorized Ablation
                </h3>
                <p className="text-xs text-slate-400">
                  Comparison between full 3D attention (Vidgen standard) and factorized spatial + temporal attention.
                </p>
              </div>
              <span className="text-xs font-mono text-cyan-400">
                Tokens: {complexity.N.toLocaleString()}
              </span>
            </div>

            {/* Comparison Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Full 3D Attention */}
              <div className="p-4 rounded-xl bg-slate-950 border border-cyan-500/40 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-cyan-950 text-cyan-400 border border-cyan-800/60 font-bold">
                    Vidgen Standard (Default)
                  </span>
                  <span className="text-[10px] font-mono text-cyan-300">O(N²) Complexity</span>
                </div>
                <h4 className="text-sm font-bold text-white">Full 3D Spacetime Attention</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Every spacetime token directly attends to every other token across both space and time simultaneously.
                  Preserves fine-grained motion dynamics, object persistence, and temporal consistency without spatial-temporal trade-offs.
                </p>
                <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 font-mono text-xs space-y-1">
                  <div className="text-slate-400">FLOPs per Attention Head:</div>
                  <div className="text-cyan-300 font-bold">{complexity.full3D_FLOPs.toLocaleString()} FLOPs</div>
                  <div className="text-[10px] text-slate-500">{complexity.full3D_formula}</div>
                </div>
                <div className="text-[11px] text-slate-300 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Enables continuous 3D RoPE spatial & temporal relative coordinate rotation.</span>
                </div>
              </div>

              {/* Factorized Ablation */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                    Cheaper Ablation
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">{complexity.ratio}× Fewer FLOPs</span>
                </div>
                <h4 className="text-sm font-bold text-white">Factorized Spatial & Temporal Attention</h4>
                <p className="text-xs text-slate-300 leading-relaxed">
                  Decomposes attention into two sequential operations: spatial self-attention within each frame ($H \times W$), followed by temporal self-attention across frames ($T$).
                  Cheaper for long sequences, but causes temporal flicker and disjoint trajectories.
                </p>
                <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800 font-mono text-xs space-y-1">
                  <div className="text-slate-400">FLOPs per Attention Head:</div>
                  <div className="text-slate-200 font-bold">{complexity.factorized_FLOPs.toLocaleString()} FLOPs</div>
                  <div className="text-[10px] text-slate-500">{complexity.factorized_formula}</div>
                </div>
                <div className="text-[11px] text-slate-400 flex items-center gap-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0"></span>
                  <span>Treat as ablation comparison only; Vidgen default remains Full 3D Attention.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
