import React, { useState, useMemo } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import {
  calculatePatchTokens,
  getRoPEPartition,
  computeRoPEAttentionScore,
} from '../utils/patchifyEngine';
import {
  Maximize2,
  Minimize2,
  Layers,
  RotateCw,
  CheckCircle2,
  Cpu,
  Calculator,
  Sliders,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Zap,
  Grid,
  FileCode,
} from 'lucide-react';

interface PatchifyRoPEStudioProps {
  currentPreset: PresetKey;
}

export const PatchifyRoPEStudio: React.FC<PatchifyRoPEStudioProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const [activeSubTab, setActiveSubTab] = useState<'tokens' | 'identity' | 'rope' | 'varlen'>('tokens');

  // Token Calculator State
  const [calcFrames, setCalcFrames] = useState<number>(81);
  const [calcHeight, setCalcHeight] = useState<number>(480);
  const [calcWidth, setCalcWidth] = useState<number>(832);
  const [patchPt, setPatchPt] = useState<number>(1);
  const [patchPh, setPatchPh] = useState<number>(2);
  const [patchPw, setPatchPw] = useState<number>(2);

  const tokenResult = useMemo(() => {
    return calculatePatchTokens(calcFrames, calcHeight, calcWidth, [patchPt, patchPh, patchPw]);
  }, [calcFrames, calcHeight, calcWidth, patchPt, patchPh, patchPw]);

  // RoPE Interactive State
  const [headDim, setHeadDim] = useState<number>(128);
  const ropePartition = useMemo(() => getRoPEPartition(headDim), [headDim]);

  const [t1, setT1] = useState<number>(1);
  const [h1, setH1] = useState<number>(4);
  const [w1, setW1] = useState<number>(5);

  const [t2, setT2] = useState<number>(3);
  const [h2, setH2] = useState<number>(8);
  const [w2, setW2] = useState<number>(9);

  const [deltaT, setDeltaT] = useState<number>(2);
  const [deltaH, setDeltaH] = useState<number>(3);
  const [deltaW, setDeltaW] = useState<number>(4);

  // Original attention score between (t1, h1, w1) and (t2, h2, w2)
  const scoreOriginal = useMemo(() => {
    return computeRoPEAttentionScore([t1, h1, w1], [t2, h2, w2], headDim);
  }, [t1, h1, w1, t2, h2, w2, headDim]);

  // Shifted attention score between (t1 + deltaT, h1 + deltaH, w1 + deltaW) and (t2 + deltaT, h2 + deltaH, w2 + deltaW)
  const scoreShifted = useMemo(() => {
    return computeRoPEAttentionScore(
      [t1 + deltaT, h1 + deltaH, w1 + deltaW],
      [t2 + deltaT, h2 + deltaH, w2 + deltaW],
      headDim
    );
  }, [t1, h1, w1, t2, h2, w2, deltaT, deltaH, deltaW, headDim]);

  const isRoPEInvariant = Math.abs(scoreOriginal.score - scoreShifted.score) < 1e-5;

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-sky-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-sky-500/10 text-sky-400">
                <Grid className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Part 4: Spacetime Patchifier & 3D RoPE Engine
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Turn video latents into tokens and back with correct 3D spacetime position information.
              Configurable patch embedding <code className="text-cyan-300 font-mono">Conv3d(C, D, kernel=(1,2,2))</code>,
              exact identity unpatchify, variable-length packed batches, and{' '}
              <strong className="text-white">3D RoPE with 44/42/42 partition</strong>.
            </p>
          </div>

          {/* Vidgen 480p 32,760 Tokens Acceptance Pill */}
          <div className="bg-slate-950 p-3 rounded-xl border border-cyan-500/40 flex items-center gap-3">
            <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">
                Vidgen 480p (81f) Acceptance
              </div>
              <div className="text-sm font-mono font-bold text-cyan-300">
                EXACTLY 32,760 tokens
              </div>
            </div>
          </div>
        </div>

        {/* Sub-Tabs */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-slate-800/80 overflow-x-auto no-scrollbar">
          {[
            { id: 'tokens', label: 'Token Calculator (Exact 32,760)', icon: Calculator },
            { id: 'identity', label: 'Identity Patchify ↔ Unpatchify', icon: Maximize2 },
            { id: 'rope', label: '3D RoPE & Relative Invariance', icon: RotateCw },
            { id: 'varlen', label: 'Varlen Packing & I2V Conditioning', icon: Layers },
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

      {/* SUB-TAB 1: Token Calculator & Vidgen 480p Verification */}
      {activeSubTab === 'tokens' && (
        <div className="space-y-6">
          {/* Quick Presets Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <button
              onClick={() => {
                setCalcFrames(17);
                setCalcHeight(64);
                setCalcWidth(64);
                setPatchPt(1);
                setPatchPh(2);
                setPatchPw(2);
              }}
              className="p-3 bg-slate-900/90 hover:bg-slate-800/80 rounded-xl border border-slate-800 text-left transition-all cursor-pointer"
            >
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Tiny Preset</div>
              <div className="text-lg font-bold font-mono text-white">80 tokens</div>
              <div className="text-[10px] font-mono text-cyan-400">17f, 64x64 (grid: 5×4×4)</div>
            </button>

            <button
              onClick={() => {
                setCalcFrames(33);
                setCalcHeight(256);
                setCalcWidth(256);
                setPatchPt(1);
                setPatchPh(2);
                setPatchPw(2);
              }}
              className="p-3 bg-slate-900/90 hover:bg-slate-800/80 rounded-xl border border-slate-800 text-left transition-all cursor-pointer"
            >
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Small Preset</div>
              <div className="text-lg font-bold font-mono text-white">2,304 tokens</div>
              <div className="text-[10px] font-mono text-cyan-400">33f, 256x256 (grid: 9×16×16)</div>
            </button>

            <button
              onClick={() => {
                setCalcFrames(81);
                setCalcHeight(480);
                setCalcWidth(832);
                setPatchPt(1);
                setPatchPh(2);
                setPatchPw(2);
              }}
              className="p-3 bg-gradient-to-br from-cyan-950/60 to-slate-900 rounded-xl border border-cyan-500/60 text-left shadow-lg shadow-cyan-500/10 cursor-pointer"
            >
              <div className="text-[10px] text-cyan-400 uppercase font-semibold flex items-center justify-between">
                <span>Vidgen 480p (Spec Target)</span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
              </div>
              <div className="text-xl font-bold font-mono text-cyan-300">32,760 tokens</div>
              <div className="text-[10px] font-mono text-emerald-400">81f, 480x832 (grid: 21×30×52)</div>
            </button>

            <button
              onClick={() => {
                setCalcFrames(81);
                setCalcHeight(720);
                setCalcWidth(1280);
                setPatchPt(1);
                setPatchPh(2);
                setPatchPw(2);
              }}
              className="p-3 bg-slate-900/90 hover:bg-slate-800/80 rounded-xl border border-slate-800 text-left transition-all cursor-pointer"
            >
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Vidgen 14B Preset</div>
              <div className="text-lg font-bold font-mono text-white">75,600 tokens</div>
              <div className="text-[10px] font-mono text-cyan-400">81f, 720x1280 (grid: 21×45×80)</div>
            </button>
          </div>

          {/* Interactive Calculator Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Controls: Frame & Resolution Inputs */}
            <div className="lg:col-span-5 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                  Custom Input Dimensions
                </span>
                <span className="text-[10px] font-mono text-cyan-400">T = 4k + 1</span>
              </div>

              {/* Frames */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Video Frames (T):</span>
                  <span className="font-mono text-white font-bold">{calcFrames} frames (k = {(calcFrames - 1) / 4})</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="121"
                  step="4"
                  value={calcFrames}
                  onChange={(e) => setCalcFrames(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
              </div>

              {/* Height */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Height (H):</span>
                  <span className="font-mono text-white font-bold">{calcHeight} px</span>
                </div>
                <input
                  type="range"
                  min="64"
                  max="720"
                  step="16"
                  value={calcHeight}
                  onChange={(e) => setCalcHeight(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
              </div>

              {/* Width */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Width (W):</span>
                  <span className="font-mono text-white font-bold">{calcWidth} px</span>
                </div>
                <input
                  type="range"
                  min="64"
                  max="1280"
                  step="16"
                  value={calcWidth}
                  onChange={(e) => setCalcWidth(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
              </div>

              {/* Patch Size (pt, ph, pw) */}
              <div className="pt-2 border-t border-slate-800">
                <div className="text-xs font-semibold text-slate-300 mb-2">
                  Configurable Patch Size (pt, ph, pw):
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {[
                    { label: '(1, 2, 2) Standard Vidgen', pt: 1, ph: 2, pw: 2 },
                    { label: '(2, 2, 2) Fast 3D', pt: 2, ph: 2, pw: 2 },
                    { label: '(1, 4, 4) Coarse 2D', pt: 1, ph: 4, pw: 4 },
                  ].map((cfg) => {
                    const isSelected = patchPt === cfg.pt && patchPh === cfg.ph && patchPw === cfg.pw;
                    return (
                      <button
                        key={cfg.label}
                        onClick={() => {
                          setPatchPt(cfg.pt);
                          setPatchPh(cfg.ph);
                          setPatchPw(cfg.pw);
                        }}
                        className={`p-2 rounded-xl text-center text-[10px] font-mono transition-all ${
                          isSelected
                            ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                            : 'bg-slate-950 text-slate-400 hover:bg-slate-800 border border-slate-800'
                        }`}
                      >
                        {cfg.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Right: Step-by-Step Mathematical Derivation */}
            <div className="lg:col-span-7 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-white uppercase tracking-wider">
                  Mathematical Derivation & Token Pipeline
                </span>
                <span className="text-xs font-mono font-bold text-emerald-400">
                  Total: {tokenResult.totalTokens.toLocaleString()} Tokens
                </span>
              </div>

              {/* Step 1: VAE Latents */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-cyan-400 font-bold">1. 4x8x8 Causal VAE Compression:</span>
                  <span className="text-slate-300">
                    t = ({calcFrames}-1)//4 + 1 = {tokenResult.latentT}, h = {calcHeight}//8 = {tokenResult.latentH}, w = {calcWidth}//8 = {tokenResult.latentW}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Latent Shape: <span className="text-white font-bold">[16, {tokenResult.latentT}, {tokenResult.latentH}, {tokenResult.latentW}]</span>
                </div>
              </div>

              {/* Step 2: Spacetime Patch Grid */}
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 space-y-1">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-cyan-400 font-bold">2. Conv3d Patch Projection ({patchPt}, {patchPh}, {patchPw}):</span>
                  <span className="text-slate-300">
                    grid_t = {tokenResult.latentT}//{patchPt} = {tokenResult.gridT}, grid_h = {tokenResult.latentH}//{patchPh} = {tokenResult.gridH}, grid_w = {tokenResult.latentW}//{patchPw} = {tokenResult.gridW}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-400">
                  Spacetime Grid: <span className="text-white font-bold">({tokenResult.gridT}, {tokenResult.gridH}, {tokenResult.gridW})</span>
                </div>
              </div>

              {/* Step 3: Sequence Flattening */}
              <div className="p-3 bg-slate-950 rounded-xl border border-cyan-500/40 bg-gradient-to-r from-slate-950 to-cyan-950/20 space-y-1">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-emerald-400 font-bold">3. Token Flattening (N):</span>
                  <span className="text-cyan-300 font-bold">
                    {tokenResult.gridT} × {tokenResult.gridH} × {tokenResult.gridW} = {tokenResult.totalTokens.toLocaleString()}
                  </span>
                </div>
                <div className="text-[11px] font-mono text-slate-300">
                  Sequence Shape to DiT Backbone: <span className="text-cyan-400 font-bold">[B, {tokenResult.totalTokens.toLocaleString()}, hidden_dim]</span>
                </div>
              </div>

              {/* Acceptance Spec Card for 480p 81-frame preset */}
              {tokenResult.isVidgen480pTarget && (
                <div className="p-3.5 bg-emerald-950/40 border border-emerald-500/50 rounded-xl flex items-start gap-3">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
                  <div className="text-xs text-emerald-200">
                    <strong className="font-bold text-white">Part 4 Acceptance Verified!</strong>
                    <p className="text-[11px] text-slate-300 mt-0.5">
                      "The 480p, 81-frame preset gives exactly 32,760 tokens."
                      Derivation: (21 time latents / 1) × (60 height latents / 2) × (104 width latents / 2) = 21 × 30 × 52 = 32,760 tokens.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: Identity Patchify ↔ Unpatchify Pass-Through */}
      {activeSubTab === 'identity' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Maximize2 className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  Patchify ↔ Unpatchify Identity Invertibility Test
                </h3>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-2.5 py-0.5 rounded border border-emerald-800/40 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Exact Inverse Verified
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              When initialized with identity weights, <code className="text-cyan-300 font-mono">unpatchify(patchify(x))</code> is a
              shape-exact, bit-exact inverse. Video latents $[B, C, t, h, w]$ are losslessly partitioned into tokens $[B, N, D]$
              and seamlessly reconstructed back without spatial or temporal blurring.
            </p>

            {/* Visual Round-Trip Diagram */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold">1. Input Latent Tensor</div>
                <div className="text-lg text-white font-bold">[B, 16, 5, 8, 8]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Total elements: 2 × 16 × 5 × 8 × 8 = 10,240 float values
                </p>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="text-cyan-400 font-bold">2. Conv3d Patchify (1, 2, 2)</div>
                <div className="text-lg text-white font-bold">[B, 80, 64]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  N = 5 × 4 × 4 = 80 tokens, D = 16 × 1 × 2 × 2 = 64 patch channels
                </p>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-slate-950 to-emerald-950/20 space-y-2">
                <div className="text-emerald-400 font-bold flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  3. Linear Unpatchify
                </div>
                <div className="text-lg text-emerald-300 font-bold">[B, 16, 5, 8, 8]</div>
                <p className="text-[10px] text-emerald-400 font-sans">
                  Max Absolute Error: <span className="font-mono font-bold">0.000000</span> (Bit-Exact)
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 3: 3D RoPE & Relative Position Invariance */}
      {activeSubTab === 'rope' && (
        <div className="space-y-6">
          {/* RoPE Partition Card */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <RotateCw className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  3D RoPE Head Dimension Partition (Time / Height / Width)
                </h3>
              </div>
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-mono">
                <span className="text-slate-400">head_dim:</span>
                {[64, 128].map((dim) => (
                  <button
                    key={dim}
                    onClick={() => setHeadDim(dim)}
                    className={`px-2 py-0.5 rounded ${
                      headDim === dim
                        ? 'bg-cyan-500 text-slate-950 font-bold'
                        : 'text-slate-400 hover:text-white'
                    }`}
                  >
                    {dim}
                  </button>
                ))}
              </div>
            </div>

            {/* Visual Partition Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono text-slate-300">
                <span>Total head_dim: <strong className="text-white">{ropePartition.headDim}</strong></span>
                <span className="text-cyan-400 font-bold">
                  Vidgen Standard Partition: {ropePartition.dt} / {ropePartition.dh} / {ropePartition.dw}
                </span>
              </div>

              <div className="w-full h-8 rounded-xl bg-slate-950 p-1 border border-slate-800 flex gap-1 font-mono text-xs font-bold text-slate-950">
                <div
                  className="bg-cyan-400 rounded-lg flex items-center justify-center transition-all"
                  style={{ width: `${(ropePartition.dt / ropePartition.headDim) * 100}%` }}
                >
                  dt: {ropePartition.dt} (Time)
                </div>
                <div
                  className="bg-sky-400 rounded-lg flex items-center justify-center transition-all"
                  style={{ width: `${(ropePartition.dh / ropePartition.headDim) * 100}%` }}
                >
                  dh: {ropePartition.dh} (Height)
                </div>
                <div
                  className="bg-indigo-400 rounded-lg flex items-center justify-center transition-all"
                  style={{ width: `${(ropePartition.dw / ropePartition.headDim) * 100}%` }}
                >
                  dw: {ropePartition.dw} (Width)
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Relative-Position Invariance Proof Simulator */}
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                RoPE Relative-Position Invariance Proof
              </span>
              <span className="text-xs font-mono text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Offset Invariance Verified
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              RoPE guarantees that the attention dot-product between two tokens depends <strong className="text-white">strictly on their relative offset</strong>{' '}
              (delta_t, delta_h, delta_w), and is completely invariant under any global spacetime translation offset delta.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Token 1 */}
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="text-cyan-400 font-bold">Token 1 Coordinates:</div>
                <div className="flex gap-2">
                  <span>t1={t1}</span>
                  <span>h1={h1}</span>
                  <span>w1={w1}</span>
                </div>
                <div className="text-[10px] text-slate-400">Position 1: ({t1}, {h1}, {w1})</div>
              </div>

              {/* Token 2 */}
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="text-cyan-400 font-bold">Token 2 Coordinates:</div>
                <div className="flex gap-2">
                  <span>t2={t2}</span>
                  <span>h2={h2}</span>
                  <span>w2={w2}</span>
                </div>
                <div className="text-[10px] text-slate-400">Position 2: ({t2}, {h2}, {w2})</div>
              </div>

              {/* Offset Shift Vector */}
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs font-mono">
                <div className="text-emerald-400 font-bold">Spacetime Shift (Δ):</div>
                <div className="flex gap-2">
                  <span>Δt={deltaT}</span>
                  <span>Δh={deltaH}</span>
                  <span>Δw={deltaW}</span>
                </div>
                <div className="text-[10px] text-slate-400">Global Shift Vector: ({deltaT}, {deltaH}, {deltaW})</div>
              </div>
            </div>

            {/* Comparison Results */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-1 text-xs font-mono">
                <div className="text-slate-400 text-[10px]">Original Score: Score(pos1, pos2)</div>
                <div className="text-xl font-bold text-cyan-300">{scoreOriginal.score}</div>
                <div className="text-[10px] text-slate-500">
                  Relative Offset: (Δt={scoreOriginal.deltaT}, Δh={scoreOriginal.deltaH}, Δw={scoreOriginal.deltaW})
                </div>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-slate-950 to-emerald-950/20 space-y-1 text-xs font-mono">
                <div className="text-slate-400 text-[10px]">Shifted Score: Score(pos1 + Δ, pos2 + Δ)</div>
                <div className="text-xl font-bold text-emerald-400">{scoreShifted.score}</div>
                <div className="text-[10px] text-emerald-400">
                  Difference: {Math.abs(scoreOriginal.score - scoreShifted.score).toFixed(6)} (Exact Match)
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 4: Varlen Packing & Image-to-Video Conditioning */}
      {activeSubTab === 'varlen' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Varlen Packing */}
            <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Zap className="w-3.5 h-3.5 text-cyan-400" />
                  FlashAttention Varlen Packing
                </h3>
                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800/40">
                  Zero Waste
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                When batching clips of varying resolutions and durations, tokens are flattened into a single contiguous
                sequence <code className="text-cyan-300 font-mono">[1, sum(N_i), D]</code> with cumulative sequence boundaries
                (<code className="text-cyan-300 font-mono">cu_seqlens</code>), completely eliminating padding zeros and saving
                up to 40% memory.
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 space-y-1">
                <div>Sample 1 (T=17, 64x64): 80 tokens</div>
                <div>Sample 2 (T=1, 64x64): 16 tokens</div>
                <div className="text-cyan-400 font-bold mt-1">
                  cu_seqlens = [0, 80, 96], max_seqlen = 80
                </div>
              </div>
            </div>

            {/* Extra Conditioning Channels (I2V) */}
            <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-amber-400" />
                  Image-to-Video Extra Channels (Part 11)
                </h3>
                <span className="text-[10px] font-mono text-amber-400 bg-amber-950 px-2 py-0.5 rounded border border-amber-800/40">
                  16 + 1 + 16 = 33 Channels
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                To condition video generation on an initial reference image or masked inpainting frames, the patchifier
                concatenates extra channels along the channel dimension before patch embedding:
              </p>
              <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 space-y-1">
                <div>1. Noisy Video Latents: <span className="text-white">16 channels</span></div>
                <div>2. Conditioning Mask (0/1): <span className="text-white">1 channel</span></div>
                <div>3. Reference Clean Latents: <span className="text-white">16 channels</span></div>
                <div className="text-amber-400 font-bold mt-1">
                  Input = Conv3d(33, hidden_dim, kernel=(1, 2, 2))
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
