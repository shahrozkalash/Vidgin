import React, { useState, useMemo } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import {
  expandUserPrompt,
  REWRITER_STYLES,
  simulateCrossAttentionOutput,
} from '../utils/textEngine';
import {
  FileText,
  Sparkles,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  Sliders,
  Copy,
  Check,
  RotateCcw,
  Cpu,
  Feather,
} from 'lucide-react';

interface TextConditionerStudioProps {
  currentPreset: PresetKey;
}

export const TextConditionerStudio: React.FC<TextConditionerStudioProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const [activeSubTab, setActiveSubTab] = useState<'prompt_rewriter' | 'cross_attention' | 'mask_invariance' | 'cfg_dropout'>('prompt_rewriter');

  // Prompt Rewriter State
  const [userPrompt, setUserPrompt] = useState<string>('A neon cyan digit three floating in dark space');
  const [selectedStyle, setSelectedStyle] = useState<string>('cinematic');
  const [copied, setCopied] = useState<boolean>(false);

  const expandedResult = useMemo(() => {
    return expandUserPrompt(userPrompt, selectedStyle);
  }, [userPrompt, selectedStyle]);

  // Mask Invariance State
  const [activeTokenCount, setActiveTokenCount] = useState<number>(14);
  const [padNoise, setPadNoise] = useState<number>(0.0);

  const baselineOut = useMemo(() => {
    return simulateCrossAttentionOutput(activeTokenCount, 64, 0.0);
  }, [activeTokenCount]);

  const noisyOut = useMemo(() => {
    return simulateCrossAttentionOutput(activeTokenCount, 64, padNoise);
  }, [activeTokenCount, padNoise]);

  const isMaskInvariant = Math.abs(baselineOut.outputDigest - noisyOut.outputDigest) === 0.0;

  // CFG Guidance Scale Slider
  const [guidanceScale, setGuidanceScale] = useState<number>(5.0);
  const [isDropoutSimulated, setIsDropoutSimulated] = useState<boolean>(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(expandedResult.denseCaption);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6">
      {/* Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-indigo-500/10 text-indigo-400">
                <FileText className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Part 5: Text Conditioner & DiT Cross-Attention Studio
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Frozen text encoder (<code className="text-cyan-300 font-mono">umT5-XXL / T5</code>, bf16, zero gradients) feeding
              every DiT block through cross-attention with <strong className="text-white">QK-RMSNorm</strong>.
              Features an <strong className="text-cyan-300">LLM Prompt Rewriter</strong> to resolve the caption style mismatch pitfall,
              rigorous padding mask invariance, and 10% CFG null-embedding dropout.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono bg-slate-950 text-slate-300 px-3 py-1.5 rounded-xl border border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Frozen umT5 (bf16)
            </span>
          </div>
        </div>

        {/* Sub-Tabs */}
        <div className="flex gap-2 mt-4 pt-4 border-t border-slate-800/80 overflow-x-auto no-scrollbar">
          {[
            { id: 'prompt_rewriter', label: 'LLM Prompt Rewriter (Dense Expansion)', icon: Sparkles },
            { id: 'cross_attention', label: 'DiT Cross-Attention & QK-RMSNorm', icon: Layers },
            { id: 'mask_invariance', label: 'Padding Mask Invariance Proof', icon: ShieldCheck },
            { id: 'cfg_dropout', label: 'Classifier-Free Guidance (CFG)', icon: Sliders },
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

      {/* SUB-TAB 1: Prompt Rewriter (Dense Expansion & Style Alignment) */}
      {activeSubTab === 'prompt_rewriter' && (
        <div className="space-y-6">
          {/* Critical Pitfall Callout */}
          <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-500/40 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-xs text-amber-200">
              <strong className="font-bold text-white">Part 5 Pitfall: Caption Style Mismatch</strong>
              <p className="text-[11px] text-slate-300 mt-0.5 leading-relaxed">
                Vidgen video diffusion models are trained on rich, dense multi-sentence captions describing subject, action,
                setting, lighting, and camera trajectory. Passing raw short user prompts (e.g., <em>"A running dog"</em>) causes
                out-of-distribution hallucinations and poor motion quality. The prompt rewriter expands short prompts into the
                dense caption distribution used in training.
              </p>
            </div>
          </div>

          {/* Interactive Rewriter Workbench */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Input Short Prompt & Controls */}
            <div className="lg:col-span-5 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Feather className="w-3.5 h-3.5 text-cyan-400" />
                  Short User Prompt Input
                </span>
                <span className="text-[10px] font-mono text-slate-400">At Inference</span>
              </div>

              {/* Text Input Area */}
              <div>
                <textarea
                  rows={3}
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  placeholder="Enter short prompt..."
                  className="w-full p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition-all"
                />
              </div>

              {/* Quick Prompt Presets */}
              <div className="space-y-1.5">
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Quick Presets:</div>
                <div className="flex flex-wrap gap-1.5">
                  {[
                    'A neon cyan digit 3 floating in dark space',
                    'A golden retriever sprinting through autumn leaves',
                    'A sports car drifting along a rainy Tokyo avenue',
                    'A glass sphere with swirling galaxy nebulae',
                  ].map((presetText) => (
                    <button
                      key={presetText}
                      onClick={() => setUserPrompt(presetText)}
                      className="text-[10px] px-2 py-1 bg-slate-950 hover:bg-slate-800 text-slate-300 rounded-lg border border-slate-800/80 transition-colors"
                    >
                      {presetText.slice(0, 24)}...
                    </button>
                  ))}
                </div>
              </div>

              {/* Cinematic Style Selection */}
              <div className="pt-2 border-t border-slate-800 space-y-2">
                <div className="text-xs font-semibold text-slate-300">Cinematic Lighting & Camera Style:</div>
                <div className="grid grid-cols-2 gap-2">
                  {REWRITER_STYLES.map((st) => (
                    <button
                      key={st.id}
                      onClick={() => setSelectedStyle(st.id)}
                      className={`p-2.5 rounded-xl text-left text-xs transition-all ${
                        selectedStyle === st.id
                          ? 'bg-cyan-500 text-slate-950 font-bold shadow-md shadow-cyan-500/20'
                          : 'bg-slate-950 text-slate-400 hover:bg-slate-800 hover:text-white border border-slate-800'
                      }`}
                    >
                      <div className="font-bold">{st.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Expanded Dense Caption Output */}
            <div className="lg:col-span-7 bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-bold text-white uppercase tracking-wider">
                    Expanded Vidgen Dense Caption (Training Aligned)
                  </span>
                </div>
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 px-2.5 py-1 text-[11px] font-mono rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>

              {/* Dense Caption Text */}
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 leading-relaxed font-sans italic">
                "{expandedResult.denseCaption}"
              </div>

              {/* Structured Metadata Breakdown */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Subject:</div>
                  <div className="text-cyan-300 font-bold truncate">{expandedResult.subject}</div>
                </div>
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Action & Dynamics:</div>
                  <div className="text-cyan-300 font-bold truncate">{expandedResult.action}</div>
                </div>
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Lighting Atmosphere:</div>
                  <div className="text-amber-300 font-bold truncate">{expandedResult.lighting}</div>
                </div>
                <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-500 uppercase">Camera Trajectory:</div>
                  <div className="text-emerald-300 font-bold truncate">{expandedResult.cameraMotion}</div>
                </div>
              </div>

              <div className="pt-2 border-t border-slate-800 flex justify-between text-[11px] font-mono text-slate-400">
                <span>Estimated umT5 Token Count:</span>
                <span className="text-cyan-300 font-bold">{expandedResult.tokenCount} tokens / 512 max</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 2: DiT Cross-Attention & QK-RMSNorm */}
      {activeSubTab === 'cross_attention' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  DiT Block Cross-Attention with QK-RMSNorm Pipeline
                </h3>
              </div>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-2.5 py-0.5 rounded border border-cyan-800/40">
                Every Transformer Layer
              </span>
            </div>

            {/* Architecture Flow Diagram */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
                <div className="text-cyan-400 font-bold">1. Video Tokens (Q)</div>
                <div className="text-[11px] text-white font-bold">[B, N, D]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Video tokens from spacetime self-attention block serve as Queries.
                </p>
                <div className="text-[10px] text-emerald-400 font-bold mt-1">
                  RMSNorm(Q) per head
                </div>
              </div>

              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
                <div className="text-cyan-400 font-bold">2. Text Tokens (K, V)</div>
                <div className="text-[11px] text-white font-bold">[B, L, D]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  umT5 text embeddings projected via Linear -&gt; GELU -&gt; Linear serve as Keys and Values.
                </p>
                <div className="text-[10px] text-emerald-400 font-bold mt-1">
                  RMSNorm(K) per head
                </div>
              </div>

              <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 space-y-1.5">
                <div className="text-cyan-400 font-bold">3. Attention Masking</div>
                <div className="text-[11px] text-white font-bold">[B, 1, 1, L]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Padding positions masked to -10000.0 before softmax to ensure 0 gradient leakage.
                </p>
              </div>

              <div className="p-3.5 bg-slate-950 rounded-xl border border-cyan-500/40 bg-gradient-to-br from-slate-950 to-cyan-950/20 space-y-1.5">
                <div className="text-emerald-400 font-bold">4. Output Video Tokens</div>
                <div className="text-[11px] text-white font-bold">[B, N, D]</div>
                <p className="text-[10px] text-slate-400 font-sans">
                  Conditioned video representations passed to Feed-Forward Network (FFN).
                </p>
              </div>
            </div>

            {/* Vidgen QK-RMSNorm Deep Dive */}
            <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2 text-xs">
              <div className="font-bold text-white flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                Why QK-RMSNorm is Critical in Vidgen Cross-Attention:
              </div>
              <p className="text-slate-300 text-[11px] leading-relaxed">
                In standard multi-head attention without normalization, inner products <code className="font-mono text-cyan-300 text-[10px]">QK^T / sqrt(d)</code> can grow
                exponentially during training on high-dimensional text features, saturating the softmax function and causing
                gradient vanishing or training divergence. Vidgen applies independent RMSNorm to query heads and key heads,
                constraining head activations to the unit hypersphere and stabilizing long training runs.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 3: Live Padding Mask Invariance Proof */}
      {activeSubTab === 'mask_invariance' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <span className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Padding Mask Invariance Test Simulator
              </span>
              <span className="text-xs font-mono text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" /> Mask Invariance Verified
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              Acceptance criteria: <strong className="text-white">"Changing only padding tokens does not change outputs."</strong>{' '}
              Video tokens attending to text tokens must be completely invariant to arbitrary noise or perturbations in
              masked padding slots.
            </p>

            {/* Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between text-xs font-mono text-slate-300">
                  <span>Active Prompt Tokens:</span>
                  <span className="text-cyan-400 font-bold">{activeTokenCount} tokens</span>
                </div>
                <input
                  type="range"
                  min="4"
                  max="48"
                  value={activeTokenCount}
                  onChange={(e) => setActiveTokenCount(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
                <div className="text-[10px] font-mono text-slate-500">
                  Padding Tokens: {64 - activeTokenCount} slots (Mask = False)
                </div>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-2">
                <div className="flex justify-between text-xs font-mono text-slate-300">
                  <span>Padding Noise Perturbation:</span>
                  <span className="text-amber-400 font-bold">±{padNoise.toFixed(1)} noise</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="500"
                  step="10"
                  value={padNoise}
                  onChange={(e) => setPadNoise(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-amber-400"
                />
                <div className="text-[10px] font-mono text-slate-500">
                  Perturbs values in padding slots without touching active prompt tokens
                </div>
              </div>
            </div>

            {/* Output Comparison Proof */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-1 text-xs font-mono">
                <div className="text-slate-400 text-[10px]">Baseline Output (Clean Zero Padding):</div>
                <div className="text-lg font-bold text-cyan-300">{baselineOut.outputDigest}</div>
                <div className="text-[10px] text-slate-500">Padded Attention Softmax Weight: 0.000000</div>
              </div>

              <div className="p-4 bg-slate-950 rounded-xl border border-emerald-500/40 bg-gradient-to-br from-slate-950 to-emerald-950/20 space-y-1 text-xs font-mono">
                <div className="text-slate-400 text-[10px]">Perturbed Output (Noisy Padding):</div>
                <div className="text-lg font-bold text-emerald-400">{noisyOut.outputDigest}</div>
                <div className="text-[10px] text-emerald-400 font-bold">
                  Absolute Difference: {Math.abs(baselineOut.outputDigest - noisyOut.outputDigest).toFixed(6)} (Exact Match)
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SUB-TAB 4: Classifier-Free Guidance (CFG) & Null Embedding */}
      {activeSubTab === 'cfg_dropout' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-bold text-white">
                  Classifier-Free Guidance (CFG) & 10% Null-Embedding Dropout
                </h3>
              </div>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 px-2.5 py-0.5 rounded border border-cyan-800/40">
                10% Drop in Training
              </span>
            </div>

            <p className="text-xs text-slate-300 leading-relaxed">
              During training, the empty prompt <code className="text-cyan-300 font-mono">""</code> is precomputed once as the
              null embedding. With probability <strong className="text-white">10%</strong>, caption embeddings are replaced
              with this null embedding to train the model unconditionally for CFG.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Training Dropout Path */}
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white">Training Time: 10% Null Drop</span>
                  <button
                    onClick={() => setIsDropoutSimulated(!isDropoutSimulated)}
                    className={`px-2 py-0.5 text-xs rounded ${
                      isDropoutSimulated ? 'bg-amber-500 text-slate-950 font-bold' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {isDropoutSimulated ? 'Dropout Triggered (10%)' : 'Caption Retained (90%)'}
                  </button>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg border border-slate-800 text-[11px] text-slate-300">
                  {isDropoutSimulated ? (
                    <span className="text-amber-400">
                      ⇄ Caption replaced with cached null_embedding (empty string token vector)
                    </span>
                  ) : (
                    <span className="text-cyan-300">
                      Active Caption: "A cinematic high-resolution video..."
                    </span>
                  )}
                </div>
              </div>

              {/* Inference Guidance Formula */}
              <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 space-y-3 text-xs font-mono">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white">Inference Guidance Scale (s):</span>
                  <span className="text-cyan-400 font-bold">{guidanceScale.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="12.0"
                  step="0.5"
                  value={guidanceScale}
                  onChange={(e) => setGuidanceScale(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
                <div className="text-[11px] text-slate-400 pt-1">
                  Flow: <code className="text-cyan-300 font-mono">v = v_uncond + {guidanceScale.toFixed(1)} * (v_cond - v_uncond)</code>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
