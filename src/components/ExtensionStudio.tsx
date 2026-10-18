import React, { useState, useMemo } from 'react';
import {
  Sparkles,
  Layers,
  Cpu,
  Zap,
  Shield,
  Clock,
  Film,
  Scissors,
  CheckCircle2,
  AlertTriangle,
  Sliders,
  ChevronRight,
  Database,
  ArrowRight,
  RefreshCw,
  HardDrive,
  Eye,
  SlidersHorizontal,
  Key,
  Flame,
  FileCheck,
  TrendingUp,
  FileText,
  Activity,
} from 'lucide-react';
import {
  LORA_PRESETS,
  LoRAPreset,
  MILESTONES,
  calculateLoRAParameters,
  calculateLongVideoPlan,
  validatePromptSafety,
  generateC2PAManifest,
} from '../utils/extensionEngine';

type ExtensionTab = 'lora' | 'i2v' | 'dual_experts' | 'distillation' | 'long_video' | 'safety' | 'milestones';

export const ExtensionStudio: React.FC = () => {
  const [activeTab, setActiveTab] = useState<ExtensionTab>('lora');

  // LoRA State
  const [selectedModel, setSelectedModel] = useState<'1.3B' | '14B' | 'small' | 'tiny'>('1.3B');
  const [loraRank, setLoraRank] = useState<number>(32);
  const [loraAlpha, setLoraAlpha] = useState<number>(32);
  const [targetModules, setTargetModules] = useState<string[]>([
    'q_proj',
    'k_proj',
    'v_proj',
    'out_proj',
    'fc1',
    'fc2',
  ]);
  const [activePreset, setActivePreset] = useState<LoRAPreset>(LORA_PRESETS[0]);
  const [isMerged, setIsMerged] = useState<boolean>(false);

  // I2V State
  const [i2vConditioningFrame, setI2vConditioningFrame] = useState<'first' | 'last' | 'custom'>('first');
  const [i2vImageGuidance, setI2vImageGuidance] = useState<number>(2.0);
  const [zeroInitVerified, setZeroInitVerified] = useState<boolean>(true);

  // Dual Experts State
  const [expertBoundary, setExpertBoundary] = useState<number>(0.875);
  const [totalSamplingSteps, setTotalSamplingSteps] = useState<number>(50);
  const [simulateOffloadStep, setSimulateOffloadStep] = useState<number>(20);

  // Distillation State
  const [fewSteps, setFewSteps] = useState<4 | 6 | 8>(4);
  const [distillShift, setDistillShift] = useState<number>(3.0);
  const [distillMethod, setDistillMethod] = useState<'reflow' | 'consistency'>('reflow');

  // Long Video State
  const [targetVideoFrames, setTargetVideoFrames] = useState<number>(161);
  const [temporalBlendMode, setTemporalBlendMode] = useState<'cosine' | 'linear'>('cosine');

  // Safety State
  const [testPrompt, setTestPrompt] = useState<string>(
    'A cinematic drone flight over a cyberpunk city with neon reflections and heavy rain'
  );
  const [watermarkKey, setWatermarkKey] = useState<string>('VIDGEN_WAN_AUTHENTIC_2026');

  // Calculations
  const loraStats = useMemo(
    () => calculateLoRAParameters(selectedModel, loraRank, targetModules),
    [selectedModel, loraRank, targetModules]
  );

  const longVideoPlan = useMemo(
    () => calculateLongVideoPlan(targetVideoFrames, 81, 16),
    [targetVideoFrames]
  );

  const safetyResult = useMemo(
    () => validatePromptSafety(testPrompt),
    [testPrompt]
  );

  const c2paManifest = useMemo(
    () => generateC2PAManifest(testPrompt, `Wan-2.1-${selectedModel}`, 42, 5.0, 50),
    [testPrompt, selectedModel]
  );

  // Dual expert step breakdown
  const dualExpertSteps = useMemo(() => {
    const steps = [];
    for (let i = 0; i <= totalSamplingSteps; i++) {
      const t = 1.0 - i / totalSamplingSteps;
      const isHigh = t >= expertBoundary;
      steps.push({ step: i, t: Number(t.toFixed(3)), isHigh });
    }
    return steps;
  }, [totalSamplingSteps, expertBoundary]);

  const highExpertStepCount = dualExpertSteps.filter((s) => s.isHigh).length;
  const lowExpertStepCount = dualExpertSteps.length - highExpertStepCount;

  const toggleTargetModule = (mod: string) => {
    if (targetModules.includes(mod)) {
      if (targetModules.length > 1) {
        setTargetModules(targetModules.filter((m) => m !== mod));
      }
    } else {
      setTargetModules([...targetModules, mod]);
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6 max-w-7xl mx-auto w-full text-slate-100">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950/60 to-slate-900 border border-indigo-900/40 shadow-xl">
        <div className="flex items-start gap-4">
          <div className="p-3 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
            <Zap className="w-7 h-7" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h1 className="text-xl font-bold text-white tracking-tight">
                Part 11: Post-Training & Extensions Studio
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full font-mono bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                Foundation Extensions
              </span>
            </div>
            <p className="text-sm text-slate-400 leading-relaxed max-w-3xl">
              Post-training capabilities for Wan-class video generation: LoRA parameter-efficient adaptation, 33-channel Image-to-Video with zero-init patch embeddings, Wan 2.2 dual-expert schedule partitioning, 4-step rectified flow reflow distillation, long video chunking, C2PA provenance watermarking, and Milestones M1–M5.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 self-end md:self-auto shrink-0">
          <div className="flex flex-col items-end text-xs">
            <span className="text-slate-400">Track B Target</span>
            <span className="font-semibold text-emerald-400 font-mono">Wan 1.3B / 14B</span>
          </div>
          <div className="h-8 w-px bg-slate-800" />
          <div className="flex flex-col items-end text-xs">
            <span className="text-slate-400">Parameter Budget</span>
            <span className="font-semibold text-indigo-300 font-mono">{loraStats.trainableRatioPercent}%</span>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex items-center gap-1.5 p-1.5 rounded-xl bg-slate-900/90 border border-slate-800 overflow-x-auto scrollbar-none">
        {[
          { id: 'lora', label: '1. LoRA Fine-Tune', icon: Layers, badge: 'Track B' },
          { id: 'i2v', label: '2. Image-to-Video (I2V)', icon: Film, badge: '33ch Zero-Init' },
          { id: 'dual_experts', label: '3. Wan 2.2 Dual Experts', icon: Cpu, badge: '0.875 Split' },
          { id: 'distillation', label: '4. Few-Step Distillation', icon: Zap, badge: '4-8 Steps' },
          { id: 'long_video', label: '5. Long Video Chunking', icon: Scissors, badge: 'Cosine Blend' },
          { id: 'safety', label: '6. Safety & C2PA Provenance', icon: Shield, badge: 'Moderation' },
          { id: 'milestones', label: 'Milestones (M1–M5)', icon: TrendingUp, badge: 'Build Order' },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as ExtensionTab)}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-md shadow-indigo-900/30 font-semibold'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{tab.label}</span>
              <span
                className={`text-[10px] px-1.5 py-0.2 rounded font-mono ${
                  isActive ? 'bg-indigo-800/60 text-indigo-200' : 'bg-slate-800 text-slate-400'
                }`}
              >
                {tab.badge}
              </span>
            </button>
          );
        })}
      </div>

      {/* TAB 1: LoRA Fine-Tuning */}
      {activeTab === 'lora' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls & Configuration */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">LoRA Hyperparameter Configuration</h3>
                </div>
                <div className="text-xs font-mono text-slate-400">
                  Target: Open Wan {selectedModel} Checkpoint
                </div>
              </div>

              {/* Model Preset Selector */}
              <div className="grid grid-cols-4 gap-3">
                {(['tiny', 'small', '1.3B', '14B'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setSelectedModel(m)}
                    className={`flex flex-col items-center justify-center p-3 rounded-xl border text-xs transition-all ${
                      selectedModel === m
                        ? 'bg-indigo-600/10 border-indigo-500 text-indigo-300 font-semibold'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:bg-slate-800/40'
                    }`}
                  >
                    <span className="font-mono text-sm">{m}</span>
                    <span className="text-[10px] text-slate-500 mt-0.5">
                      {m === '1.3B' ? 'Track B Primary' : m === '14B' ? 'Flagship Scale' : 'Lightweight'}
                    </span>
                  </button>
                ))}
              </div>

              {/* Rank & Alpha Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-medium">LoRA Rank (r)</span>
                    <span className="font-mono text-indigo-400 font-semibold">r = {loraRank}</span>
                  </div>
                  <input
                    type="range"
                    min="8"
                    max="64"
                    step="8"
                    value={loraRank}
                    onChange={(e) => {
                      const r = Number(e.target.value);
                      setLoraRank(r);
                      if (loraAlpha < r) setLoraAlpha(r);
                    }}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>r=8 (Ultra Light)</span>
                    <span>r=32 (Recommended)</span>
                    <span>r=64 (Expressive)</span>
                  </div>
                </div>

                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950/70 border border-slate-800/80">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-medium">LoRA Alpha (α)</span>
                    <span className="font-mono text-indigo-400 font-semibold">α = {loraAlpha}</span>
                  </div>
                  <input
                    type="range"
                    min="16"
                    max="128"
                    step="16"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>Scaling = α / r = {(loraAlpha / loraRank).toFixed(2)}x</span>
                  </div>
                </div>
              </div>

              {/* Target Linear Projections */}
              <div className="flex flex-col gap-2.5">
                <span className="text-xs font-medium text-slate-300">
                  Targeted Linear Projections (Attention & Feed-Forward):
                </span>
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                  {[
                    { id: 'q_proj', label: 'Q-Proj', desc: 'Attention Query' },
                    { id: 'k_proj', label: 'K-Proj', desc: 'Attention Key' },
                    { id: 'v_proj', label: 'V-Proj', desc: 'Attention Value' },
                    { id: 'out_proj', label: 'Out-Proj', desc: 'Attention Out' },
                    { id: 'fc1', label: 'FFN Up (fc1)', desc: 'GELU Expansion' },
                    { id: 'fc2', label: 'FFN Down (fc2)', desc: 'Projection' },
                  ].map((mod) => {
                    const isChecked = targetModules.includes(mod.id);
                    return (
                      <button
                        key={mod.id}
                        onClick={() => toggleTargetModule(mod.id)}
                        className={`flex flex-col items-center justify-center p-2.5 rounded-lg border text-xs transition-all ${
                          isChecked
                            ? 'bg-indigo-600/20 border-indigo-500/50 text-indigo-200'
                            : 'bg-slate-950/60 border-slate-800/80 text-slate-500 hover:text-slate-300'
                        }`}
                      >
                        <span className="font-mono font-semibold">{mod.label}</span>
                        <span className="text-[9px] text-slate-500">{mod.desc}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* In-Place Weight Merge Action */}
              <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-white">Inference Adapter Status:</span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-mono font-semibold ${
                        isMerged
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                          : 'bg-amber-950 text-amber-400 border border-amber-800'
                      }`}
                    >
                      {isMerged ? 'MERGED (Zero Overhead)' : 'SEPARATE (Adapter Branch)'}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    {isMerged
                      ? 'LoRA ΔW = (α/r)·B·A folded in-place into base weights W₀. No extra matmuls or latency penalty during inference.'
                      : 'Dual forward branch y = W₀·x + (α/r)·B·A·x active. Allows dynamic weight switching or hot swapping.'}
                  </p>
                </div>

                <button
                  onClick={() => setIsMerged(!isMerged)}
                  className={`px-4 py-2 rounded-lg text-xs font-semibold whitespace-nowrap transition-all shadow-md ${
                    isMerged
                      ? 'bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700'
                      : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/40'
                  }`}
                >
                  {isMerged ? 'Unmerge Adapter' : 'Merge Weights In-Place'}
                </button>
              </div>
            </div>

            {/* Curated LoRA Presets Library */}
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">Curated LoRA Adapter Library</h3>
                </div>
                <span className="text-xs text-slate-500">PEFT Weights for Wan 1.3B</span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {LORA_PRESETS.map((preset) => {
                  const isSelected = activePreset.id === preset.id;
                  return (
                    <div
                      key={preset.id}
                      onClick={() => {
                        setActivePreset(preset);
                        setLoraRank(preset.rank);
                        setLoraAlpha(preset.alpha);
                      }}
                      className={`cursor-pointer p-4 rounded-xl border flex flex-col gap-2 transition-all ${
                        isSelected
                          ? 'bg-slate-800/90 border-indigo-500 shadow-md ring-1 ring-indigo-500/50'
                          : 'bg-slate-950/60 border-slate-800 hover:bg-slate-900/60'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-3.5 h-3.5 rounded-full border border-white/20"
                            style={{ background: preset.sampleThumbnail }}
                          />
                          <span className="text-xs font-semibold text-white">{preset.name}</span>
                        </div>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-indigo-300">
                          {preset.fileSizeMb} MB
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-2">{preset.description}</p>
                      <div className="mt-1 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                        <span>Rank: {preset.rank}</span>
                        <span>α: {preset.alpha}</span>
                        <span className="text-indigo-400">{preset.category}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Metrics & Parameter Budget Sidebar */}
          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <HardDrive className="w-4 h-4 text-indigo-400" />
                Parameter Budget & Efficiency
              </h3>

              <div className="flex flex-col gap-3">
                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Base Model Parameters</span>
                  <span className="text-base font-bold font-mono text-white">
                    {(loraStats.baseParams / 1e9).toFixed(2)} Billion
                  </span>
                  <span className="text-[10px] text-slate-500">Fully frozen (requires_grad = False)</span>
                </div>

                <div className="p-3 rounded-xl bg-indigo-950/30 border border-indigo-800/40 flex flex-col gap-1">
                  <span className="text-xs text-indigo-300">Trainable LoRA Parameters</span>
                  <span className="text-base font-bold font-mono text-indigo-400">
                    {(loraStats.loraParams / 1e6).toFixed(2)} Million
                  </span>
                  <div className="flex items-center justify-between text-[10px] text-slate-400">
                    <span>Trainable Ratio:</span>
                    <span className="font-semibold text-emerald-400 font-mono">
                      {loraStats.trainableRatioPercent}% of base
                    </span>
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/80 flex flex-col gap-1">
                  <span className="text-xs text-slate-400">Adapter Checkpoint Size</span>
                  <span className="text-base font-bold font-mono text-amber-400">
                    {loraStats.adapterSizeMb} MB
                  </span>
                  <span className="text-[10px] text-slate-500">vs ~2.8 GB for full FP16 checkpoint</span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-400 flex flex-col gap-2">
                <span className="font-semibold text-slate-200">Recommended Fine-Tuning Recipe:</span>
                <ul className="list-disc list-inside space-y-1 text-[11px] text-slate-400">
                  <li>Dataset: 500 to 2,000 captioned clips</li>
                  <li>Learning Rate: 1e-4 with AdamW</li>
                  <li>Resolution: 480x480 or 480x832</li>
                  <li>Schedule: 1,500 to 3,000 steps</li>
                  <li>Batch size: 1-2 per GPU with grad accum 4</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Image-to-Video (I2V) */}
      {activeTab === 'i2v' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Film className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">33-Channel Image-to-Video Architecture</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  Zero-Init Identity Verified
                </span>
              </div>

              {/* Channel Breakdown Visualizer */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-300">Input Latent Channel Allocation (33 Channels):</span>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div className="p-3 rounded-lg bg-indigo-950/40 border border-indigo-800/40 flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-indigo-300">Channels 0 .. 15</span>
                      <span className="font-mono text-[10px] text-indigo-400">16 Channels</span>
                    </div>
                    <span className="text-xs text-white font-medium">Noisy Latent Video (x_t)</span>
                    <p className="text-[10px] text-slate-400">
                      Standard flow matching noisy latents sampled from Gaussian prior.
                    </p>
                    <span className="text-[10px] font-mono text-emerald-400 mt-1">Pretrained Weights</span>
                  </div>

                  <div className="p-3 rounded-lg bg-amber-950/40 border border-amber-800/40 flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-amber-300">Channels 16 .. 31</span>
                      <span className="font-mono text-[10px] text-amber-400">16 Channels</span>
                    </div>
                    <span className="text-xs text-white font-medium">Conditioning Frame Latents</span>
                    <p className="text-[10px] text-slate-400">
                      Encoded VAE latents of the conditioning reference image.
                    </p>
                    <span className="text-[10px] font-mono text-amber-300 mt-1">Zero-Initialized (W=0)</span>
                  </div>

                  <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/40 flex flex-col gap-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-semibold text-rose-300">Channel 32</span>
                      <span className="font-mono text-[10px] text-rose-400">1 Channel</span>
                    </div>
                    <span className="text-xs text-white font-medium">Binary Conditioning Mask</span>
                    <p className="text-[10px] text-slate-400">
                      1.0 on conditioning frame(s), 0.0 on generated frames.
                    </p>
                    <span className="text-[10px] font-mono text-rose-300 mt-1">Zero-Initialized (W=0)</span>
                  </div>
                </div>
              </div>

              {/* Zero-Init Mathematical Guarantee */}
              <div className="p-4 rounded-xl bg-slate-950/60 border border-slate-800/80 flex flex-col gap-2 text-xs">
                <span className="font-semibold text-slate-200">The Zero-Init Mathematical Theorem:</span>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  When converting a T2V checkpoint to I2V, patch_embed weight tensor is expanded from <code className="text-indigo-300 font-mono">[D, 16, pt, ph, pw]</code> to <code className="text-indigo-300 font-mono">[D, 33, pt, ph, pw]</code>. By setting <code className="text-amber-300 font-mono">W[:, 16:33, ...] = 0.0</code>, the initial forward pass evaluates to:
                </p>
                <div className="p-2.5 rounded bg-slate-900 font-mono text-center text-xs text-emerald-400">
                  Conv3D(x_noisy || x_cond || mask; [W_t2v, 0]) = Conv3D(x_noisy; W_t2v) + 0.0 ≡ T2V(x_noisy)
                </div>
                <p className="text-slate-500 text-[10px]">
                  This preserves 100% of pretrained video motion and semantic generation capabilities at step 0 without cataclysmic forgetting.
                </p>
              </div>

              {/* Interactive Conditioning Settings */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <span className="text-xs font-medium text-slate-300">Conditioning Frame Anchor:</span>
                  <div className="grid grid-cols-3 gap-2">
                    {[
                      { id: 'first', label: 'First Frame (t=0)', desc: 'Standard I2V' },
                      { id: 'last', label: 'Last Frame (t=T)', desc: 'Reverse Video' },
                      { id: 'custom', label: 'First + Last', desc: 'Loop / Morph' },
                    ].map((opt) => (
                      <button
                        key={opt.id}
                        onClick={() => setI2vConditioningFrame(opt.id as any)}
                        className={`p-2 rounded-lg border text-xs text-center transition-all ${
                          i2vConditioningFrame === opt.id
                            ? 'bg-indigo-600/20 border-indigo-500 text-indigo-300 font-semibold'
                            : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex flex-col gap-2 p-4 rounded-xl bg-slate-950 border border-slate-800">
                  <div className="flex justify-between items-center text-xs">
                    <span className="text-slate-300 font-medium">Image Guidance Scale (CFG)</span>
                    <span className="font-mono text-indigo-400 font-semibold">{i2vImageGuidance.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="1.0"
                    max="4.0"
                    step="0.1"
                    value={i2vImageGuidance}
                    onChange={(e) => setI2vImageGuidance(Number(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>1.0 (Soft adherence)</span>
                    <span>2.0 (Recommended)</span>
                    <span>4.0 (Rigid frame locked)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                I2V Training Collator Rules
              </h3>
              <div className="flex flex-col gap-3 text-xs text-slate-400">
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-1">
                  <span className="font-semibold text-slate-200">1. Random Conditioning Probability</span>
                  <p className="text-[11px]">
                    Condition on frame 0 with p = 0.90; train unconditioned (T2V mode) with p = 0.10.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-1">
                  <span className="font-semibold text-slate-200">2. CFG Conditioning Dropout</span>
                  <p className="text-[11px]">
                    Drop conditioning frame latents to zero with p = 0.10 during conditioned batches to support dual-CFG guidance at inference.
                  </p>
                </div>
                <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-1">
                  <span className="font-semibold text-slate-200">3. VAE Latent Normalization</span>
                  <p className="text-[11px]">
                    Ensure conditioning image latents are scaled by the exact same 16-channel causal VAE standard deviation factors.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: Wan 2.2 Dual Experts */}
      {activeTab === 'dual_experts' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">Wan 2.2 Two-Expert Timestep Partitioning</h3>
                </div>
                <span className="text-xs font-mono text-indigo-400 bg-indigo-950 px-2 py-0.5 rounded border border-indigo-800">
                  Schedule Split, NOT Token-MoE
                </span>
              </div>

              {/* Boundary Slider */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">Timestep Boundary Threshold (τ)</span>
                  <span className="font-mono text-indigo-400 font-bold text-sm">τ = {expertBoundary.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.500"
                  max="0.950"
                  step="0.025"
                  value={expertBoundary}
                  onChange={(e) => setExpertBoundary(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>0.500 (Half schedule)</span>
                  <span className="text-emerald-400">0.875 (Official Wan 2.2 Default)</span>
                  <span>0.950 (Tail only)</span>
                </div>
              </div>

              {/* Timeline Graphic */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex justify-between text-xs text-slate-400">
                  <span>Inference Schedule Progress: t = 1.0 (Noise) → t = 0.0 (Clean Video)</span>
                  <span className="font-mono text-indigo-300">50 Steps Total</span>
                </div>

                {/* Split Bar */}
                <div className="h-9 w-full rounded-lg overflow-hidden flex font-mono text-[11px] font-semibold">
                  <div
                    style={{ width: `${(1.0 - expertBoundary) * 100}%` }}
                    className="bg-gradient-to-r from-amber-600 to-amber-500 text-slate-950 flex items-center justify-center px-2 transition-all"
                  >
                    High-Noise Expert ({highExpertStepCount} steps)
                  </div>
                  <div
                    style={{ width: `${expertBoundary * 100}%` }}
                    className="bg-gradient-to-r from-indigo-600 to-cyan-500 text-white flex items-center justify-center px-2 transition-all"
                  >
                    Low-Noise Expert ({lowExpertStepCount} steps)
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-xs mt-1">
                  <div className="p-3 rounded-lg bg-amber-950/30 border border-amber-800/40 flex flex-col gap-1">
                    <span className="font-semibold text-amber-300">High-Noise Expert Domain:</span>
                    <span className="text-[11px] text-slate-300">Timesteps t ∈ [{expertBoundary.toFixed(3)}, 1.000]</span>
                    <p className="text-[10px] text-slate-400">
                      Shapes global composition, semantic scene layout, large camera motions, and high-level dynamics.
                    </p>
                  </div>
                  <div className="p-3 rounded-lg bg-indigo-950/30 border border-indigo-800/40 flex flex-col gap-1">
                    <span className="font-semibold text-indigo-300">Low-Noise Expert Domain:</span>
                    <span className="text-[11px] text-slate-300">Timesteps t ∈ [0.000, {expertBoundary.toFixed(3)})</span>
                    <p className="text-[10px] text-slate-400">
                      Refines photorealistic micro-textures, crisp facial features, fine hair strands, and removes residual artifact fuzz.
                    </p>
                  </div>
                </div>
              </div>

              {/* Memory Offloading Simulator */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white">VRAM Offload & Swapping Mechanics</span>
                  <span className="text-[10px] font-mono text-emerald-400">14.8 GB Peak VRAM Cap</span>
                </div>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Rather than keeping both 14B models on GPU (which would require ~30 GB VRAM), the engine swaps them across system RAM and GPU VRAM at timestep boundary <code className="text-indigo-300">τ = {expertBoundary.toFixed(3)}</code>:
                </p>
                <div className="grid grid-cols-3 gap-2 text-center text-xs">
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">Step 0 → Step {highExpertStepCount}</span>
                    <span className="font-semibold text-amber-400">High-Noise on GPU</span>
                    <span className="text-[10px] text-slate-400 block">Low-Noise in CPU RAM</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-indigo-950/50 border border-indigo-800">
                    <span className="text-[10px] text-indigo-300 block">Boundary Crossing</span>
                    <span className="font-semibold text-white">Async Stream Swap</span>
                    <span className="text-[10px] text-indigo-400 block">~420ms Transfer</span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-500 block">Step {highExpertStepCount + 1} → 50</span>
                    <span className="font-semibold text-cyan-400">Low-Noise on GPU</span>
                    <span className="text-[10px] text-slate-400 block">High-Noise in CPU RAM</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Flame className="w-4 h-4 text-amber-400" />
                Training Loss Masking
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                During fine-tuning, each expert receives loss gradients exclusively on its assigned timestep interval:
              </p>
              <div className="p-3 rounded-lg bg-slate-950 font-mono text-[11px] text-slate-300 flex flex-col gap-2">
                <span className="text-amber-400"># High-Noise Expert Loss:</span>
                <code>mask = (t &gt;= {expertBoundary.toFixed(3)}).float()</code>
                <code>loss_high = (loss * mask).sum() / mask.sum()</code>
                <div className="h-px bg-slate-800 my-1" />
                <span className="text-indigo-400"># Low-Noise Expert Loss:</span>
                <code>mask = (t &lt; {expertBoundary.toFixed(3)}).float()</code>
                <code>loss_low = (loss * mask).sum() / mask.sum()</code>
              </div>
              <p className="text-[11px] text-slate-500">
                This enables both experts to specialize their attention heads and MLP capacities without compromising opposing frequencies.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: Few-Step Distillation */}
      {activeTab === 'distillation' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-white">4 to 8-Step Rectified Flow Reflow Distillation</h3>
                </div>
                <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2 py-0.5 rounded border border-emerald-800">
                  12.8x Speedup
                </span>
              </div>

              {/* Step Count Selector */}
              <div className="grid grid-cols-3 gap-3">
                {([4, 6, 8] as const).map((stepCount) => (
                  <button
                    key={stepCount}
                    onClick={() => setFewSteps(stepCount)}
                    className={`p-3.5 rounded-xl border flex flex-col items-center gap-1 transition-all ${
                      fewSteps === stepCount
                        ? 'bg-amber-600/20 border-amber-500 text-amber-300 font-semibold shadow-md'
                        : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span className="text-base font-mono font-bold">{stepCount} Steps</span>
                    <span className="text-[10px] text-slate-500">
                      {stepCount === 4 ? '1.9s (Ultra-Fast)' : stepCount === 6 ? '2.8s (Balanced)' : '3.6s (Near-Teacher)'}
                    </span>
                  </button>
                ))}
              </div>

              {/* 2-Rectified Flow Path Straightening Visualizer */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <span className="text-xs font-semibold text-slate-200">
                  2-Rectified Flow (Reflow) Trajectory Straightening:
                </span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3.5 rounded-lg bg-rose-950/20 border border-rose-800/30 flex flex-col gap-1.5">
                    <span className="text-xs font-semibold text-rose-300">1-Rectified Flow (Curved Paths)</span>
                    <p className="text-[11px] text-slate-400">
                      Standard flow matching models have slight path curvature due to finite-sample stochastic matching. Requires 30–50 Euler steps to integrate ODE accurately.
                    </p>
                    <div className="mt-2 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-rose-500 w-3/4 rounded-full" />
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-800/30 flex flex-col gap-1.5">
                    <span className="text-xs font-semibold text-emerald-300">2-Rectified Flow (Provably Straight)</span>
                    <p className="text-[11px] text-slate-400">
                      Synthetic trajectory endpoint pairs (x_1, x_hat_0) sampled from teacher make velocity v = x_1 - x_hat_0 constant along the line. Enables 4 Euler steps!
                    </p>
                    <div className="mt-2 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full bg-emerald-500 w-full rounded-full" />
                    </div>
                  </div>
                </div>
              </div>

              {/* Latency Comparison Table */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-2">
                <span className="text-xs font-semibold text-white">Inference Latency Benchmark (81 frames, 480x832):</span>
                <div className="grid grid-cols-4 gap-2 text-center text-xs pt-1">
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Baseline (50 steps)</span>
                    <span className="font-mono text-slate-300 font-semibold">24.5 s</span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Distilled (8 steps)</span>
                    <span className="font-mono text-cyan-400 font-semibold">3.6 s</span>
                  </div>
                  <div className="p-2 rounded bg-slate-900 border border-slate-800">
                    <span className="text-[10px] text-slate-400 block">Distilled (6 steps)</span>
                    <span className="font-mono text-indigo-400 font-semibold">2.8 s</span>
                  </div>
                  <div className="p-2 rounded bg-amber-950/50 border border-amber-800">
                    <span className="text-[10px] text-amber-300 block">Reflow (4 steps)</span>
                    <span className="font-mono text-amber-400 font-bold">1.9 s (12.8x)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Activity className="w-4 h-4 text-indigo-400" />
                Shifted Time Grid for Few-Step Flow
              </h3>
              <p className="text-xs text-slate-400">
                Few-step generation requires specialized non-linear time grid warping:
              </p>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 font-mono text-[11px] text-slate-300">
                <code>u = linspace(1.0, 0.0, {fewSteps + 1})</code>
                <code>sigma = (s * u) / (1.0 + (s - 1.0) * u)</code>
                <div className="mt-2 text-amber-400">
                  s = {distillShift.toFixed(1)} (Shift parameter)
                </div>
              </div>
              <div className="flex flex-col gap-1.5 text-xs text-slate-400">
                <span className="font-semibold text-slate-200">Resulting Sigma Steps:</span>
                {Array.from({ length: fewSteps + 1 }).map((_, idx) => {
                  const u = 1.0 - idx / fewSteps;
                  const s = (distillShift * u) / (1.0 + (distillShift - 1.0) * u);
                  return (
                    <div key={idx} className="flex justify-between items-center text-[10px] font-mono px-2 py-1 bg-slate-950 rounded">
                      <span className="text-slate-500">Step {idx}:</span>
                      <span className="text-indigo-300">σ = {s.toFixed(3)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 5: Long Video Chunking */}
      {activeTab === 'long_video' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Scissors className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-white">Long Video Temporal Chunking & Overlap</h3>
                </div>
                <span className="text-xs font-mono text-indigo-400 bg-indigo-950 px-2 py-0.5 rounded border border-indigo-800">
                  {longVideoPlan.totalPixelFrames} Frames ({(longVideoPlan.totalPixelFrames / 16).toFixed(1)}s at 16 FPS)
                </span>
              </div>

              {/* Target Length Slider */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-slate-300 font-medium">Desired Video Duration:</span>
                  <span className="font-mono text-indigo-400 font-bold text-sm">
                    {targetVideoFrames} Frames ({(targetVideoFrames / 16).toFixed(1)}s)
                  </span>
                </div>
                <input
                  type="range"
                  min="81"
                  max="401"
                  step="40"
                  value={targetVideoFrames}
                  onChange={(e) => setTargetVideoFrames(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>81 frames (5.1s)</span>
                  <span>161 frames (10.1s)</span>
                  <span>241 frames (15.1s)</span>
                  <span>401 frames (25.1s)</span>
                </div>
              </div>

              {/* Chunk Sequence Map */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-200">
                    Chunk Execution Plan ({longVideoPlan.numChunks} Sequential Chunks):
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Chunk Size: 81 f | Stride: 65 f | Overlap: 16 f
                  </span>
                </div>

                <div className="flex flex-col gap-2">
                  {longVideoPlan.chunks.map((chunk) => (
                    <div
                      key={chunk.index}
                      className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between text-xs"
                    >
                      <div className="flex items-center gap-3">
                        <span className="font-mono font-bold text-indigo-400">Chunk {chunk.index}</span>
                        <span className="text-slate-300 font-mono">
                          Frames [{chunk.startFrame} .. {chunk.endFrame}]
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {chunk.isConditioned ? (
                          <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950 text-amber-400 border border-amber-800">
                            I2V Autoregressive Conditioned
                          </span>
                        ) : (
                          <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-950 text-indigo-400 border border-indigo-800">
                            T2V Initial Seed Chunk
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Cosine Temporal Blending Mode */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-2">
                <span className="text-xs font-semibold text-white">Seamless Boundary Cross-Fade:</span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  In the 16-frame overlap zone between chunks, latents are blended with a smooth cosine function:
                </p>
                <div className="p-2.5 rounded bg-slate-900 font-mono text-center text-xs text-indigo-300">
                  w_fade_out(t) = cos²(π/2 · t / L) &nbsp;|&nbsp; w_fade_in(t) = sin²(π/2 · t / L) &nbsp;⇒&nbsp; Σ w = 1.000
                </div>
                <p className="text-[10px] text-slate-500">
                  Eliminates visual seam cuts, temporal jitter, and VAE chunk boundary flickering.
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                Long-Form Memory Footprint
              </h3>
              <div className="flex flex-col gap-3 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Peak VRAM per Chunk</span>
                  <span className="font-mono font-semibold text-emerald-400">14.8 GB (Unchanged!)</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Stitched Latents (RAM)</span>
                  <span className="font-mono font-semibold text-slate-300">
                    ~{(longVideoPlan.totalLatentFrames * 0.045).toFixed(1)} GB
                  </span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Total Generation Time</span>
                  <span className="font-mono font-semibold text-indigo-300">
                    {(longVideoPlan.numChunks * 24.5).toFixed(1)}s (50 steps)
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 6: Safety & C2PA Provenance */}
      {activeTab === 'safety' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-emerald-400" />
                  <h3 className="text-sm font-semibold text-white">NSFW Policy Filter & Moderation</h3>
                </div>
                <span
                  className={`text-xs font-mono px-2 py-0.5 rounded border ${
                    safetyResult.isSafe
                      ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                      : 'bg-rose-950 text-rose-400 border-rose-800'
                  }`}
                >
                  Status: {safetyResult.isSafe ? 'APPROVED' : 'REJECTED'}
                </span>
              </div>

              {/* Interactive Prompt Tester */}
              <div className="flex flex-col gap-2">
                <label className="text-xs font-medium text-slate-300">Test Input Prompt for Moderation Policy:</label>
                <textarea
                  rows={3}
                  value={testPrompt}
                  onChange={(e) => setTestPrompt(e.target.value)}
                  className="w-full p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 font-mono resize-none"
                  placeholder="Enter a prompt to test safety filters..."
                />
              </div>

              {/* Flagged Status */}
              {!safetyResult.isSafe && (
                <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/60 flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                  <div className="flex flex-col gap-1">
                    <span className="text-xs font-semibold text-rose-300">Prompt Violation Detected</span>
                    <p className="text-[11px] text-slate-300">
                      The prompt triggered the following restricted categories:{' '}
                      <span className="font-semibold text-rose-300 font-mono">
                        {safetyResult.categories.join(', ')}
                      </span>
                    </p>
                    <span className="text-[10px] text-slate-400 font-mono">
                      Matched terms: {safetyResult.flaggedWords.join(', ')}
                    </span>
                  </div>
                </div>
              )}

              {/* C2PA Provenance Credentials */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-white flex items-center gap-2">
                    <FileCheck className="w-4 h-4 text-indigo-400" />
                    C2PA Content Credentials Manifest (v2.1)
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">IPTC Compliant</span>
                </div>

                <div className="p-3 rounded-lg bg-slate-900 border border-slate-800/80 font-mono text-[10px] text-slate-300 overflow-x-auto">
                  <pre>{JSON.stringify(c2paManifest, null, 2)}</pre>
                </div>
              </div>

              {/* Invisible Forensic Watermark */}
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-2">
                <span className="text-xs font-semibold text-white">Invisible Frequency-Domain Watermark:</span>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Every exported video receives an invisible discrete cosine transform (DCT) watermark encoded with cryptographic key <code className="text-indigo-300 font-mono">{watermarkKey}</code>. Resistant to H.264 compression, cropping, and frame downsampling.
                </p>
                <div className="flex items-center justify-between text-xs mt-1 pt-2 border-t border-slate-800">
                  <span className="text-slate-400">Forensic Recovery Correlation Score:</span>
                  <span className="font-mono font-bold text-emerald-400">0.96 / 1.00 (High Confidence)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-6">
            <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-indigo-400" />
                Video Output Frame Checker
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Post-generation safety checks scan decoded pixel frames prior to MP4 encoding:
              </p>
              <div className="flex flex-col gap-2 text-xs">
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Excessive Skin Tone Threshold</span>
                  <span className="font-mono text-slate-200">&lt; 45%</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Temporal Flash & Strobe Safety</span>
                  <span className="font-mono text-emerald-400">Compliant (Harding FPA)</span>
                </div>
                <div className="p-3 rounded-lg bg-slate-950 border border-slate-800 flex justify-between items-center">
                  <span className="text-slate-400">Resolution & Aspect Contract</span>
                  <span className="font-mono text-slate-200">Strict 4k+1 temporal</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 7: Milestones Tracker (M1 through M5) */}
      {activeTab === 'milestones' && (
        <div className="flex flex-col gap-6">
          <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-semibold text-white">Build Order & Master Milestones (M1–M5)</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Step-by-step verification milestones for the complete video generation engine.
                </p>
              </div>
              <span className="text-xs font-mono text-emerald-400 bg-emerald-950 px-2.5 py-1 rounded-full border border-emerald-800">
                Core Foundation Complete
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-2">
              {MILESTONES.map((item) => (
                <div
                  key={item.id}
                  className="p-5 rounded-xl bg-slate-950/80 border border-slate-800/90 flex flex-col justify-between gap-3 shadow-lg"
                >
                  <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
                        {item.milestone}
                      </span>
                      <span
                        className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                          item.status === 'COMPLETED'
                            ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                            : 'bg-indigo-950 text-indigo-300 border border-indigo-800'
                        }`}
                      >
                        {item.status.replace('_', ' ')}
                      </span>
                    </div>
                    <h4 className="text-sm font-bold text-white">{item.title}</h4>
                    <p className="text-xs text-slate-400 leading-relaxed">{item.description}</p>
                  </div>

                  <div className="pt-3 border-t border-slate-800/80 flex flex-col gap-1.5 text-[11px]">
                    <div className="flex justify-between items-center text-slate-400">
                      <span>Target Hardware:</span>
                      <span className="font-mono text-slate-300">{item.targetHardware}</span>
                    </div>
                    <div className="flex justify-between items-center text-slate-400">
                      <span>Metric Target:</span>
                      <span className="font-mono text-emerald-400 text-right">{item.metrics}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
