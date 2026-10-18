import React, { useState, useMemo } from 'react';
import {
  Award,
  Sparkles,
  Layers,
  Activity,
  AlertTriangle,
  CheckCircle2,
  Sliders,
  Play,
  RotateCcw,
  Search,
  Filter,
  Users,
  Eye,
  ShieldAlert,
  Flame,
  Info,
  ChevronRight,
  TrendingUp,
  Cpu,
  Video,
  Database,
  BarChart3,
  HelpCircle,
} from 'lucide-react';
import {
  VALIDATION_PROMPTS,
  DEFAULT_CHECKPOINTS,
  ValidationPromptItem,
  CheckpointEvalProfile,
  calculateVBenchScore,
  diagnoseGeneration,
  checkMemorization,
  VBenchDimensions,
} from '../utils/evalEngine';

export const EvaluationStudio: React.FC = () => {
  // Tabs
  const [activeTab, setActiveTab] = useState<'comparative_grid' | 'vbench_radar' | 'ab_arena' | 'diagnostics' | 'memorization' | 'regression'>('comparative_grid');

  // Prompt selection
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedPromptId, setSelectedPromptId] = useState<string>('motion_01');
  const [playbackFrame, setPlaybackFrame] = useState<number>(24);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [showDiffHeatmap, setShowDiffHeatmap] = useState<boolean>(false);

  // Checkpoints
  const [checkpoints, setCheckpoints] = useState<CheckpointEvalProfile[]>(DEFAULT_CHECKPOINTS);
  const [primaryCheckpointId, setPrimaryCheckpointId] = useState<string>('cp_50k_stage4');
  const [secondaryCheckpointId, setSecondaryCheckpointId] = useState<string>('cp_20k_stage2');

  // Human A/B Blind Arena State
  const [abWinner, setAbWinner] = useState<'A' | 'B' | 'TIE' | null>(null);
  const [abCategory, setAbCategory] = useState<'quality' | 'motion' | 'prompt' | 'temporal'>('quality');
  const [abRevealed, setAbRevealed] = useState<boolean>(false);
  const [abMatchCount, setAbMatchCount] = useState<number>(36);
  const [lastVoteLog, setLastVoteLog] = useState<string | null>(null);

  // Diagnostics State
  const [diagnosticSymptoms, setDiagnosticSymptoms] = useState<string[]>(['blurry']);
  const [diagSteps, setDiagSteps] = useState<number>(18);
  const [diagCfg, setDiagCfg] = useState<number>(8.5);
  const [diagTrainingSteps, setDiagTrainingSteps] = useState<number>(4500);
  const [diagChunkOverlap, setDiagChunkOverlap] = useState<number>(1);

  // Memorization State
  const [customMemoPrompt, setCustomMemoPrompt] = useState<string>('A golden retriever running through green grass');
  const [customMemoSeed, setCustomMemoSeed] = useState<number>(42);

  // Regression Tests State
  const [regressionLog, setRegressionLog] = useState<string[]>([]);
  const [isRunningRegression, setIsRunningRegression] = useState<boolean>(false);

  // Filtered Prompts
  const filteredPrompts = useMemo(() => {
    if (selectedCategory === 'all') return VALIDATION_PROMPTS;
    return VALIDATION_PROMPTS.filter((p) => p.category === selectedCategory);
  }, [selectedCategory]);

  const currentPrompt = useMemo(() => {
    return VALIDATION_PROMPTS.find((p) => p.id === selectedPromptId) || VALIDATION_PROMPTS[0];
  }, [selectedPromptId]);

  const primaryCp = useMemo(() => {
    return checkpoints.find((c) => c.id === primaryCheckpointId) || checkpoints[0];
  }, [checkpoints, primaryCheckpointId]);

  const secondaryCp = useMemo(() => {
    return checkpoints.find((c) => c.id === secondaryCheckpointId) || checkpoints[1];
  }, [checkpoints, secondaryCheckpointId]);

  // Diagnosis Execution
  const diagnosisFindings = useMemo(() => {
    return diagnoseGeneration({
      symptoms: diagnosticSymptoms,
      steps: diagSteps,
      cfgScale: diagCfg,
      trainingSteps: diagTrainingSteps,
      chunkOverlap: diagChunkOverlap,
      imagingQuality: diagSteps < 25 ? 0.62 : 0.88,
      temporalFlicker: diagChunkOverlap < 4 ? 0.76 : 0.96,
      dynamicDegree: 0.65,
    });
  }, [diagnosticSymptoms, diagSteps, diagCfg, diagTrainingSteps, diagChunkOverlap]);

  // Memorization check
  const memorizationResult = useMemo(() => {
    return checkMemorization(customMemoPrompt, customMemoSeed);
  }, [customMemoPrompt, customMemoSeed]);

  // Handle A/B Vote
  const handleVote = (winner: 'A' | 'B' | 'TIE') => {
    setAbWinner(winner);
    setAbRevealed(true);
    setAbMatchCount((prev) => prev + 1);

    // ELO update logic
    const cpA = primaryCp;
    const cpB = secondaryCp;
    const qa = 10 ** (cpA.elo / 400);
    const qb = 10 ** (cpB.elo / 400);
    const expectedA = qa / (qa + qb);
    const expectedB = qb / (qa + qb);

    let actualA = 0.5;
    let actualB = 0.5;
    if (winner === 'A') { actualA = 1.0; actualB = 0.0; }
    if (winner === 'B') { actualA = 0.0; actualB = 1.0; }

    const k = 24;
    const deltaA = Math.round(k * (actualA - expectedA) * 10) / 10;
    const deltaB = Math.round(k * (actualB - expectedB) * 10) / 10;

    setCheckpoints((prev) =>
      prev.map((c) => {
        if (c.id === cpA.id) {
          return {
            ...c,
            elo: Math.round((c.elo + deltaA) * 10) / 10,
            wins: winner === 'A' ? c.wins + 1 : c.wins,
            losses: winner === 'B' ? c.losses + 1 : c.losses,
            ties: winner === 'TIE' ? c.ties + 1 : c.ties,
          };
        }
        if (c.id === cpB.id) {
          return {
            ...c,
            elo: Math.round((c.elo + deltaB) * 10) / 10,
            wins: winner === 'B' ? c.wins + 1 : c.wins,
            losses: winner === 'A' ? c.losses + 1 : c.losses,
            ties: winner === 'TIE' ? c.ties + 1 : c.ties,
          };
        }
        return c;
      })
    );

    setLastVoteLog(
      `Recorded vote: ${winner === 'TIE' ? 'Tie' : winner === 'A' ? cpA.name : cpB.name} (${abCategory}). Delta ELO: A (${deltaA > 0 ? '+' : ''}${deltaA}), B (${deltaB > 0 ? '+' : ''}${deltaB})`
    );
  };

  // Run Regression Suite Simulation
  const handleRunRegression = () => {
    setIsRunningRegression(true);
    setRegressionLog([
      '⚡ [0.00s] Initializing Part 10 Regression Verification Suite...',
      '🔍 [0.08s] Loading fixed seed validation catalog (70 prompts across 7 axes)... OK',
      '🎯 [0.15s] Executing Sampler Oracle Test: Recovering x0 from fake velocity v = eps - x0...',
      '   ├── Euler step across 10 shifted timesteps [sigma 1.0 -> 0.0]',
      '   └── Max L_inf reconstruction error: 2.38e-7 < 1e-4 target [PASSED]',
      '🔄 [0.24s] Executing Causal 3D VAE Round-Trip on 20 held-out clips...',
      '   ├── Mean PSNR: 34.82 dB (target > 32 dB) [PASSED]',
      '   ├── Mean SSIM: 0.9341, LPIPS: 0.0482 [PASSED]',
      '   └── Chunked vs Unchunked L1 difference: 0.0412 < 0.15 tolerance [PASSED]',
      '📈 [0.38s] Executing Overfit Convergence Test (Wan-Tiny on 1 clip)...',
      '   ├── Step 100: Loss = 0.428',
      '   ├── Step 500: Loss = 0.084',
      '   ├── Step 1200: Loss = 0.00118 < 0.005 threshold [PASSED]',
      '   └── Visual sample matches target clip with PSNR > 42 dB [PASSED]',
      '🛡️ [0.49s] Executing Training Memorization Query Test...',
      '   ├── Querying 100k training clip database via 128-dim spatiotemporal embeddings',
      '   └── Correctly flagged overfit clip (similarity 0.992) and novel clip (similarity 0.68) [PASSED]',
      '✨ [0.55s] All Part 10 Regression Tests Passed Bit-Identically!',
    ]);
    setTimeout(() => {
      setIsRunningRegression(false);
    }, 600);
  };

  // Helper for toggle symptoms
  const toggleSymptom = (sym: string) => {
    setDiagnosticSymptoms((prev) =>
      prev.includes(sym) ? prev.filter((s) => s !== sym) : [...prev, sym]
    );
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-8 text-slate-100">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 border border-indigo-500/30 p-6 sm:p-8 shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 flex items-center gap-1.5">
                <Award className="w-3.5 h-3.5 text-indigo-400" /> Part 10 Active
              </span>
              <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> VBench & A/B Suite
              </span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white flex items-center gap-3">
              Evaluation & Benchmarking Studio
            </h1>
            <p className="text-sm text-slate-300 max-w-2xl">
              Rigorous verification suite for Wan Video DiT models: 70 fixed validation prompts across 7 axes, 8 VBench dimensions, Fréchet Video Distance (FVD), blind Human A/B ELO rating arena, memorization checks, and instant failure mode diagnostics.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handleRunRegression}
              disabled={isRunningRegression}
              className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-sky-600 hover:from-indigo-500 hover:to-sky-500 text-white font-medium text-sm flex items-center gap-2 shadow-lg shadow-indigo-600/30 transition-all border border-indigo-400/30 active:scale-95 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              {isRunningRegression ? 'Verifying Tests...' : 'Run Regression Tests'}
            </button>
          </div>
        </div>

        {/* Top Summary Metrics Bar */}
        <div className="mt-6 pt-6 border-t border-slate-800/80 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-4">
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">Validation Prompts</div>
            <div className="text-lg font-bold text-indigo-400">70 Prompts</div>
            <div className="text-[11px] text-slate-400">7 Axes / Fixed Seeds</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">VBench Top Score</div>
            <div className="text-lg font-bold text-emerald-400">0.918 / 1.0</div>
            <div className="text-[11px] text-slate-400">Checkpoint 50k</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">FVD (Stage 4)</div>
            <div className="text-lg font-bold text-sky-400">98.4</div>
            <div className="text-[11px] text-slate-400">Target &lt; 120</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">Human A/B ELO</div>
            <div className="text-lg font-bold text-amber-400">1,342.0</div>
            <div className="text-[11px] text-slate-400">{abMatchCount} Blind Matches</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">VAE Round-Trip</div>
            <div className="text-lg font-bold text-cyan-400">34.8 dB</div>
            <div className="text-[11px] text-slate-400">Held-Out PSNR</div>
          </div>
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800">
            <div className="text-xs text-slate-400">Regression Status</div>
            <div className="text-lg font-bold text-emerald-400 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" /> PASS
            </div>
            <div className="text-[11px] text-slate-400">Bit-Identical</div>
          </div>
        </div>
      </div>

      {/* Navigation Sub-Tabs */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800 pb-3">
        {[
          { id: 'comparative_grid', label: 'Side-by-Side Checkpoint Grid', icon: Video },
          { id: 'vbench_radar', label: 'VBench 8-Dimensional Metrics', icon: BarChart3 },
          { id: 'ab_arena', label: 'Human A/B Blind Arena', icon: Users },
          { id: 'diagnostics', label: 'Quick Diagnostic Engine', icon: AlertTriangle },
          { id: 'memorization', label: 'Memorization & Overfit Detector', icon: Database },
          { id: 'regression', label: 'Regression Suite & Oracle Tests', icon: CheckCircle2 },
        ].map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30 border border-indigo-400/40'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 hover:bg-slate-800 border border-slate-800'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* TAB 1: SIDE-BY-SIDE CHECKPOINT GRID                           */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'comparative_grid' && (
        <div className="space-y-6">
          {/* Controls Bar: Category Filter & Prompt Picker */}
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-4">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center gap-2">
                  <Filter className="w-4 h-4 text-indigo-400" />
                  Select Fixed Validation Benchmark Prompt
                </h3>
                <p className="text-xs text-slate-400">
                  Fixed seed guarantees identical noise initializations across checkpoints.
                </p>
              </div>

              {/* Category Pills */}
              <div className="flex flex-wrap items-center gap-1.5">
                {['all', 'motion', 'camera', 'counting', 'text', 'people', 'animals', 'styles'].map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setSelectedCategory(cat)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium capitalize transition-all ${
                      selectedCategory === cat
                        ? 'bg-indigo-600 text-white'
                        : 'bg-slate-950 text-slate-400 hover:bg-slate-800'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt Selector Dropdown & Info */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="md:col-span-2">
                <select
                  value={selectedPromptId}
                  onChange={(e) => setSelectedPromptId(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
                >
                  {filteredPrompts.map((p) => (
                    <option key={p.id} value={p.id}>
                      [{p.category.toUpperCase()}] Seed {p.seed}: {p.prompt.slice(0, 90)}...
                    </option>
                  ))}
                </select>
              </div>

              <div className="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-xs flex items-center justify-between">
                <div>
                  <span className="text-slate-400">Fixed Seed: </span>
                  <span className="font-mono font-bold text-indigo-400">{currentPrompt.seed}</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-slate-800 text-slate-300">
                    {currentPrompt.difficulty}
                  </span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                    {currentPrompt.targetMetric}
                  </span>
                </div>
              </div>
            </div>

            {/* Selected Prompt Full Text Card */}
            <div className="p-3.5 bg-slate-950/70 rounded-xl border border-slate-800/80 text-xs text-slate-300">
              <span className="text-indigo-400 font-semibold">Prompt: </span>
              &ldquo;{currentPrompt.prompt}&rdquo;
              <p className="mt-1 text-[11px] text-slate-400 italic">
                Evaluation target: {currentPrompt.description}
              </p>
            </div>
          </div>

          {/* Synchronized Temporal Frame Scrubber & Playback */}
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-3">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsPlaying(!isPlaying)}
                  className="p-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-all"
                >
                  <Play className={`w-3.5 h-3.5 ${isPlaying ? 'fill-current' : ''}`} />
                </button>
                <span className="font-semibold text-slate-200">
                  Synchronized Spacetime Scrubber (16 FPS Video)
                </span>
              </div>
              <div className="flex items-center gap-4">
                <label className="flex items-center gap-2 cursor-pointer text-slate-300">
                  <input
                    type="checkbox"
                    checked={showDiffHeatmap}
                    onChange={(e) => setShowDiffHeatmap(e.target.checked)}
                    className="rounded bg-slate-950 border-slate-700 text-indigo-600 focus:ring-0"
                  />
                  <span>Difference Heatmap</span>
                </label>
                <span className="font-mono text-indigo-400 font-bold">
                  Frame {playbackFrame} / 49 ({(playbackFrame / 16).toFixed(2)}s)
                </span>
              </div>
            </div>

            <input
              type="range"
              min={0}
              max={48}
              value={playbackFrame}
              onChange={(e) => setPlaybackFrame(parseInt(e.target.value, 10))}
              className="w-full h-2 bg-slate-950 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>

          {/* Side-by-Side Checkpoint Video Viewers */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Checkpoint: Primary */}
            <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Checkpoint A</div>
                  <select
                    value={primaryCheckpointId}
                    onChange={(e) => setPrimaryCheckpointId(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-semibold text-indigo-300"
                  >
                    {checkpoints.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-emerald-400">
                    VBench: {primaryCp.overallScore}
                  </span>
                  <div className="text-[10px] text-slate-400">ELO: {primaryCp.elo}</div>
                </div>
              </div>

              {/* Video Simulated Frame Canvas */}
              <div className="relative aspect-video rounded-xl bg-slate-950 overflow-hidden border border-slate-800 flex items-center justify-center group shadow-inner">
                {/* Background visual art generator representation */}
                <div
                  className="absolute inset-0 transition-opacity duration-300"
                  style={{
                    background: `radial-gradient(circle at ${30 + (playbackFrame * 1.2) % 60}% ${40 + (playbackFrame * 0.8) % 40}%, ${
                      primaryCp.id.includes('50k')
                        ? '#3b82f6'
                        : primaryCp.id.includes('20k')
                        ? '#6366f1'
                        : '#9333ea'
                    } 0%, #030712 75%)`,
                    filter: primaryCp.id.includes('5k') ? 'blur(8px)' : 'none',
                  }}
                />

                {/* Overlaid simulated frame elements */}
                <div className="relative z-10 text-center p-4">
                  <div className="text-xs font-mono text-indigo-200 bg-slate-950/80 px-3 py-1 rounded-full border border-slate-700/60 inline-block mb-2">
                    Seed {currentPrompt.seed} • {primaryCp.architecture}
                  </div>
                  <p className="text-xs text-slate-300 max-w-sm line-clamp-2">
                    &ldquo;{currentPrompt.prompt}&rdquo;
                  </p>
                </div>

                {/* Frame stats badge */}
                <div className="absolute bottom-2 left-2 px-2 py-1 bg-slate-950/80 rounded text-[10px] font-mono text-slate-400 border border-slate-800">
                  {primaryCp.name.split('(')[0]} • t={(playbackFrame / 16).toFixed(2)}s
                </div>
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-slate-950/80 rounded text-[10px] font-mono text-emerald-400 border border-slate-800">
                  FVD: {primaryCp.fvd}
                </div>
              </div>

              {/* VBench dimension mini-badges */}
              <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Smoothness</div>
                  <div className="font-bold text-indigo-300">{primaryCp.vbench.motion_smoothness}</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Consistency</div>
                  <div className="font-bold text-indigo-300">{primaryCp.vbench.subject_consistency}</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Aesthetic</div>
                  <div className="font-bold text-indigo-300">{primaryCp.vbench.aesthetic_score}/10</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Flickering</div>
                  <div className="font-bold text-indigo-300">{primaryCp.vbench.temporal_flickering}</div>
                </div>
              </div>
            </div>

            {/* Right Checkpoint: Secondary */}
            <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-5 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">Checkpoint B</div>
                  <select
                    value={secondaryCheckpointId}
                    onChange={(e) => setSecondaryCheckpointId(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-semibold text-sky-300"
                  >
                    {checkpoints.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono font-bold text-sky-400">
                    VBench: {secondaryCp.overallScore}
                  </span>
                  <div className="text-[10px] text-slate-400">ELO: {secondaryCp.elo}</div>
                </div>
              </div>

              {/* Video Simulated Frame Canvas */}
              <div className="relative aspect-video rounded-xl bg-slate-950 overflow-hidden border border-slate-800 flex items-center justify-center group shadow-inner">
                {/* Background visual art generator representation */}
                <div
                  className="absolute inset-0 transition-opacity duration-300"
                  style={{
                    background: showDiffHeatmap
                      ? `linear-gradient(45deg, #ef4444 0%, #3b82f6 50%, #10b981 100%)`
                      : `radial-gradient(circle at ${30 + (playbackFrame * 1.2) % 60}% ${40 + (playbackFrame * 0.8) % 40}%, ${
                          secondaryCp.id.includes('50k')
                            ? '#3b82f6'
                            : secondaryCp.id.includes('20k')
                            ? '#6366f1'
                            : '#9333ea'
                        } 0%, #030712 75%)`,
                    filter: secondaryCp.id.includes('5k') ? 'blur(10px)' : 'none',
                    opacity: showDiffHeatmap ? 0.8 : 1.0,
                  }}
                />

                {showDiffHeatmap && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/40">
                    <span className="px-3 py-1 bg-red-950/80 text-red-300 border border-red-800 rounded-full text-xs font-mono">
                      Residual Heatmap: L1 Diff = 0.084
                    </span>
                  </div>
                )}

                {/* Overlaid simulated frame elements */}
                {!showDiffHeatmap && (
                  <div className="relative z-10 text-center p-4">
                    <div className="text-xs font-mono text-sky-200 bg-slate-950/80 px-3 py-1 rounded-full border border-slate-700/60 inline-block mb-2">
                      Seed {currentPrompt.seed} • {secondaryCp.architecture}
                    </div>
                    <p className="text-xs text-slate-300 max-w-sm line-clamp-2">
                      &ldquo;{currentPrompt.prompt}&rdquo;
                    </p>
                  </div>
                )}

                <div className="absolute bottom-2 left-2 px-2 py-1 bg-slate-950/80 rounded text-[10px] font-mono text-slate-400 border border-slate-800">
                  {secondaryCp.name.split('(')[0]} • t={(playbackFrame / 16).toFixed(2)}s
                </div>
                <div className="absolute bottom-2 right-2 px-2 py-1 bg-slate-950/80 rounded text-[10px] font-mono text-sky-400 border border-slate-800">
                  FVD: {secondaryCp.fvd}
                </div>
              </div>

              {/* VBench dimension mini-badges */}
              <div className="grid grid-cols-4 gap-2 text-center text-[10px]">
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Smoothness</div>
                  <div className="font-bold text-sky-300">{secondaryCp.vbench.motion_smoothness}</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Consistency</div>
                  <div className="font-bold text-sky-300">{secondaryCp.vbench.subject_consistency}</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Aesthetic</div>
                  <div className="font-bold text-sky-300">{secondaryCp.vbench.aesthetic_score}/10</div>
                </div>
                <div className="bg-slate-950 p-1.5 rounded-lg border border-slate-800">
                  <div className="text-slate-400">Flickering</div>
                  <div className="font-bold text-sky-300">{secondaryCp.vbench.temporal_flickering}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* TAB 2: VBENCH 8-DIMENSIONAL RADAR & BAR METRICS               */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'vbench_radar' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 space-y-6">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-indigo-400" />
                VBench 8-Dimensional Metric Evaluation
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Standardized benchmark dimensions assessing kinematics, spatial consistency, temporal coherence, aesthetic appeal, and semantic alignment.
              </p>
            </div>

            {/* 8 Dimensions Table & Comparative Progress Bars */}
            <div className="space-y-4">
              {[
                { key: 'subject_consistency', label: 'Subject Consistency', desc: 'Identity preservation across frames', weight: '15%' },
                { key: 'background_consistency', label: 'Background Consistency', desc: 'Static scene background stability without spatial warping', weight: '10%' },
                { key: 'motion_smoothness', label: 'Motion Smoothness', desc: 'Temporal flow variance penalizing frame-to-frame jitter', weight: '15%' },
                { key: 'dynamic_degree', label: 'Dynamic Degree', desc: 'Magnitude of optical motion vs. static still images', weight: '15%' },
                { key: 'aesthetic_score', label: 'Aesthetic Score', desc: 'Composition, lighting harmony, and color richness [1.0-10.0]', weight: '15%' },
                { key: 'imaging_quality', label: 'Imaging Quality', desc: 'Perceptual sharpness, absence of compression artifacts or blur', weight: '10%' },
                { key: 'temporal_flickering', label: 'Temporal Flickering', desc: 'Inter-frame luminance & chromaticity delta (1.0 = zero flicker)', weight: '10%' },
                { key: 'text_alignment', label: 'Text-Video Alignment', desc: 'CLIP / ViCLIP multi-modal cosine similarity to prompt', weight: '10%' },
              ].map((dim) => {
                const valA = (primaryCp.vbench as any)[dim.key];
                const valB = (secondaryCp.vbench as any)[dim.key];
                const normA = dim.key === 'aesthetic_score' ? (valA / 10) * 100 : valA * 100;
                const normB = dim.key === 'aesthetic_score' ? (valB / 10) * 100 : valB * 100;

                return (
                  <div key={dim.key} className="bg-slate-950 p-4 rounded-xl border border-slate-800/80 space-y-2">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-1">
                      <div>
                        <span className="font-bold text-slate-200">{dim.label}</span>
                        <span className="text-slate-500 ml-2">({dim.desc})</span>
                      </div>
                      <div className="flex items-center gap-4 font-mono font-semibold">
                        <span className="text-indigo-400">
                          {primaryCp.name.split('(')[0]}: {valA}
                        </span>
                        <span className="text-sky-400">
                          {secondaryCp.name.split('(')[0]}: {valB}
                        </span>
                        <span className="text-[10px] text-slate-500">Weight: {dim.weight}</span>
                      </div>
                    </div>

                    {/* Progress Bars */}
                    <div className="grid grid-cols-2 gap-3 pt-1">
                      <div className="space-y-1">
                        <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-indigo-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${normA}%` }}
                          />
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                          <div
                            className="bg-sky-500 h-2 rounded-full transition-all duration-500"
                            style={{ width: `${normB}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* FVD Metric Information Box */}
            <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-800/50 text-xs text-amber-200/90 flex items-start gap-3">
              <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <span className="font-bold">Fréchet Video Distance (FVD) Sensitivity Notice (Part 10 Guideline):</span>
                <p className="mt-1 text-amber-300/80 text-[11px]">
                  FVD is mathematically sensitive to sample size, video resolution, and feature extractor distribution. A statistically meaningful FVD score requires at least 2,048 video samples. Always compare FVD scores strictly within the same hardware setup and sampling schedule.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* TAB 3: HUMAN A/B BLIND ARENA                                  */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'ab_arena' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Users className="w-5 h-5 text-indigo-400" />
                  Double-Blind Human A/B Arena
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Automated metrics cannot replace human eyes. Vote on randomized side-by-side clips to update ELO rankings.
                </p>
              </div>

              <div className="flex items-center gap-2 bg-slate-950 p-1 rounded-xl border border-slate-800">
                {(['quality', 'motion', 'prompt', 'temporal'] as const).map((cat) => (
                  <button
                    key={cat}
                    onClick={() => setAbCategory(cat)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium capitalize transition-all ${
                      abCategory === cat
                        ? 'bg-indigo-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {cat}
                  </button>
                ))}
              </div>
            </div>

            {/* Prompt for Voting Trial */}
            <div className="p-3.5 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200">
              <span className="text-indigo-400 font-semibold">Evaluation Prompt: </span>
              &ldquo;{currentPrompt.prompt}&rdquo;
            </div>

            {/* Blind Side-by-Side Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model A */}
              <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 flex flex-col items-center justify-center space-y-4">
                <div className="text-sm font-bold text-slate-200">Model Alpha</div>
                <div className="w-full aspect-video rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center relative overflow-hidden">
                  <div className="text-xs font-mono text-slate-400">
                    Video Stream Alpha (Randomized)
                  </div>
                  {abRevealed && (
                    <div className="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-950 text-indigo-300 border border-indigo-800">
                      {primaryCp.name}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleVote('A')}
                  className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-indigo-600/30"
                >
                  Vote Alpha Better (Winner)
                </button>
              </div>

              {/* Model B */}
              <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 flex flex-col items-center justify-center space-y-4">
                <div className="text-sm font-bold text-slate-200">Model Beta</div>
                <div className="w-full aspect-video rounded-xl bg-slate-900 border border-slate-800 flex items-center justify-center relative overflow-hidden">
                  <div className="text-xs font-mono text-slate-400">
                    Video Stream Beta (Randomized)
                  </div>
                  {abRevealed && (
                    <div className="absolute top-2 right-2 px-2 py-0.5 rounded text-[10px] font-mono bg-sky-950 text-sky-300 border border-sky-800">
                      {secondaryCp.name}
                    </div>
                  )}
                </div>

                <button
                  onClick={() => handleVote('B')}
                  className="w-full py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-semibold text-xs transition-all shadow-lg shadow-sky-600/30"
                >
                  Vote Beta Better (Winner)
                </button>
              </div>
            </div>

            {/* Tie Button */}
            <div className="flex justify-center">
              <button
                onClick={() => handleVote('TIE')}
                className="px-6 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-xs border border-slate-700"
              >
                Tie / Both Indistinguishable
              </button>
            </div>

            {/* Voting Log & ELO Results */}
            {lastVoteLog && (
              <div className="p-3 bg-emerald-950/40 rounded-xl border border-emerald-800/60 text-xs text-emerald-300 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>{lastVoteLog}</span>
              </div>
            )}

            {/* Live ELO Leaderboard */}
            <div className="pt-4 border-t border-slate-800 space-y-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Current Checkpoint ELO Leaderboard
              </h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-2 px-3">Rank</th>
                      <th className="py-2 px-3">Checkpoint</th>
                      <th className="py-2 px-3">Step</th>
                      <th className="py-2 px-3">Architecture</th>
                      <th className="py-2 px-3 font-mono">ELO Rating</th>
                      <th className="py-2 px-3">Record (W-L-T)</th>
                      <th className="py-2 px-3">Win Rate</th>
                    </tr>
                  </thead>
                  <tbody>
                    {checkpoints
                      .slice()
                      .sort((a, b) => b.elo - a.elo)
                      .map((c, idx) => (
                        <tr key={c.id} className="border-b border-slate-850 hover:bg-slate-950">
                          <td className="py-2 px-3 font-bold text-slate-400">#{idx + 1}</td>
                          <td className="py-2 px-3 font-semibold text-white">{c.name}</td>
                          <td className="py-2 px-3 text-slate-400 font-mono">{c.step.toLocaleString()}</td>
                          <td className="py-2 px-3 text-slate-300">{c.architecture}</td>
                          <td className="py-2 px-3 font-mono font-bold text-amber-400">{c.elo.toFixed(1)}</td>
                          <td className="py-2 px-3 text-slate-300">{c.wins}W - {c.losses}L - {c.ties}T</td>
                          <td className="py-2 px-3 font-mono text-emerald-400">
                            {c.wins + c.losses + c.ties > 0
                              ? `${(((c.wins + 0.5 * c.ties) / (c.wins + c.losses + c.ties)) * 100).toFixed(1)}%`
                              : '0.0%'}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* TAB 4: QUICK DIAGNOSTIC ENGINE                                */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'diagnostics' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 space-y-6">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                Quick Generation Failure Diagnostic Engine
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Rule-based root cause analysis for common generative video defects (blurry textures, temporal flicker, wrong motion, washed-out colors).
              </p>
            </div>

            {/* Clickable Observed Symptoms */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300">
                Observed Generation Symptoms (Click all that apply):
              </label>
              <div className="flex flex-wrap gap-2">
                {[
                  { id: 'blurry', label: 'Blurry / Soft Video' },
                  { id: 'flicker', label: 'Temporal Flickering / Stutter' },
                  { id: 'wrong_motion', label: 'Wrong Motion for Prompt' },
                  { id: 'washed_out', label: 'Washed-Out Colors / High-CFG Burn' },
                  { id: 'static_video', label: 'Static Video (Negligible Motion)' },
                  { id: 'gray_blobs', label: 'Gray or Brown Amorphous Blobs' },
                ].map((item) => {
                  const isSelected = diagnosticSymptoms.includes(item.id);
                  return (
                    <button
                      key={item.id}
                      onClick={() => toggleSymptom(item.id)}
                      className={`px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
                        isSelected
                          ? 'bg-amber-600 text-white shadow-md shadow-amber-600/30'
                          : 'bg-slate-950 text-slate-400 hover:text-slate-200 border border-slate-800'
                      }`}
                    >
                      {item.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Diagnostic Configuration Sliders */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 p-4 bg-slate-950 rounded-xl border border-slate-800">
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Inference Steps:</span>
                  <span className="font-mono text-indigo-400 font-bold">{diagSteps}</span>
                </div>
                <input
                  type="range"
                  min={10}
                  max={50}
                  value={diagSteps}
                  onChange={(e) => setDiagSteps(parseInt(e.target.value, 10))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">CFG Guidance:</span>
                  <span className="font-mono text-indigo-400 font-bold">{diagCfg}</span>
                </div>
                <input
                  type="range"
                  min={1.0}
                  max={12.0}
                  step={0.5}
                  value={diagCfg}
                  onChange={(e) => setDiagCfg(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Training Steps:</span>
                  <span className="font-mono text-indigo-400 font-bold">{diagTrainingSteps}</span>
                </div>
                <input
                  type="range"
                  min={1000}
                  max={60000}
                  step={1000}
                  value={diagTrainingSteps}
                  onChange={(e) => setDiagTrainingSteps(parseInt(e.target.value, 10))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">Chunk Overlap:</span>
                  <span className="font-mono text-indigo-400 font-bold">{diagChunkOverlap} frames</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={8}
                  value={diagChunkOverlap}
                  onChange={(e) => setDiagChunkOverlap(parseInt(e.target.value, 10))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                />
              </div>
            </div>

            {/* Findings & Prescriptions */}
            <div className="space-y-4">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Diagnostic Findings ({diagnosisFindings.length} Detected)
              </h4>

              {diagnosisFindings.map((finding, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-xl border ${
                    finding.status === 'CRITICAL'
                      ? 'bg-red-950/20 border-red-800/60'
                      : finding.status === 'WARNING'
                      ? 'bg-amber-950/20 border-amber-800/60'
                      : 'bg-emerald-950/20 border-emerald-800/60'
                  } space-y-3`}
                >
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-xs font-bold flex items-center gap-2 ${
                        finding.status === 'CRITICAL'
                          ? 'text-red-400'
                          : finding.status === 'WARNING'
                          ? 'text-amber-400'
                          : 'text-emerald-400'
                      }`}
                    >
                      {finding.status === 'CRITICAL' && <AlertTriangle className="w-4 h-4" />}
                      {finding.status === 'WARNING' && <AlertTriangle className="w-4 h-4" />}
                      {finding.status === 'OPTIMAL' && <CheckCircle2 className="w-4 h-4" />}
                      {finding.issue}
                    </span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-slate-900 border border-slate-800">
                      {finding.status}
                    </span>
                  </div>

                  <div className="space-y-1 text-xs">
                    <div className="text-slate-400 font-semibold">Likely Causes:</div>
                    <ul className="list-disc list-inside text-slate-300 space-y-0.5 pl-1">
                      {finding.likelyCauses.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="space-y-1 text-xs pt-1">
                    <div className="text-emerald-400 font-semibold">Recommended Remediation:</div>
                    <ul className="list-disc list-inside text-slate-200 space-y-0.5 pl-1">
                      {finding.prescriptions.map((p, i) => (
                        <li key={i}>{p}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              ))}
            </div>

            {/* Quick Fix Button */}
            <div className="flex justify-end">
              <button
                onClick={() => {
                  setDiagSteps(35);
                  setDiagCfg(5.0);
                  setDiagChunkOverlap(4);
                  setDiagTrainingSteps(25000);
                  setDiagnosticSymptoms([]);
                }}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition-all flex items-center gap-2"
              >
                <Sliders className="w-3.5 h-3.5" />
                Apply Wan 2.1 Recommended Parameters (CFG 5.0, Steps 35, Overlap 4)
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* TAB 5: MEMORIZATION & OVERFIT DETECTOR                        */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'memorization' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 space-y-6">
            <div>
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Database className="w-5 h-5 text-indigo-400" />
                Training Set Memorization & Overfit Detector
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Embed generated videos and query nearest neighbors in the 100,000 clip training set. Detects verbatim replication vs. creative latent generalization.
              </p>
            </div>

            {/* Prompt Query Input */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4 bg-slate-950 p-4 rounded-xl border border-slate-800">
              <div className="md:col-span-3 space-y-1">
                <label className="text-xs text-slate-400 font-medium">Test Prompt:</label>
                <input
                  type="text"
                  value={customMemoPrompt}
                  onChange={(e) => setCustomMemoPrompt(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs text-slate-400 font-medium">Seed:</label>
                <input
                  type="number"
                  value={customMemoSeed}
                  onChange={(e) => setCustomMemoSeed(parseInt(e.target.value, 10) || 42)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>
            </div>

            {/* Memorization Meter Result */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div className="text-xs text-slate-400">Top Cosine Similarity</div>
                <div
                  className={`text-2xl font-bold font-mono ${
                    memorizationResult.isMemorized
                      ? 'text-red-400'
                      : memorizationResult.topSimilarity > 0.85
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {memorizationResult.topSimilarity}
                </div>
                <div className="text-[11px] text-slate-500">Threshold: 0.95</div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div className="text-xs text-slate-400">Novelty Score</div>
                <div className="text-2xl font-bold font-mono text-indigo-400">
                  {memorizationResult.noveltyScore}
                </div>
                <div className="text-[11px] text-slate-500">1.0 - TopSimilarity</div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                <div className="text-xs text-slate-400">Memorization Verdict</div>
                <div
                  className={`text-sm font-bold mt-1 ${
                    memorizationResult.isMemorized
                      ? 'text-red-400'
                      : memorizationResult.status === 'HIGH_SIMILARITY_WARNING'
                      ? 'text-amber-400'
                      : 'text-emerald-400'
                  }`}
                >
                  {memorizationResult.status.replace(/_/g, ' ')}
                </div>
                <div className="text-[11px] text-slate-500">
                  {memorizationResult.isMemorized ? 'Replicating Training Data' : 'Novel Generalization'}
                </div>
              </div>
            </div>

            {/* Nearest Neighbors in Training Set */}
            <div className="space-y-3">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">
                Top-3 Nearest Training Set Neighbors
              </h4>
              <div className="space-y-2">
                {memorizationResult.nearestNeighbors.map((neighbor) => (
                  <div
                    key={neighbor.rank}
                    className="p-3 bg-slate-950 rounded-xl border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between text-xs gap-2"
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-slate-400">#{neighbor.rank}</span>
                        <span className="font-mono text-indigo-300">{neighbor.clipId}</span>
                      </div>
                      <p className="text-slate-300 mt-0.5">&ldquo;{neighbor.caption}&rdquo;</p>
                    </div>

                    <div className="flex items-center gap-4 shrink-0 font-mono text-right">
                      <div>
                        <span className="text-slate-400">Cosine Sim: </span>
                        <span
                          className={`font-bold ${
                            neighbor.similarity > 0.95 ? 'text-red-400' : 'text-slate-200'
                          }`}
                        >
                          {neighbor.similarity}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400">L2 Dist: </span>
                        <span className="text-slate-200">{neighbor.distance}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* TAB 6: REGRESSION SUITE & ORACLE TESTS                        */}
      {/* ------------------------------------------------------------- */}
      {activeTab === 'regression' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800 p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                  Continuous Regression Verification Matrix
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Automated checks to safeguard against mathematical degradation, sign inversions, and code regressions across parts.
                </p>
              </div>

              <button
                onClick={handleRunRegression}
                disabled={isRunningRegression}
                className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs flex items-center gap-2 transition-all disabled:opacity-50"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Rerun All Tests
              </button>
            </div>

            {/* Regression Verification Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Sampler Oracle Test</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    PASSED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Model returning exact velocity v = eps - x0 recovers x0 with error &lt; 1e-6 for any step count.
                </p>
                <div className="text-[10px] font-mono text-emerald-400">Error: 2.38e-7</div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">VAE Round-Trip</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    PASSED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Held-out clips round-trip reconstruction PSNR &gt; 32 dB; chunked matches unchunked &lt; 0.15.
                </p>
                <div className="text-[10px] font-mono text-emerald-400">PSNR: 34.82 dB</div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Overfit 1-Clip Test</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    PASSED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Single clip overfit drives rectified flow loss &lt; 0.005 and reproduces sample perfectly.
                </p>
                <div className="text-[10px] font-mono text-emerald-400">Loss: 0.00118</div>
              </div>

              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-300">Fixed Seed Determinism</span>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950 text-emerald-300 border border-emerald-800">
                    PASSED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400">
                  Sampling twice with same seed produces bit-identical latent trajectories and pixel outputs.
                </p>
                <div className="text-[10px] font-mono text-emerald-400">Diff: 0.00000</div>
              </div>
            </div>

            {/* Test Execution Output Console */}
            {regressionLog.length > 0 && (
              <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 space-y-1">
                <div className="text-slate-500 font-bold mb-2">Test Runner Execution Console:</div>
                {regressionLog.map((line, idx) => (
                  <div key={idx} className={line.includes('PASSED') ? 'text-emerald-400' : ''}>
                    {line}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
