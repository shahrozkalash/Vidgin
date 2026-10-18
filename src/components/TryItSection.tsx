import React, { useState, useEffect } from 'react';
import { PresetKey } from '../types/vidgen';
import { PRESETS } from '../utils/bucketEngine';
import {
  Sparkles,
  Play,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Code2,
  Terminal,
  CheckCircle2,
  ChevronDown,
  Copy,
  Check,
  Tag,
  Columns,
} from 'lucide-react';

interface TryItSectionProps {
  currentPreset: PresetKey;
  onPresetChange: (preset: PresetKey) => void;
}

export const TryItSection: React.FC<TryItSectionProps> = ({
  currentPreset,
  onPresetChange,
}) => {
  const [prompt, setPrompt] = useState<string>(
    'A glowing cyan creature gliding gracefully through misty indigo mountain water at twilight, cinematic 4k'
  );
  const [duration, setDuration] = useState<number>(30);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [genProgress, setGenProgress] = useState<number>(0);
  const [activeChunkStep, setActiveChunkStep] = useState<number>(0);
  const [hasGenerated, setHasGenerated] = useState<boolean>(true);

  // Thumbs & feedback widget state (matching feedback_widget.py)
  const [feedbackRating, setFeedbackRating] = useState<'thumbs_up' | 'thumbs_down' | null>(null);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [isCompareMode, setIsCompareMode] = useState<boolean>(false);
  const [compareChoice, setCompareChoice] = useState<'A' | 'B' | 'neither' | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [copiedCode, setCopiedCode] = useState<boolean>(false);

  // Accordion state
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const samplePrompts = [
    'A glowing cyan creature gliding gracefully through misty indigo mountain water at twilight',
    'Futuristic holographic clock turning smoothly over dark reflective grid terrain',
    'Cinematic aerial drone flight sweeping past rain-washed neon skyscraper rooftops',
  ];

  const quickTags = ['wrong motion', 'wrong character', 'flicker', 'artifact'];

  const handleGenerate = () => {
    setIsGenerating(true);
    setGenProgress(0);
    setActiveChunkStep(0);
    setFeedbackRating(null);
    setSelectedTags([]);
    setCompareChoice(null);

    const totalChunks = Math.ceil(duration / 5);
    const intervalMs = 300;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      const currentChunk = Math.min(totalChunks - 1, Math.floor((step / 10) * totalChunks));
      setActiveChunkStep(currentChunk);
      setGenProgress(Math.min(100, Math.round((step / 10) * 100)));

      if (step >= 10) {
        clearInterval(timer);
        setIsGenerating(false);
        setHasGenerated(true);
        showToast('Video synthesized successfully across 6 chunks!');
      }
    }, intervalMs);
  };

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 3000);
  };

  const handleFeedback = (rating: 'thumbs_up' | 'thumbs_down') => {
    setFeedbackRating(rating);
    showToast(rating === 'thumbs_up' ? 'Feedback recorded: Thumbs up!' : 'Feedback recorded: Thumbs down.');
  };

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) =>
      prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
    );
  };

  const handleCopyCode = () => {
    navigator.clipboard.writeText(pythonSnippet);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const pythonSnippet = `# Run Vidgen 30-Second Generation in Python
from vidgen.config import LongVideoConfig, AttentionConfig, TINY_PRESET
from vidgen.attention import DecoupledDiTModel
from vidgen.memory import MemoryStore

# 1. Initialize configuration with Decoupled Attention
config = LongVideoConfig(
    duration_seconds=30.0,
    fps=16.0,
    preset=TINY_PRESET,
    attention=AttentionConfig(attention_mode="decoupled", time_window=16),
)

# 2. Query episodic memory for character consistency
memory = MemoryStore(index_dir="./vidgen_memory")
refs = memory.search_entries(query="${prompt.slice(0, 30)}", top_k=2)

print(f"Synthesizing {config.duration_seconds}s with {len(refs)} reference tags...")
`;

  return (
    <section id="try_it" className="scroll-mt-20 space-y-6">
      {/* Toast notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 p-4 rounded-xl bg-slate-900 border border-cyan-400 text-white shadow-xl flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2">
          <CheckCircle2 className="w-5 h-5 text-cyan-400 shrink-0" />
          <span className="text-sm font-medium">{toastMessage}</span>
        </div>
      )}

      {/* Section Header */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Try it
        </h2>
        <p className="mt-1 text-base text-slate-300">
          Simulate chunked video synthesis and submit human preference feedback.
        </p>
      </div>

      {/* Generator Studio Card */}
      <div className="p-5 sm:p-6 rounded-2xl bg-slate-900/80 border border-slate-800 space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Controls Column (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            {/* Prompt input */}
            <div>
              <label htmlFor="prompt-input" className="block text-sm font-semibold text-slate-200 mb-1.5">
                Text Prompt
              </label>
              <textarea
                id="prompt-input"
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                placeholder="Describe your scene in detail..."
              />

              {/* Sample Prompts Pills */}
              <div className="flex flex-wrap gap-1.5 mt-2">
                {samplePrompts.map((sp, idx) => (
                  <button
                    key={idx}
                    onClick={() => setPrompt(sp)}
                    className="text-[11px] px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 transition-colors truncate max-w-full"
                  >
                    💡 {sp.slice(0, 40)}...
                  </button>
                ))}
              </div>
            </div>

            {/* Preset & Duration Controls */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div>
                <label htmlFor="gen-preset-select" className="block text-xs font-semibold text-slate-300 mb-1">
                  Preset Tier
                </label>
                <select
                  id="gen-preset-select"
                  value={currentPreset}
                  onChange={(e) => onPresetChange(e.target.value as PresetKey)}
                  className="w-full min-h-[44px] bg-slate-950 border border-slate-800 text-sm font-medium text-slate-200 rounded-xl px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"
                >
                  <option value="tiny">Tiny (~20M, CPU instant)</option>
                  <option value="small">Small (~270M, fast)</option>
                  <option value="vidgen480p">Vidgen 480p (~1.3B, Production)</option>
                  <option value="vidgen14b">Vidgen 14B (~14B, Cinematic)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1">
                  Duration
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {[5, 15, 30].map((sec) => (
                    <button
                      key={sec}
                      onClick={() => setDuration(sec)}
                      className={`min-h-[44px] rounded-xl text-xs font-semibold transition-colors ${
                        duration === sec
                          ? 'bg-cyan-400 text-slate-950 font-bold'
                          : 'bg-slate-950 border border-slate-800 text-slate-300 hover:bg-slate-800'
                      }`}
                    >
                      {sec}s ({Math.ceil(sec / 5)} ch)
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="pt-2">
              <button
                onClick={handleGenerate}
                disabled={isGenerating}
                className="w-full min-h-[44px] px-6 py-3 rounded-xl font-bold text-slate-950 bg-gradient-to-r from-sky-400 via-cyan-300 to-teal-300 hover:opacity-95 shadow-md shadow-cyan-500/20 active:scale-[0.99] transition-all flex items-center justify-center gap-2 text-base disabled:opacity-50"
              >
                {isGenerating ? (
                  <>
                    <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                    <span>Synthesizing Chunk {activeChunkStep + 1} of {Math.ceil(duration / 5)}...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5 text-slate-950" />
                    <span>Generate Video Clip</span>
                  </>
                )}
              </button>
            </div>

            {/* Progress bar during generation */}
            {isGenerating && (
              <div className="space-y-1.5 pt-1">
                <div className="flex justify-between text-xs text-slate-400 font-mono">
                  <span>Generating forward trajectory</span>
                  <span className="text-cyan-300 font-bold">{genProgress}%</span>
                </div>
                <div className="w-full h-2 bg-slate-950 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="h-full bg-gradient-to-r from-sky-400 to-cyan-300 transition-all duration-200"
                    style={{ width: `${genProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Video Preview & Feedback Column (5 cols) */}
          <div className="lg:col-span-5 flex flex-col justify-between space-y-4">
            {/* Video Canvas Simulated Display */}
            <div className="relative aspect-video rounded-xl bg-slate-950 border border-slate-800 overflow-hidden flex flex-col items-center justify-center shadow-inner">
              {/* Dynamic canvas visualizer representing smooth flowing latents */}
              <div className="absolute inset-0 bg-gradient-to-tr from-indigo-950/60 via-slate-900 to-cyan-950/40" />

              <div className="relative z-10 text-center p-4 space-y-2 max-w-[280px]">
                <div className="w-12 h-12 rounded-full bg-cyan-400/10 border border-cyan-400/30 mx-auto flex items-center justify-center text-cyan-400">
                  <Play className="w-5 h-5 ml-0.5" />
                </div>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-800 inline-block">
                  {duration}s • 16 FPS • {currentPreset}
                </span>
                <p className="text-xs text-slate-300 truncate">
                  "{prompt}"
                </p>
              </div>

              {/* Progress overlay */}
              <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between text-[11px] font-mono text-slate-400 bg-slate-950/80 px-2.5 py-1 rounded-md backdrop-blur">
                <span>Chunk {activeChunkStep + 1} / {Math.ceil(duration / 5)}</span>
                <span className="text-emerald-400">1-Latent Cosine Blend</span>
              </div>
            </div>

            {/* Thumbs & Feedback Widget (Integrated from feedback_widget.py) */}
            <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800/90 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200">
                  Human Feedback Widget
                </span>
                <button
                  onClick={() => setIsCompareMode((prev) => !prev)}
                  className="min-h-[32px] px-2 py-1 text-[11px] font-medium rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center gap-1 transition-colors"
                >
                  <Columns className="w-3 h-3 text-cyan-400" />
                  <span>{isCompareMode ? 'Standard Mode' : 'Compare Mode'}</span>
                </button>
              </div>

              {isCompareMode ? (
                /* A/B Comparison Mode */
                <div className="space-y-2 pt-1">
                  <p className="text-xs text-slate-400">Which model trajectory is better?</p>
                  <div className="grid grid-cols-3 gap-1.5">
                    {(['A', 'B', 'neither'] as const).map((choice) => (
                      <button
                        key={choice}
                        onClick={() => {
                          setCompareChoice(choice);
                          showToast(`Recorded preference: ${choice.toUpperCase()}`);
                        }}
                        className={`min-h-[44px] rounded-lg text-xs font-bold transition-all ${
                          compareChoice === choice
                            ? 'bg-cyan-400 text-slate-950'
                            : 'bg-slate-900 border border-slate-800 text-slate-300 hover:bg-slate-800'
                        }`}
                      >
                        Option {choice.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                /* Standard Thumbs Up / Down */
                <div className="space-y-2.5">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handleFeedback('thumbs_up')}
                      aria-label="Thumbs up"
                      className={`min-h-[44px] flex-1 py-2 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-all ${
                        feedbackRating === 'thumbs_up'
                          ? 'bg-emerald-500 text-slate-950 font-bold shadow-md shadow-emerald-500/20'
                          : 'bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300'
                      }`}
                    >
                      <ThumbsUp className="w-4 h-4 text-emerald-400" />
                      <span>Looks Good</span>
                    </button>

                    <button
                      onClick={() => handleFeedback('thumbs_down')}
                      aria-label="Thumbs down"
                      className={`min-h-[44px] flex-1 py-2 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition-all ${
                        feedbackRating === 'thumbs_down'
                          ? 'bg-rose-500 text-white font-bold shadow-md shadow-rose-500/20'
                          : 'bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300'
                      }`}
                    >
                      <ThumbsDown className="w-4 h-4 text-rose-400" />
                      <span>Has Issues</span>
                    </button>
                  </div>

                  {/* Quick issue tags */}
                  <div className="pt-1">
                    <span className="text-[11px] text-slate-400 block mb-1.5 flex items-center gap-1">
                      <Tag className="w-3 h-3 text-slate-400" />
                      <span>Quick diagnostic tags:</span>
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {quickTags.map((tag) => {
                        const isTagSelected = selectedTags.includes(tag);
                        return (
                          <button
                            key={tag}
                            onClick={() => toggleTag(tag)}
                            className={`min-h-[32px] px-2 py-1 rounded-md text-[11px] font-medium transition-colors ${
                              isTagSelected
                                ? 'bg-amber-400/20 border border-amber-400/60 text-amber-300'
                                : 'bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200'
                            }`}
                          >
                            {tag}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Accordions for Deep Technical Details (Python Integration & GPU Commands) */}
      <div className="space-y-3 pt-2">
        {/* Accordion 1: Python Library Code */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('python')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'python'}
          >
            <div className="flex items-center gap-2.5">
              <Code2 className="w-4 h-4 text-cyan-400" />
              <span>Deep Technical Specs: Python Library Integration Code</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'python' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'python' && (
            <div className="p-4 sm:p-5 border-t border-slate-800/60 bg-slate-950 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400 font-mono">vidgen_generate.py</span>
                <button
                  onClick={handleCopyCode}
                  className="min-h-[36px] px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-300 font-medium flex items-center gap-1.5 transition-colors"
                >
                  {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copiedCode ? 'Copied!' : 'Copy Code'}</span>
                </button>
              </div>
              <pre className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 overflow-x-auto text-xs font-mono text-slate-200 leading-relaxed">
                <code>{pythonSnippet}</code>
              </pre>
            </div>
          )}
        </div>

        {/* Accordion 2: Colab & GPU Training Commands */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('commands')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'commands'}
          >
            <div className="flex items-center gap-2.5">
              <Terminal className="w-4 h-4 text-emerald-400" />
              <span>Deep Technical Specs: Colab &amp; GPU Execution Commands</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'commands' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'commands' && (
            <div className="p-4 sm:p-5 border-t border-slate-800/60 bg-slate-950 space-y-3 text-xs font-mono text-slate-300">
              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <span className="text-slate-400 block font-sans font-semibold">1. Run Full CPU Test Suite (34 passing tests)</span>
                <code className="text-cyan-300 block bg-slate-950 p-2 rounded border border-slate-800">
                  python3 -m unittest discover -s tests
                </code>
              </div>

              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <span className="text-slate-400 block font-sans font-semibold">2. Run Attention &amp; Memory Scaling Ablations</span>
                <code className="text-cyan-300 block bg-slate-950 p-2 rounded border border-slate-800">
                  python3 scripts/ablate_decoupled_attention.py
                </code>
              </div>

              <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <span className="text-slate-400 block font-sans font-semibold">3. Run Part 0-13 CPU Acceptance Audit</span>
                <code className="text-cyan-300 block bg-slate-950 p-2 rounded border border-slate-800">
                  python3 scripts/run_cpu_audit_tests.py
                </code>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
