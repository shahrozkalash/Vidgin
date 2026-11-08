import React, { useState } from 'react';
import { Play, RotateCcw, CheckCircle2, Video, ArrowUp, Sliders } from 'lucide-react';

const SAMPLES = [
  {
    label: '💡 Glowing creature',
    text: 'A glowing cyan creature gliding gracefully through misty indigo mountain water at twilight',
  },
  {
    label: '💡 Holographic clock',
    text: 'Futuristic holographic clock turning smoothly over dark reflective grid terrain',
  },
  {
    label: '💡 Neon city flight',
    text: 'Cinematic aerial drone flight sweeping past rain-washed neon skyscraper rooftops',
  },
];

export const TryItSection: React.FC = () => {
  const [prompt, setPrompt] = useState<string>(
    'A glowing cyan creature gliding gracefully through misty indigo mountain water at twilight, cinematic 4k'
  );
  const [duration, setDuration] = useState<number>(30); // 5 to 60s
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [genProgress, setGenProgress] = useState<number>(0);
  const [activeChunkStep, setActiveChunkStep] = useState<number>(0);
  const [hasGenerated, setHasGenerated] = useState<boolean>(true);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const totalChunks = Math.ceil(duration / 5);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 3000);
  };

  const handleGenerate = () => {
    if (isGenerating || !prompt.trim()) return;
    setIsGenerating(true);
    setGenProgress(0);
    setActiveChunkStep(0);

    const chunks = Math.ceil(duration / 5);
    const intervalMs = 300;
    let step = 0;

    const timer = setInterval(() => {
      step++;
      const currentChunk = Math.min(chunks - 1, Math.floor((step / 10) * chunks));
      setActiveChunkStep(currentChunk);
      setGenProgress(Math.min(100, Math.round((step / 10) * 100)));

      if (step >= 10) {
        clearInterval(timer);
        setIsGenerating(false);
        setHasGenerated(true);
        showToast(`Your ${duration}s video is ready`);
      }
    }, intervalMs);
  };

  return (
    <section id="try_it" className="scroll-mt-20 px-4 pb-4">
      {/* Toast notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-full bg-slate-900 text-white shadow-xl flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2">
          <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
          <span className="text-sm font-medium">{toastMessage}</span>
        </div>
      )}

      {/* Chat-style card */}
      <div className="max-w-[720px] mx-auto bg-white rounded-[28px] shadow-xl shadow-green-900/5 border border-green-900/10 overflow-hidden">
        {/* Card header */}
        <div className="flex items-center gap-3 px-5 sm:px-6 py-4 border-b border-slate-100">
          <div className="w-10 h-10 rounded-2xl bg-green-600 flex items-center justify-center shrink-0">
            <Video className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="font-bold text-slate-900 text-lg leading-tight">Vidgin</div>
            <div className="text-sm text-slate-500 leading-tight">Turns words into video</div>
          </div>
        </div>

        <div className="px-5 sm:px-6 py-5 space-y-5">
          {/* User message bubble */}
          <div className="flex justify-end">
            <div className="max-w-[85%] bg-green-600 text-white rounded-3xl rounded-br-lg px-4 py-3 text-[15px] leading-relaxed">
              {prompt.trim() ? prompt : '…'}
            </div>
          </div>

          {/* Vidgin answer bubble */}
          <div className="flex justify-start">
            <div className="max-w-[94%] w-full bg-slate-50 border border-slate-100 rounded-3xl rounded-bl-lg px-4 py-4 space-y-3">
              <p className="text-[15px] text-slate-700 font-medium">
                {isGenerating
                  ? `Making your video — part ${activeChunkStep + 1} of ${totalChunks}…`
                  : `Here's your ${duration}-second video:`}
              </p>

              {/* Video preview */}
              <div className="relative aspect-video rounded-2xl bg-gradient-to-tr from-green-100 via-emerald-50 to-teal-100 border border-green-100 overflow-hidden flex items-center justify-center">
                <div className="w-14 h-14 rounded-full bg-white shadow-md flex items-center justify-center">
                  <Play className="w-6 h-6 text-green-700 ml-0.5" />
                </div>
                <span className="absolute bottom-2 left-2 text-[11px] font-medium bg-white/85 px-2 py-0.5 rounded-full text-slate-600">
                  {duration}s video
                </span>
              </div>

              {/* Progress bar during generation */}
              {isGenerating && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Making it, step by step</span>
                    <span className="text-green-700 font-bold">{genProgress}%</span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-green-500 transition-all duration-200"
                      style={{ width: `${genProgress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Again */}
              {!isGenerating && hasGenerated && (
                <div>
                  <button
                    onClick={handleGenerate}
                    className="min-h-[40px] px-4 py-2 rounded-full bg-white border border-slate-200 text-sm font-medium text-slate-700 hover:bg-slate-100 flex items-center gap-1.5 transition-colors cursor-pointer"
                  >
                    <RotateCcw className="w-4 h-4" />
                    <span>Again</span>
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Choose duration */}
          <div className="space-y-2.5 pt-1">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <label htmlFor="duration-slider" className="text-[15px] font-semibold text-slate-700 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-green-600" />
                <span>
                  How long? <strong className="text-green-700">{duration}s</strong>
                </span>
              </label>
              <div className="flex flex-wrap gap-1.5">
                {[5, 15, 30, 45, 60].map((sec) => (
                  <button
                    key={sec}
                    onClick={() => setDuration(sec)}
                    className={`min-h-[40px] px-3.5 py-1.5 text-sm font-semibold rounded-full transition-colors cursor-pointer ${
                      duration === sec
                        ? 'bg-green-600 text-white shadow-sm'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
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
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-green-600"
            />

            <div className="flex justify-between text-xs text-slate-400">
              <span>5s (1 chunk)</span>
              <span>15s (3 chunks)</span>
              <span>30s (6 chunks)</span>
              <span>60s (12 chunks)</span>
            </div>
          </div>

          {/* Idea chips */}
          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((sample) => (
              <button
                key={sample.label}
                onClick={() => setPrompt(sample.text)}
                className="min-h-[40px] px-3.5 py-2 rounded-full bg-white border border-slate-200 text-sm font-medium text-slate-700 hover:bg-slate-50 transition-colors cursor-pointer"
              >
                {sample.label}
              </button>
            ))}
          </div>
        </div>

        {/* Message input row */}
        <div className="px-5 sm:px-6 pb-5">
          <div className="flex items-end gap-2 bg-slate-50 border border-slate-200 rounded-[26px] p-2 focus-within:ring-2 focus-within:ring-green-500 focus-within:border-transparent transition-shadow">
            <textarea
              id="prompt-input"
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your video…"
              aria-label="Describe your video"
              className="flex-1 bg-transparent px-3 py-2 text-[15px] text-slate-800 placeholder-slate-400 focus:outline-none resize-none"
            />
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !prompt.trim()}
              aria-label="Generate video"
              className="w-11 h-11 shrink-0 rounded-full bg-green-600 hover:bg-green-700 disabled:opacity-40 text-white flex items-center justify-center transition-colors cursor-pointer"
            >
              {isGenerating ? (
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <ArrowUp className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};
