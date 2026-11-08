import React, { useState } from 'react';
import { CheckCircle2, ArrowUp, Sliders } from 'lucide-react';

export const TryItSection: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [duration, setDuration] = useState<number>(30); // 5 to 60s
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [genProgress, setGenProgress] = useState<number>(0);
  const [activeChunkStep, setActiveChunkStep] = useState<number>(0);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // The exchange the user actually sent — nothing shows until they send one.
  const [sentPrompt, setSentPrompt] = useState<string | null>(null);
  const [sentDuration, setSentDuration] = useState<number>(30);
  const [isDone, setIsDone] = useState<boolean>(false);

  const totalChunks = Math.ceil(sentDuration / 5);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 3000);
  };

  const handleGenerate = () => {
    if (isGenerating || !prompt.trim()) return;
    const sentText = prompt.trim();
    const sentSecs = duration;
    setSentPrompt(sentText);
    setSentDuration(sentSecs);
    setIsDone(false);
    setIsGenerating(true);
    setGenProgress(0);
    setActiveChunkStep(0);

    const chunks = Math.ceil(sentSecs / 5);
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
        setIsDone(true);
        showToast(`Your ${sentSecs}s video is ready`);
      }
    }, intervalMs);
  };

  return (
    <section id="try_it" className="scroll-mt-20 px-4 pb-4 flex-1 flex flex-col w-full">
      {/* Toast notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-full bg-slate-900 text-white shadow-xl flex items-center gap-2 animate-in fade-in slide-in-from-bottom-2">
          <CheckCircle2 className="w-5 h-5 text-green-400 shrink-0" />
          <span className="text-sm font-medium">{toastMessage}</span>
        </div>
      )}

      {/* Chat-style card */}
      <div className="max-w-[720px] w-full mx-auto bg-white rounded-[28px] shadow-xl shadow-green-900/5 border border-green-900/10 overflow-hidden flex-1 flex flex-col">
        <div className="px-5 sm:px-6 py-5 space-y-5 flex-1">
          {/* Empty state — shown until the user sends a prompt */}
          {sentPrompt === null && (
            <p className="text-center text-slate-400 text-[15px] py-4">
              Type what you want to see below, and Vidgin will make it.
            </p>
          )}

          {/* The user's sent prompt — appears only after they send it */}
          {sentPrompt !== null && (
            <div className="flex justify-end">
              <div className="max-w-[85%] bg-green-600 text-white rounded-3xl rounded-br-lg px-4 py-3 text-[15px] leading-relaxed">
                {sentPrompt}
              </div>
            </div>
          )}

          {/* Vidgin's reply — progress while making, simple done line after */}
          {sentPrompt !== null && (
            <div className="flex justify-start">
              <div className="max-w-[94%] w-full bg-slate-50 border border-slate-100 rounded-3xl rounded-bl-lg px-4 py-4 space-y-3">
                {isGenerating ? (
                  <>
                    <p className="text-[15px] text-slate-700 font-medium">
                      {`Making your video — part ${activeChunkStep + 1} of ${totalChunks}…`}
                    </p>
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
                  </>
                ) : (
                  isDone && (
                    <p className="text-[15px] text-slate-700 font-medium flex items-center gap-2">
                      <CheckCircle2 className="w-5 h-5 text-green-600 shrink-0" />
                      <span>{`Done — your ${sentDuration}-second video is ready.`}</span>
                    </p>
                  )
                )}
              </div>
            </div>
          )}

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
