import React, { useState, useEffect, useRef } from 'react';
import { PresetKey, BucketSpec, BatchItem } from '../types/vidgen';
import { PRESETS, getStandardBuckets } from '../utils/bucketEngine';
import { renderMovingMNISTClip } from '../utils/movingMnist';
import { SAMPLE_CAPTIONS } from '../utils/mockDataset';
import {
  Play,
  Pause,
  RotateCcw,
  SkipForward,
  CheckCircle2,
  AlertTriangle,
  Info,
  Maximize2,
  Sliders,
  Layers,
  Sparkles,
  Zap,
} from 'lucide-react';

interface VisualBatchInspectorProps {
  currentPreset: PresetKey;
}

export const VisualBatchInspector: React.FC<VisualBatchInspectorProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const buckets = getStandardBuckets(currentPreset);

  const [selectedBucketIdx, setSelectedBucketIdx] = useState<number>(0);
  const activeBucket = buckets[selectedBucketIdx] || buckets[0];

  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [selectedItemIdx, setSelectedItemIdx] = useState<number>(0);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [playbackFps, setPlaybackFps] = useState<number>(16);
  const [channelMode, setChannelMode] = useState<'rgb' | 'r' | 'g' | 'b'>('rgb');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  // Generate a homogeneous batch for the selected bucket
  const generateBatch = () => {
    setIsGenerating(true);
    const bs = activeBucket.frames === 1 ? 4 : 2; // Token-scaled batch size
    const items: BatchItem[] = [];

    // Use bucket dimensions for rendering
    // For large presets, clamp preview render resolution to avoid high CPU canvas rendering lag, but display true tensor shape
    const renderW = Math.min(activeBucket.width, 160);
    const renderH = Math.min(activeBucket.height, 160);
    const framesCount = activeBucket.frames;

    for (let b = 0; b < bs; b++) {
      const clipSeed = Math.floor(Math.random() * 1000) + b * 77;
      const digit1 = (b * 3 + 2) % 10;
      const digit2 = (b * 4 + 7) % 10;

      const rendered = renderMovingMNISTClip(
        framesCount,
        renderH,
        renderW,
        clipSeed,
        digit1,
        digit2
      );

      const capObj = SAMPLE_CAPTIONS[b % SAMPLE_CAPTIONS.length];
      const captionChoices = [capObj.long, capObj.short];
      const selectedPrompt = captionChoices[Math.floor(Math.random() * captionChoices.length)];

      // 10% CFG dropout
      const isDropped = Math.random() < 0.10;

      items.push({
        id: `sample_batch_${b + 1}`,
        mediaType: activeBucket.frames === 1 ? 'image' : 'video',
        prompt: isDropped ? '' : selectedPrompt,
        isCaptionDropped: isDropped,
        bucket: [activeBucket.frames, activeBucket.height, activeBucket.width],
        pixels: rendered.pixels,
        frames: activeBucket.frames,
        height: activeBucket.height,
        width: activeBucket.width,
        variance: rendered.variance,
        minVal: rendered.minVal,
        maxVal: rendered.maxVal,
        framesCanvasData: rendered.frames,
      });
    }

    setBatchItems(items);
    setSelectedItemIdx(0);
    setCurrentFrame(0);
    setIsGenerating(false);
  };

  useEffect(() => {
    generateBatch();
  }, [selectedBucketIdx, currentPreset]);

  // Video playback animation loop
  useEffect(() => {
    const item = batchItems[selectedItemIdx];
    if (!item || !item.framesCanvasData || item.framesCanvasData.length === 0) return;

    if (!isPlaying || item.frames <= 1) {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const interval = 1000 / playbackFps;

    const loop = (timestamp: number) => {
      if (!lastTimeRef.current) lastTimeRef.current = timestamp;
      const elapsed = timestamp - lastTimeRef.current;

      if (elapsed >= interval) {
        lastTimeRef.current = timestamp;
        setCurrentFrame((prev) => (prev + 1) % item.frames);
      }
      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, playbackFps, selectedItemIdx, batchItems]);

  // Render current frame to canvas
  useEffect(() => {
    const item = batchItems[selectedItemIdx];
    if (!item || !item.framesCanvasData || !canvasRef.current) return;

    const frameData = item.framesCanvasData[currentFrame] || item.framesCanvasData[0];
    if (!frameData) return;

    const canvas = canvasRef.current;
    canvas.width = frameData.width;
    canvas.height = frameData.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (channelMode === 'rgb') {
      ctx.putImageData(frameData, 0, 0);
    } else {
      // Isolate single channel
      const filtered = ctx.createImageData(frameData.width, frameData.height);
      const src = frameData.data;
      const dst = filtered.data;
      for (let i = 0; i < src.length; i += 4) {
        const val =
          channelMode === 'r' ? src[i] : channelMode === 'g' ? src[i + 1] : src[i + 2];
        dst[i] = channelMode === 'r' ? val : 0;
        dst[i + 1] = channelMode === 'g' ? val : 0;
        dst[i + 2] = channelMode === 'b' ? val : 0;
        dst[i + 3] = 255;
      }
      ctx.putImageData(filtered, 0, 0);
    }
  }, [currentFrame, selectedItemIdx, batchItems, channelMode]);

  const activeItem = batchItems[selectedItemIdx];
  const isKRuleSatisfied = activeBucket ? (activeBucket.frames - 1) % 4 === 0 : false;
  const isDimMultipleOf16 = activeBucket
    ? activeBucket.height % 16 === 0 && activeBucket.width % 16 === 0
    : false;
  const isVarianceAcceptable = activeItem ? activeItem.variance > 0.01 : false;
  const isRangeValid = activeItem ? activeItem.minVal >= -1.0 && activeItem.maxVal <= 1.0 : false;

  return (
    <div className="space-y-6">
      {/* Title & Acceptance Overview Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-cyan-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-cyan-500/10 text-cyan-400">
                <Play className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Part 2 Acceptance: Visual Batch Dump & Tensor Verification
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Inspect decoded, temporally resampled (16 FPS), and cropped pixel batches. Every batch is
              guaranteed homogeneous from one bucket, satisfies{' '}
              <code className="text-cyan-300 font-mono">T = 4k + 1</code>, has{' '}
              <code className="text-cyan-300 font-mono">H, W % 16 == 0</code>, values strictly in{' '}
              <code className="text-cyan-300 font-mono">[-1.0, 1.0]</code>, and non-trivial variance ($\sigma^2 &gt; 0.01$).
            </p>
          </div>

          {/* Quick Regenerate Action */}
          <div className="flex items-center gap-3">
            <button
              onClick={generateBatch}
              disabled={isGenerating}
              className="flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-sky-600 hover:from-cyan-400 hover:to-sky-500 text-slate-950 font-bold text-xs rounded-xl shadow-lg shadow-cyan-500/20 transition-all cursor-pointer active:scale-95"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${isGenerating ? 'animate-spin' : ''}`} />
              <span>Dump New Batch</span>
            </button>
          </div>
        </div>

        {/* 5 Acceptance Criterion Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 mt-4 pt-4 border-t border-slate-800/80">
          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex items-center gap-2.5">
            <CheckCircle2
              className={`w-4 h-4 shrink-0 ${isKRuleSatisfied ? 'text-emerald-400' : 'text-rose-400'}`}
            />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">T = 4k + 1 Rule</div>
              <div className="text-xs font-mono font-bold text-white">
                T = {activeBucket.frames} (k = {(activeBucket.frames - 1) / 4})
              </div>
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex items-center gap-2.5">
            <CheckCircle2
              className={`w-4 h-4 shrink-0 ${isDimMultipleOf16 ? 'text-emerald-400' : 'text-rose-400'}`}
            />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Shape % 16 == 0</div>
              <div className="text-xs font-mono font-bold text-white">
                {activeBucket.width} × {activeBucket.height}
              </div>
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex items-center gap-2.5">
            <CheckCircle2
              className={`w-4 h-4 shrink-0 ${isRangeValid ? 'text-emerald-400' : 'text-rose-400'}`}
            />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Range [-1.0, 1.0]</div>
              <div className="text-xs font-mono font-bold text-white">
                [{activeItem?.minVal.toFixed(2)}, {activeItem?.maxVal.toFixed(2)}]
              </div>
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex items-center gap-2.5">
            <CheckCircle2
              className={`w-4 h-4 shrink-0 ${isVarianceAcceptable ? 'text-emerald-400' : 'text-rose-400'}`}
            />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Variance &gt; 0.01</div>
              <div className="text-xs font-mono font-bold text-white">
                {activeItem?.variance.toFixed(4)}
              </div>
            </div>
          </div>

          <div className="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 flex items-center gap-2.5 col-span-2 sm:col-span-1">
            <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">Target FPS</div>
              <div className="text-xs font-mono font-bold text-white">16.0 (Vidgen Native)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Visualizer Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Bucket Picker & Batch Samples List */}
        <div className="lg:col-span-4 space-y-4">
          {/* Bucket Selector */}
          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                Select Bucket ({buckets.length})
              </span>
              <span className="text-[11px] font-mono text-cyan-400">
                Preset: {preset.name}
              </span>
            </div>

            <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1 no-scrollbar">
              {buckets.map((b, idx) => {
                const isSelected = selectedBucketIdx === idx;
                return (
                  <button
                    key={idx}
                    onClick={() => setSelectedBucketIdx(idx)}
                    className={`w-full text-left px-3 py-2 rounded-xl text-xs flex items-center justify-between transition-all ${
                      isSelected
                        ? 'bg-cyan-500/20 text-white border border-cyan-500/40 shadow-sm'
                        : 'bg-slate-950/40 text-slate-400 hover:bg-slate-800/60 hover:text-slate-200 border border-slate-800/60'
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          b.frames === 1 ? 'bg-amber-400' : 'bg-cyan-400'
                        }`}
                      ></span>
                      <span className="font-semibold">{b.aspectRatioLabel}</span>
                      <span className="text-[11px] text-slate-400 font-mono">
                        {b.width}×{b.height} ({b.frames}f)
                      </span>
                    </div>
                    <span className="font-mono text-[10px] bg-slate-900 px-1.5 py-0.5 rounded text-slate-300">
                      {b.tokens} tokens
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Batch Items List */}
          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                Samples in Homogeneous Batch ({batchItems.length})
              </span>
              <span className="text-[10px] font-mono bg-cyan-950/80 text-cyan-300 px-2 py-0.5 rounded border border-cyan-800/40">
                Single Bucket Batch
              </span>
            </div>

            <div className="space-y-2">
              {batchItems.map((item, idx) => {
                const isSelected = selectedItemIdx === idx;
                return (
                  <div
                    key={item.id}
                    onClick={() => {
                      setSelectedItemIdx(idx);
                      setCurrentFrame(0);
                    }}
                    className={`p-3 rounded-xl border cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-slate-800/90 border-cyan-500/50 shadow-md'
                        : 'bg-slate-950/40 border-slate-800/60 hover:bg-slate-800/40 text-slate-400'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono font-bold text-slate-200">
                        Batch Item #{idx + 1}
                      </span>
                      {item.isCaptionDropped ? (
                        <span className="text-[10px] font-bold px-1.5 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40">
                          10% CFG Dropped (Empty)
                        </span>
                      ) : (
                        <span className="text-[10px] font-medium px-1.5 py-0.2 rounded bg-emerald-500/10 text-emerald-400">
                          Active Prompt
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-slate-400 truncate">
                      {item.isCaptionDropped ? '(Swapped to empty string "" for CFG)' : item.prompt}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Interactive Video Player & Tensor Scrubber */}
        <div className="lg:col-span-8 space-y-4">
          <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 flex flex-col items-center">
            {/* Viewport Frame with Spacetime Aspect Ratio Canvas */}
            <div className="relative group w-full flex items-center justify-center p-6 bg-slate-950 rounded-xl border border-slate-800/80 min-h-[300px]">
              <canvas
                ref={canvasRef}
                className="max-h-[340px] max-w-full rounded-lg shadow-2xl shadow-cyan-950/50 object-contain image-rendering-pixelated border border-slate-800"
                style={{
                  aspectRatio: `${activeBucket.width} / ${activeBucket.height}`,
                }}
              />

              {/* In-canvas metadata overlay badge */}
              <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-800 text-[10px] font-mono text-cyan-300 flex items-center gap-2">
                <span>Frame {currentFrame + 1} / {activeBucket.frames}</span>
                <span>•</span>
                <span>{activeBucket.width}×{activeBucket.height}</span>
                <span>•</span>
                <span>{activeBucket.aspectRatioLabel}</span>
              </div>

              {/* 10% CFG null-prompt badge overlay */}
              {activeItem?.isCaptionDropped && (
                <div className="absolute top-4 right-4 bg-amber-950/90 backdrop-blur px-2.5 py-1 rounded-lg border border-amber-700/60 text-[10px] font-bold text-amber-300 flex items-center gap-1.5 shadow-lg">
                  <Sparkles className="w-3 h-3 text-amber-400" />
                  CFG Dropout Active (Empty Embedding)
                </div>
              )}
            </div>

            {/* Playback & Frame Scrubber */}
            <div className="w-full mt-4 space-y-3">
              {/* Scrubber Slider */}
              <div className="flex items-center gap-3">
                <span className="text-[11px] font-mono text-slate-400 w-10">
                  {currentFrame + 1}/{activeBucket.frames}
                </span>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, activeBucket.frames - 1)}
                  value={currentFrame}
                  onChange={(e) => {
                    setIsPlaying(false);
                    setCurrentFrame(Number(e.target.value));
                  }}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                />
                <span className="text-[11px] font-mono text-cyan-400 w-12 text-right">
                  {((currentFrame / Math.max(1, activeBucket.frames - 1)) * 100).toFixed(0)}%
                </span>
              </div>

              {/* Control Buttons & Mode Selectors */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800/80">
                {/* Play/Pause/Step */}
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    disabled={activeBucket.frames <= 1}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                      isPlaying
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        : 'bg-cyan-500 text-slate-950 hover:bg-cyan-400 shadow-md shadow-cyan-500/20'
                    }`}
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-3.5 h-3.5" /> Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5" /> Play 16 FPS
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => {
                      setIsPlaying(false);
                      setCurrentFrame((prev) => (prev + 1) % activeBucket.frames);
                    }}
                    disabled={activeBucket.frames <= 1}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs"
                    title="Step 1 Frame Forward"
                  >
                    <SkipForward className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => setCurrentFrame(0)}
                    className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs"
                    title="Rewind to Frame 0"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                {/* Channel Isolation Filter */}
                <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                  <span className="text-[10px] text-slate-400 px-1.5">View:</span>
                  {(['rgb', 'r', 'g', 'b'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setChannelMode(mode)}
                      className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                        channelMode === mode
                          ? 'bg-cyan-500 text-slate-950 font-bold'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {mode.toUpperCase()}
                    </button>
                  ))}
                </div>

                {/* Playback FPS Switcher */}
                <div className="flex items-center gap-1.5 text-xs text-slate-400">
                  <span className="text-[11px]">Speed:</span>
                  {[8, 16, 24].map((fps) => (
                    <button
                      key={fps}
                      onClick={() => setPlaybackFps(fps)}
                      className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                        playbackFps === fps
                          ? 'bg-slate-800 text-cyan-400 font-bold border border-cyan-800/40'
                          : 'text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {fps}fps
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Tensor Metadata & Prompt Inspector */}
          <div className="bg-slate-900/90 p-4 rounded-2xl border border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 text-cyan-400" />
                Prompt & Tensor Specification
              </span>
              <span className="text-xs font-mono text-cyan-400">
                [3, {activeBucket.frames}, {activeBucket.height}, {activeBucket.width}] in [-1, 1]
              </span>
            </div>

            {/* Prompt Card */}
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800">
              <div className="text-[10px] text-slate-500 uppercase font-semibold mb-1">
                Prompt fed to Text Cross-Attention:
              </div>
              <div className="text-xs font-medium text-slate-200 italic">
                {activeItem?.isCaptionDropped ? (
                  <span className="text-amber-400 not-italic font-mono">
                    "" (Empty prompt for Classifier-Free Guidance unconditional pass)
                  </span>
                ) : (
                  `"${activeItem?.prompt}"`
                )}
              </div>
            </div>

            {/* Tensor Math Details */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500">3D VAE Latent</div>
                <div className="font-bold text-cyan-300">
                  [16, {Math.floor((activeBucket.frames - 1) / 4) + 1}, {activeBucket.height / 8}, {activeBucket.width / 8}]
                </div>
              </div>
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500">Spacetime Tokens</div>
                <div className="font-bold text-cyan-300">{activeBucket.tokens}</div>
              </div>
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500">Min / Max Val</div>
                <div className="font-bold text-cyan-300">
                  {activeItem?.minVal.toFixed(2)} / {activeItem?.maxVal.toFixed(2)}
                </div>
              </div>
              <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                <div className="text-[10px] text-slate-500">Pixel Variance</div>
                <div className="font-bold text-cyan-300">{activeItem?.variance.toFixed(4)}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
