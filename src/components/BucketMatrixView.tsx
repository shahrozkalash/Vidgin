import React, { useState } from 'react';
import { PresetKey, BucketSpec } from '../types/vidgen';
import { PRESETS, getStandardBuckets, findNearestBucket, computeBucketBatchSize } from '../utils/bucketEngine';
import {
  Layers,
  ArrowRight,
  Maximize2,
  Minimize2,
  Sliders,
  CheckCircle2,
  Sparkles,
  Calculator,
  Compass,
} from 'lucide-react';

interface BucketMatrixViewProps {
  currentPreset: PresetKey;
}

export const BucketMatrixView: React.FC<BucketMatrixViewProps> = ({ currentPreset }) => {
  const preset = PRESETS[currentPreset];
  const buckets = getStandardBuckets(currentPreset);

  // Playground state for nearest bucket assignment
  const [testFrames, setTestFrames] = useState<number>(preset.baseFrames);
  const [testHeight, setTestHeight] = useState<number>(preset.key === 'tiny' ? 52 : 210);
  const [testWidth, setTestWidth] = useState<number>(preset.key === 'tiny' ? 76 : 310);
  const [isTestImage, setIsTestImage] = useState<boolean>(false);
  const [imageRatio, setImageRatio] = useState<number>(0.20);
  const [targetTokens, setTargetTokens] = useState<number>(preset.targetTokensPerBatch);

  const videoBuckets = buckets.filter((b) => b.frames > 1);
  const imageBuckets = buckets.filter((b) => b.frames === 1);

  const assignedBucket = findNearestBucket(
    isTestImage ? 1 : testFrames,
    testHeight,
    testWidth,
    buckets,
    isTestImage
  );

  return (
    <div className="space-y-6">
      {/* Header & Token Scaling Concept Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/40 p-5 rounded-2xl border border-slate-800 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="p-1 rounded-md bg-indigo-500/10 text-indigo-400">
                <Layers className="w-4 h-4" />
              </span>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Spacetime Multi-Aspect Bucketing & Constant Token Budget
              </h2>
            </div>
            <p className="text-xs text-slate-300 max-w-3xl">
              Vidgen models bucket clips into <code className="text-cyan-300 font-mono">(T, H, W)</code> triples
              with <code className="text-cyan-300 font-mono">H, W % 16 == 0</code> and{' '}
              <code className="text-cyan-300 font-mono">T = 4k + 1</code>. Every batch is drawn from a single
              bucket, and batch size scales inversely with tokens so GPU memory and compute remain uniform.
            </p>
          </div>

          {/* Quick Target Token Config */}
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <Calculator className="w-4 h-4 text-cyan-400" />
            <div>
              <div className="text-[10px] text-slate-400 uppercase font-semibold">
                Target Tokens / Step
              </div>
              <div className="text-sm font-mono font-bold text-cyan-300">
                {targetTokens} tokens
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bucket Grid: Video (T=4k+1) vs Image (T=1) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Video Buckets (T = 4k + 1) */}
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
              <h3 className="text-sm font-bold text-white">
                Video Buckets (T = {preset.baseFrames} frames, k = {(preset.baseFrames - 1) / 4})
              </h3>
            </div>
            <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800/40">
              4x Temporal VAE ({Math.floor((preset.baseFrames - 1) / 4) + 1} latents)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {videoBuckets.map((b) => {
              const scaledBs = computeBucketBatchSize(b, targetTokens);
              const totalTokensInBatch = scaledBs * b.tokens;
              return (
                <div
                  key={b.label}
                  className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800/80 hover:border-cyan-500/40 transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-cyan-300 font-mono">
                        {b.aspectRatioLabel}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded">
                        Aspect: {b.aspectRatio.toFixed(2)}
                      </span>
                    </div>

                    {/* Miniature Aspect Box Visualizer */}
                    <div className="h-14 w-full bg-slate-900/60 rounded-lg flex items-center justify-center p-1 my-2 border border-slate-800/40">
                      <div
                        className="bg-cyan-500/20 border border-cyan-400/60 rounded flex items-center justify-center text-[10px] font-mono text-cyan-300 transition-all"
                        style={{
                          height: b.aspectRatio >= 1 ? '70%' : '90%',
                          aspectRatio: `${b.width} / ${b.height}`,
                        }}
                      >
                        {b.width}×{b.height}
                      </div>
                    </div>

                    <div className="text-[11px] font-mono text-slate-300 space-y-0.5">
                      <div>Spacetime Tokens: <strong className="text-white">{b.tokens}</strong></div>
                      <div>Scaled Batch Size: <strong className="text-cyan-400">{scaledBs} clips</strong></div>
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>Batch Tokens:</span>
                    <span className="text-emerald-400 font-bold">{totalTokensInBatch}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Image Buckets (T = 1) */}
        <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span>
              <h3 className="text-sm font-bold text-white">
                Image Buckets (T = 1 frame, k = 0)
              </h3>
            </div>
            <span className="text-[11px] font-mono text-amber-400 bg-amber-950/60 px-2 py-0.5 rounded border border-amber-800/40">
              Joint Texture Training (Mix: {(imageRatio * 100).toFixed(0)}%)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {imageBuckets.map((b) => {
              const scaledBs = computeBucketBatchSize(b, targetTokens);
              const totalTokensInBatch = scaledBs * b.tokens;
              return (
                <div
                  key={b.label}
                  className="bg-slate-950/80 p-3.5 rounded-xl border border-slate-800/80 hover:border-amber-500/40 transition-all flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-bold text-amber-300 font-mono">
                        {b.aspectRatioLabel}
                      </span>
                      <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded">
                        Aspect: {b.aspectRatio.toFixed(2)}
                      </span>
                    </div>

                    {/* Miniature Aspect Box Visualizer */}
                    <div className="h-14 w-full bg-slate-900/60 rounded-lg flex items-center justify-center p-1 my-2 border border-slate-800/40">
                      <div
                        className="bg-amber-500/20 border border-amber-400/60 rounded flex items-center justify-center text-[10px] font-mono text-amber-300 transition-all"
                        style={{
                          height: b.aspectRatio >= 1 ? '70%' : '90%',
                          aspectRatio: `${b.width} / ${b.height}`,
                        }}
                      >
                        {b.width}×{b.height}
                      </div>
                    </div>

                    <div className="text-[11px] font-mono text-slate-300 space-y-0.5">
                      <div>Spatial Tokens: <strong className="text-white">{b.tokens}</strong></div>
                      <div>Scaled Batch Size: <strong className="text-amber-400">{scaledBs} clips</strong></div>
                    </div>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono text-slate-400">
                    <span>Batch Tokens:</span>
                    <span className="text-emerald-400 font-bold">{totalTokensInBatch}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Interactive Nearest Bucket Assignment Playground */}
      <div className="bg-slate-900/90 p-5 rounded-2xl border border-slate-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Compass className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-white">
              Interactive Clip Nearest-Bucket Assignment Playground
            </h3>
          </div>
          <span className="text-xs text-slate-400">
            Tests Euclidean Aspect & Temporal distance metric
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Controls: Input Arbitrary Video / Image dimensions */}
          <div className="lg:col-span-5 space-y-3 bg-slate-950 p-4 rounded-xl border border-slate-800">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300">Media Type:</label>
              <div className="flex items-center gap-1 bg-slate-900 p-0.5 rounded-lg border border-slate-800">
                <button
                  onClick={() => setIsTestImage(false)}
                  className={`px-2 py-0.5 text-xs rounded ${
                    !isTestImage ? 'bg-cyan-500 text-slate-950 font-bold' : 'text-slate-400'
                  }`}
                >
                  Video
                </button>
                <button
                  onClick={() => setIsTestImage(true)}
                  className={`px-2 py-0.5 text-xs rounded ${
                    isTestImage ? 'bg-amber-500 text-slate-950 font-bold' : 'text-slate-400'
                  }`}
                >
                  Image (T=1)
                </button>
              </div>
            </div>

            {!isTestImage && (
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Raw Frames:</span>
                  <span className="font-mono text-white">{testFrames} frames</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max={Math.max(120, preset.baseFrames * 2)}
                  value={testFrames}
                  onChange={(e) => setTestFrames(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
                />
              </div>
            )}

            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Raw Width:</span>
                <span className="font-mono text-white">{testWidth} px</span>
              </div>
              <input
                type="range"
                min="32"
                max={preset.key === 'tiny' ? 128 : 512}
                value={testWidth}
                onChange={(e) => setTestWidth(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
              />
            </div>

            <div>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>Raw Height:</span>
                <span className="font-mono text-white">{testHeight} px</span>
              </div>
              <input
                type="range"
                min="32"
                max={preset.key === 'tiny' ? 128 : 512}
                value={testHeight}
                onChange={(e) => setTestHeight(Number(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none accent-cyan-400"
              />
            </div>

            <div className="pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-400 flex justify-between">
              <span>Input Aspect Ratio:</span>
              <span className="text-white font-bold">{(testWidth / Math.max(1, testHeight)).toFixed(2)}</span>
            </div>
          </div>

          {/* Transformation Arrow */}
          <div className="lg:col-span-2 flex flex-col items-center justify-center text-slate-500">
            <div className="p-2 rounded-full bg-slate-800/80 text-cyan-400">
              <ArrowRight className="w-5 h-5" />
            </div>
            <span className="text-[10px] font-mono mt-1 text-slate-400">assign_bucket()</span>
          </div>

          {/* Result Card */}
          <div className="lg:col-span-5 bg-gradient-to-br from-slate-950 via-slate-950 to-cyan-950/30 p-4 rounded-xl border border-cyan-500/40 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                Assigned Target Bucket
              </span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800/50">
                Multiple of 16
              </span>
            </div>

            <div className="text-lg font-bold text-white font-mono flex items-center gap-2">
              <span className="text-cyan-400">{assignedBucket.aspectRatioLabel}</span>
              <span>•</span>
              <span>{assignedBucket.width} × {assignedBucket.height}</span>
              <span>•</span>
              <span className="text-slate-400">{assignedBucket.frames} frames</span>
            </div>

            <div className="space-y-1 text-xs text-slate-300 font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Short Side Resize:</span>
                <span>min({testWidth}, {testHeight}) → scaled</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Spatial Crop:</span>
                <span>Random/Center to ({assignedBucket.height}, {assignedBucket.width})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Temporal Resample:</span>
                <span>16 FPS resample + Crop to {assignedBucket.frames} (T = 4k+1)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Assigned Token Count:</span>
                <span className="text-cyan-300 font-bold">{assignedBucket.tokens} tokens</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
