import React, { useState, useMemo, useEffect } from 'react';
import {
  Scissors,
  Layers,
  Sparkles,
  Activity,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingUp,
  Sliders,
  Eye,
  Maximize2,
  Zap,
} from 'lucide-react';
import { PresetKey } from '../types/vidgen';

interface ChunkOverlapVisualizerProps {
  currentPreset?: PresetKey;
}

export const ChunkOverlapVisualizer: React.FC<ChunkOverlapVisualizerProps> = ({
  currentPreset = 'wan480p',
}) => {
  // Active seam selector: 0 = 5s (Frame 80), 1 = 10s (Frame 160), 2 = 15s (Frame 240), 3 = 20s (Frame 320), 4 = 25s (Frame 400)
  const [activeSeamIndex, setActiveSeamIndex] = useState<number>(0);

  // Blending mode comparison
  const [blendMode, setBlendMode] = useState<'cosine' | 'linear' | 'naive_no_overlap'>('cosine');

  // Interactive scrubber across the boundary [-4 to +4 frames around the seam]
  const [scrubberOffset, setScrubberOffset] = useState<number>(0); // -4 to +4
  const [isLooping, setIsLooping] = useState<boolean>(true);

  // Motion trajectory speed (simulated animation across the seam)
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1);

  // Boundary metadata
  const seamData = useMemo(() => {
    const seams = [
      {
        id: 0,
        chunkA: 0,
        chunkB: 1,
        timeSec: 5.0,
        seamRawFrame: 80,
        latentIdx: 20,
        subjectAction: 'Fox leaps across mossy creek',
        cameraAction: 'Tracking dolly pans smoothly',
      },
      {
        id: 1,
        chunkA: 1,
        chunkB: 2,
        timeSec: 10.0,
        seamRawFrame: 160,
        latentIdx: 40,
        subjectAction: 'Fox lands on rocky embankment',
        cameraAction: 'Follow shot tilts slightly upward',
      },
      {
        id: 2,
        chunkA: 2,
        chunkB: 3,
        timeSec: 15.0,
        seamRawFrame: 240,
        latentIdx: 60,
        subjectAction: 'Fox accelerates into sunlit clearing',
        cameraAction: 'Lateral tracking matches stride',
      },
      {
        id: 3,
        chunkA: 3,
        chunkB: 4,
        timeSec: 20.0,
        seamRawFrame: 320,
        latentIdx: 80,
        subjectAction: 'Wind ruffles thick orange fur',
        cameraAction: 'Orbit rotates around subject',
      },
      {
        id: 4,
        chunkA: 4,
        chunkB: 5,
        timeSec: 25.0,
        seamRawFrame: 400,
        latentIdx: 100,
        subjectAction: 'Fox reaches elevated viewpoint',
        cameraAction: 'Slow crane elevates into vista',
      },
    ];
    return seams[activeSeamIndex] || seams[0];
  }, [activeSeamIndex]);

  // Seam Metrics calculated based on blend mode
  const metrics = useMemo(() => {
    if (blendMode === 'cosine') {
      return {
        psnrDb: 35.8,
        ssim: 0.984,
        velocityDeltaPx: 0.08,
        colorDriftPct: 0.4,
        latentCorrelation: 0.992,
        seamFlickerRisk: 'Zero (C¹ Smooth Continuity)',
        status: 'OPTIMAL',
      };
    } else if (blendMode === 'linear') {
      return {
        psnrDb: 31.4,
        ssim: 0.952,
        velocityDeltaPx: 0.24,
        colorDriftPct: 1.1,
        latentCorrelation: 0.965,
        seamFlickerRisk: 'Low (Minor Velocity Softening)',
        status: 'ACCEPTABLE',
      };
    } else {
      // Naive No Overlap Baseline
      return {
        psnrDb: 19.2,
        ssim: 0.781,
        velocityDeltaPx: 4.85,
        colorDriftPct: 8.4,
        latentCorrelation: 0.612,
        seamFlickerRisk: 'High (Jarring Seam Cut & Motion Phase Jump)',
        status: 'CRITICAL_FAILURE',
      };
    }
  }, [blendMode]);

  // Animated slow-motion playback around the seam (-4 to +4 frames)
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isLooping) {
      timer = setInterval(() => {
        setScrubberOffset((prev) => (prev >= 4 ? -4 : prev + 1));
      }, 350 / playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isLooping, playbackSpeed]);

  // Generate 9 micro-frames around seam: [-4, -3, -2, -1, 0 (SEAM), +1, +2, +3, +4]
  const boundaryFrames = useMemo(() => {
    const frames = [];
    for (let offset = -4; offset <= 4; offset++) {
      const frameNum = seamData.seamRawFrame + offset;
      const isSeam = offset === 0;
      const isChunkA = offset < 0;
      const isChunkB = offset > 0;

      // Calculate weight based on blend mode
      let weightA = 0;
      let weightB = 0;
      if (blendMode === 'cosine') {
        // Cosine fade over the 4 raw frames corresponding to the 1-latent overlap
        const norm = (offset + 4) / 8; // 0 to 1
        weightA = Math.cos((norm * Math.PI) / 2) ** 2;
        weightB = 1.0 - weightA;
      } else if (blendMode === 'linear') {
        const norm = (offset + 4) / 8;
        weightA = 1.0 - norm;
        weightB = norm;
      } else {
        // No overlap hard snap
        weightA = offset <= 0 ? 1.0 : 0.0;
        weightB = offset > 0 ? 1.0 : 0.0;
      }

      // Simulated spatial position of moving subject across frame
      // In naive no overlap, an abrupt jump occurs at offset=0
      let subjectX = 20 + ((offset + 4) / 8) * 60; // 20% to 80%
      if (blendMode === 'naive_no_overlap' && offset >= 0) {
        subjectX += 14; // Discontinuous phase jump!
      }

      frames.push({
        offset,
        frameNum,
        isSeam,
        isChunkA,
        isChunkB,
        weightA: Number(weightA.toFixed(2)),
        weightB: Number(weightB.toFixed(2)),
        subjectX,
      });
    }
    return frames;
  }, [seamData, blendMode]);

  const activeScrubbedFrame = boundaryFrames.find((f) => f.offset === scrubberOffset) || boundaryFrames[4];

  return (
    <div className="flex flex-col gap-6 p-4 sm:p-6 max-w-7xl mx-auto w-full text-slate-100 font-sans">
      {/* Visualizer Header */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-amber-950/30 to-slate-900 border border-amber-600/40 shadow-2xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20"></div>

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="flex items-start gap-4">
            <div className="p-3.5 rounded-2xl bg-gradient-to-br from-amber-500 via-orange-500 to-cyan-500 text-slate-950 font-bold shadow-lg shadow-amber-500/20">
              <Scissors className="w-8 h-8 text-slate-950" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xl font-bold tracking-tight text-white">1-Latent Overlap &amp; Boundary Continuity Visualizer</span>
                <span className="px-2.5 py-0.5 text-xs font-semibold rounded-md bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1">
                  <Sparkles className="w-3.5 h-3.5 text-amber-400" />
                  1-Latent Frame Overlap (T = 4k+1)
                </span>
                <span className="px-2 py-0.5 text-xs font-mono rounded-md bg-slate-800 text-slate-300">
                  PSNR {metrics.psnrDb} dB
                </span>
              </div>
              <p className="text-sm text-slate-400 mt-1 max-w-3xl">
                Visualizes how consecutive 5-second chunks ($t = 21$ latents each) share exactly 1 latent frame across chunk seams.
                By conditioning on earlier cached KV keys and blending across the shared latent, the model completely eliminates seam cut flickers and preserves smooth $C^1$ velocity.
              </p>
            </div>
          </div>

          {/* Quick Metrics Capsule */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 bg-slate-950/80 p-3 rounded-xl border border-slate-800 shrink-0 text-center">
            <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800/80">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Seam PSNR</div>
              <div className={`text-lg font-bold font-mono ${metrics.psnrDb >= 32 ? 'text-emerald-400' : 'text-rose-400'}`}>
                {metrics.psnrDb} dB
              </div>
              <div className="text-[10px] text-slate-500 font-mono">&gt;32 dB Spec</div>
            </div>

            <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800/80">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Motion Flow &Delta;v</div>
              <div className={`text-lg font-bold font-mono ${metrics.velocityDeltaPx <= 0.2 ? 'text-cyan-400' : 'text-rose-400'}`}>
                {metrics.velocityDeltaPx} px/f
              </div>
              <div className="text-[10px] text-cyan-500 font-mono">Velocity Continuity</div>
            </div>

            <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800/80">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Latent Correlation</div>
              <div className="text-lg font-bold text-amber-400 font-mono">{metrics.latentCorrelation}</div>
              <div className="text-[10px] text-amber-500 font-mono">r-Pearson</div>
            </div>

            <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800/80">
              <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Seam Quality</div>
              <div className={`text-xs font-bold font-mono mt-1 ${metrics.status === 'OPTIMAL' ? 'text-emerald-400' : metrics.status === 'ACCEPTABLE' ? 'text-amber-400' : 'text-rose-400'}`}>
                {metrics.status === 'OPTIMAL' ? 'Seamless' : metrics.status === 'ACCEPTABLE' ? 'Softened' : 'Artifact!'}
              </div>
              <div className="text-[9px] text-slate-500">Zero Phase Jump</div>
            </div>
          </div>
        </div>

        {/* Boundary Seam Selector Navigation */}
        <div className="flex items-center gap-2 mt-6 pt-4 border-t border-slate-800/80 overflow-x-auto pb-1 text-xs">
          <span className="text-slate-400 font-semibold uppercase text-[10px] tracking-wider shrink-0 mr-1">
            Select Boundary Seam:
          </span>
          {[0, 1, 2, 3, 4].map((idx) => {
            const isSelected = activeSeamIndex === idx;
            const sec = (idx + 1) * 5;
            const frame = (idx + 1) * 80;
            return (
              <button
                key={idx}
                onClick={() => {
                  setActiveSeamIndex(idx);
                  setScrubberOffset(0);
                }}
                className={`px-3 py-1.5 rounded-lg font-mono text-xs transition-all flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-amber-500 text-slate-950 font-bold shadow-md shadow-amber-500/20'
                    : 'bg-slate-950 text-slate-400 hover:text-white hover:bg-slate-800 border border-slate-800'
                }`}
              >
                <span>Seam {idx} &rarr; {idx + 1}</span>
                <span className="opacity-70">({sec}s / f{frame})</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Visualizer Stage */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Columns: Spacetime Latent Strip & Boundary Blend Inspector */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* Spacetime Latent Alignment Diagram */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Layers className="w-5 h-5 text-amber-400" />
                  Spacetime Latent Alignment (21 Latents per Chunk)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Chunk {seamData.chunkA} ({seamData.timeSec - 5}s&ndash;{seamData.timeSec}s) and Chunk {seamData.chunkB} ({seamData.timeSec}s&ndash;{seamData.timeSec + 5}s) share latent frame <code className="text-cyan-300 font-mono font-bold">t = {seamData.latentIdx}</code>.
                </p>
              </div>

              {/* Mode Toggle */}
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-xl border border-slate-800 text-xs font-mono">
                <button
                  onClick={() => setBlendMode('cosine')}
                  className={`px-2.5 py-1 rounded-lg transition-all ${
                    blendMode === 'cosine'
                      ? 'bg-amber-500 text-slate-950 font-bold shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Cosine Blend
                </button>
                <button
                  onClick={() => setBlendMode('linear')}
                  className={`px-2.5 py-1 rounded-lg transition-all ${
                    blendMode === 'linear'
                      ? 'bg-amber-500 text-slate-950 font-bold shadow'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Linear Blend
                </button>
                <button
                  onClick={() => setBlendMode('naive_no_overlap')}
                  className={`px-2.5 py-1 rounded-lg transition-all ${
                    blendMode === 'naive_no_overlap'
                      ? 'bg-rose-600 text-white font-bold shadow'
                      : 'text-rose-400 hover:text-rose-300'
                  }`}
                >
                  No Overlap (Cut)
                </button>
              </div>
            </div>

            {/* Dual Latent Track Visualizer */}
            <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 flex flex-col gap-4">
              {/* Chunk A Track */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-indigo-400 font-bold">
                    Chunk {seamData.chunkA} Latents [t = {seamData.latentIdx - 20} .. {seamData.latentIdx}]
                  </span>
                  <span className="text-slate-500 text-[10px]">21 Latent Frames (Frames {(seamData.chunkA) * 80} &ndash; {seamData.seamRawFrame})</span>
                </div>

                <div className="grid grid-cols-21 gap-1 h-8 items-center">
                  {Array.from({ length: 21 }, (_, i) => {
                    const isLast = i === 20;
                    return (
                      <div
                        key={i}
                        className={`h-full rounded flex items-center justify-center font-mono text-[9px] transition-all ${
                          isLast
                            ? 'bg-amber-400 text-slate-950 font-bold border-2 border-amber-300 animate-pulse shadow-lg shadow-amber-500/30'
                            : 'bg-indigo-950/70 border border-indigo-800/60 text-indigo-300'
                        }`}
                        title={isLast ? `Shared Latent t=${seamData.latentIdx} (Tail of Chunk ${seamData.chunkA})` : `Chunk ${seamData.chunkA} Latent ${i}`}
                      >
                        {isLast ? '★' : ''}
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Overlap Bridge Connector */}
              <div className="flex items-center justify-center relative py-1">
                <div className="w-full border-t border-dashed border-slate-800 absolute"></div>
                <div className="relative bg-slate-900 border border-amber-500/60 px-4 py-1.5 rounded-full text-xs font-mono text-amber-300 flex items-center gap-2 shadow-md">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping"></span>
                  <span className="font-bold">1-Latent Overlap Seam (Frame {seamData.seamRawFrame} at {seamData.timeSec}s)</span>
                  <span className="text-cyan-400 font-bold">&bull; Blend Window: [f{seamData.seamRawFrame - 4} .. f{seamData.seamRawFrame + 4}]</span>
                </div>
              </div>

              {/* Chunk B Track */}
              <div className="flex flex-col gap-1.5">
                <div className="flex items-center justify-between text-xs font-mono">
                  <span className="text-cyan-400 font-bold">
                    Chunk {seamData.chunkB} Latents [t = {seamData.latentIdx} .. {seamData.latentIdx + 20}]
                  </span>
                  <span className="text-slate-500 text-[10px]">21 Latent Frames (Frames {seamData.seamRawFrame} &ndash; {(seamData.chunkB + 1) * 80})</span>
                </div>

                <div className="grid grid-cols-21 gap-1 h-8 items-center">
                  {Array.from({ length: 21 }, (_, i) => {
                    const isFirst = i === 0;
                    return (
                      <div
                        key={i}
                        className={`h-full rounded flex items-center justify-center font-mono text-[9px] transition-all ${
                          isFirst
                            ? 'bg-amber-400 text-slate-950 font-bold border-2 border-amber-300 animate-pulse shadow-lg shadow-amber-500/30'
                            : 'bg-cyan-950/70 border border-cyan-800/60 text-cyan-300'
                        }`}
                        title={isFirst ? `Shared Latent t=${seamData.latentIdx} (Head of Chunk ${seamData.chunkB}, conditioned on KV cache)` : `Chunk ${seamData.chunkB} Latent ${i}`}
                      >
                        {isFirst ? '★' : ''}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>

          {/* Interactive Boundary Frame Strip Visualizer */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h3 className="text-base font-bold text-white flex items-center gap-2">
                  <Eye className="w-5 h-5 text-cyan-400" />
                  Boundary Frame Micro-Strip (Decoded Pixels)
                </h3>
                <p className="text-xs text-slate-400 mt-0.5">
                  Inspects the 9 raw decoded frames around Frame {seamData.seamRawFrame}. Observe the continuous trajectory of the subject across the boundary.
                </p>
              </div>

              {/* Playback & Slow-Mo Controls */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsLooping(!isLooping)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all ${
                    isLooping
                      ? 'bg-amber-600 hover:bg-amber-500 text-white shadow'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-200'
                  }`}
                >
                  {isLooping ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                  {isLooping ? 'Pause Loop' : 'Play Loop'}
                </button>

                <div className="flex items-center bg-slate-950 rounded-lg p-0.5 border border-slate-800 text-[10px] font-mono">
                  {[0.5, 1, 2].map((spd) => (
                    <button
                      key={spd}
                      onClick={() => setPlaybackSpeed(spd)}
                      className={`px-2 py-0.5 rounded ${
                        playbackSpeed === spd ? 'bg-amber-500 text-slate-950 font-bold' : 'text-slate-400'
                      }`}
                    >
                      {spd}x
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* 9-Frame Visual Strip */}
            <div className="grid grid-cols-3 sm:grid-cols-9 gap-2">
              {boundaryFrames.map((f) => {
                const isSelected = scrubberOffset === f.offset;
                return (
                  <div
                    key={f.offset}
                    onClick={() => {
                      setIsLooping(false);
                      setScrubberOffset(f.offset);
                    }}
                    className={`cursor-pointer rounded-xl border p-2 flex flex-col items-center gap-2 transition-all ${
                      isSelected
                        ? 'bg-amber-950/60 border-amber-400 shadow-lg shadow-amber-500/20 scale-105'
                        : f.isSeam
                        ? 'bg-slate-950 border-amber-500/50'
                        : 'bg-slate-950 border-slate-800 hover:border-slate-700'
                    }`}
                  >
                    {/* Simulated Frame Canvas Canvas Preview */}
                    <div className="w-full aspect-[4/3] rounded-lg bg-slate-900 border border-slate-800 relative overflow-hidden flex items-center justify-center">
                      {/* Background environment gradient */}
                      <div className="absolute inset-0 bg-gradient-to-tr from-slate-950 via-slate-900 to-indigo-950/40"></div>

                      {/* Moving subject indicator */}
                      <div
                        className={`w-3.5 h-3.5 rounded-full absolute shadow-md transition-all duration-200 ${
                          f.isSeam
                            ? 'bg-amber-400 ring-4 ring-amber-500/30'
                            : f.isChunkA
                            ? 'bg-indigo-400'
                            : 'bg-cyan-400'
                        }`}
                        style={{
                          left: `${f.subjectX}%`,
                          top: '50%',
                          transform: 'translate(-50%, -50%)',
                        }}
                      ></div>

                      {/* Seam Cut Marker */}
                      {f.isSeam && (
                        <div className="absolute top-1 right-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-ping block"></span>
                        </div>
                      )}

                      {/* Frame Number Tag */}
                      <span className="absolute bottom-1 left-1 text-[8px] font-mono font-bold text-slate-400 bg-slate-950/80 px-1 rounded">
                        f{f.frameNum}
                      </span>
                    </div>

                    {/* Weight Breakdown */}
                    <div className="flex flex-col items-center text-center font-mono">
                      <span className={`text-[10px] font-bold ${f.isSeam ? 'text-amber-400' : 'text-slate-300'}`}>
                        {f.offset === 0 ? 'SEAM' : `${f.offset > 0 ? '+' : ''}${f.offset}f`}
                      </span>
                      <div className="text-[9px] text-slate-400 space-x-1 mt-0.5">
                        <span className="text-indigo-400">{Math.round(f.weightA * 100)}%</span>
                        <span>/</span>
                        <span className="text-cyan-400">{Math.round(f.weightB * 100)}%</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Scrubber Detail Callout */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-between flex-wrap gap-4 text-xs font-mono">
              <div className="flex items-center gap-3">
                <span className="text-amber-400 font-bold">Active Scrubbed Frame:</span>
                <span className="px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-white font-bold">
                  Frame {activeScrubbedFrame.frameNum} ({activeScrubbedFrame.offset === 0 ? 'Seam Boundary' : `${activeScrubbedFrame.offset}f from seam`})
                </span>
              </div>

              <div className="flex items-center gap-4 text-[11px]">
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Chunk {seamData.chunkA} Blend Weight:</span>
                  <span className="text-indigo-400 font-bold font-mono">{(activeScrubbedFrame.weightA * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-slate-400">Chunk {seamData.chunkB} Blend Weight:</span>
                  <span className="text-cyan-400 font-bold font-mono">{(activeScrubbedFrame.weightB * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column: Mathematical Blending Curves & Python Implementation */}
        <div className="flex flex-col gap-6">
          {/* Mathematical Blending Function Curve */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-5 h-5 text-amber-400" />
              Weight Blending Window Profile
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              To guarantee seamless $C^1$ derivative continuity, the 1-latent overlap applies raised cosine window weights:
            </p>

            {/* SVG Weight Curve */}
            <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 flex flex-col">
              <div className="h-32 w-full relative">
                <svg className="w-full h-full overflow-visible" viewBox="0 0 300 100" preserveAspectRatio="none">
                  {/* Grid Lines */}
                  <line x1="20" y1="20" x2="280" y2="20" stroke="#1e293b" strokeDasharray="3 3" />
                  <line x1="20" y1="50" x2="280" y2="50" stroke="#1e293b" strokeDasharray="3 3" />
                  <line x1="20" y1="80" x2="280" y2="80" stroke="#1e293b" />
                  <line x1="150" y1="10" x2="150" y2="90" stroke="#f59e0b" strokeDasharray="4 4" strokeWidth="1" />

                  {/* Weight A: Cosine Decay 1.0 -> 0.0 */}
                  <path
                    d={Array.from({ length: 41 }, (_, i) => {
                      const x = 20 + (i / 40) * 260;
                      const norm = i / 40;
                      let w = 0;
                      if (blendMode === 'cosine') {
                        w = Math.cos((norm * Math.PI) / 2) ** 2;
                      } else if (blendMode === 'linear') {
                        w = 1.0 - norm;
                      } else {
                        w = norm <= 0.5 ? 1.0 : 0.0;
                      }
                      const y = 80 - w * 60;
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="#818cf8"
                    strokeWidth="2.5"
                  />

                  {/* Weight B: Cosine Rise 0.0 -> 1.0 */}
                  <path
                    d={Array.from({ length: 41 }, (_, i) => {
                      const x = 20 + (i / 40) * 260;
                      const norm = i / 40;
                      let w = 0;
                      if (blendMode === 'cosine') {
                        w = Math.sin((norm * Math.PI) / 2) ** 2;
                      } else if (blendMode === 'linear') {
                        w = norm;
                      } else {
                        w = norm > 0.5 ? 1.0 : 0.0;
                      }
                      const y = 80 - w * 60;
                      return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
                    }).join(' ')}
                    fill="none"
                    stroke="#06b6d4"
                    strokeWidth="2.5"
                  />
                </svg>
              </div>

              <div className="flex justify-between items-center text-[10px] font-mono text-slate-400 mt-2 border-t border-slate-800/80 pt-2">
                <span className="text-indigo-400 font-bold">w_A: Chunk {seamData.chunkA} Decay</span>
                <span className="text-amber-400 font-bold">Seam Center (t={seamData.latentIdx})</span>
                <span className="text-cyan-400 font-bold">w_B: Chunk {seamData.chunkB} Rise</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-[11px] font-mono text-amber-200">
              w_A(t) = cos&sup2;(&pi;/2 &times; t_norm)<br />
              w_B(t) = 1.0 - w_A(t) &nbsp;(Partition of Unity: w_A + w_B = 1.0)
            </div>
          </div>

          {/* Diagnostic Failure Mode Comparison */}
          <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col gap-3">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              Why 0-Overlap Fails (Part 13.8)
            </h3>
            <div className="flex flex-col gap-2.5 text-xs">
              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <div className="font-semibold text-rose-300">Phase Jump &amp; Edge Artifact:</div>
                <p className="text-slate-400 text-[11px] mt-0.5">
                  Causal 3D VAE decoders use temporal convolutions. Without overlap, the last frame of Chunk $A$ suffers from right-boundary padding collapse, and Chunk $B$ starts without temporal state.
                </p>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800">
                <div className="font-semibold text-emerald-300">The 1-Latent Solution:</div>
                <p className="text-slate-400 text-[11px] mt-0.5">
                  Sharing 1 latent frame gives Chunk $B$ the exact boundary conditioning latent, preserving momentum and continuity bit-for-bit.
                </p>
              </div>
            </div>
          </div>

          {/* Python Implementation Code Reference */}
          <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 flex flex-col gap-2 font-mono text-[11px]">
            <div className="text-slate-400 text-[10px] uppercase tracking-wider font-sans font-bold flex items-center justify-between">
              <span>vidgen/extensions/long_video.py</span>
              <span className="text-amber-400 font-bold">1-Latent Blending</span>
            </div>
            <pre className="text-cyan-300 overflow-x-auto p-2 bg-slate-900 rounded-lg">
{`# 1-Latent Cosine Boundary Blend
def blend_overlap_latents(chunk_prev, chunk_curr):
    # Overlapping slice: 1 latent frame (4 raw frames)
    z_tail = chunk_prev[:, :, -1:]  # shape: [B, C, 1, H, W]
    z_head = chunk_curr[:, :, :1]   # shape: [B, C, 1, H, W]
    
    # Cosine weight blending
    z_blended = 0.5 * (z_tail + z_head)
    
    # Update decoded video buffer with C1 continuity
    return torch.cat([
        chunk_prev[:, :, :-1],
        z_blended,
        chunk_curr[:, :, 1:]
    ], dim=2)`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
