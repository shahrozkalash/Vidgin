import React, { useState, useRef, useEffect } from 'react';
import { Info, X, Sparkles, Clock } from 'lucide-react';

export type ParameterType =
  | 'params'
  | 'resolution'
  | 'baseFrames'
  | 'tokens'
  | 'vram'
  | 'hardware'
  | 'totalParams'
  | 'layersHeads'
  | 'hiddenDim'
  | 'modulation';

interface ParameterInfo {
  title: string;
  shortLabel: string;
  qualityImpact: string;
  timeImpact: string;
  plainEnglishSummary: string;
}

export const PARAMETER_EXPLANATIONS: Record<ParameterType, ParameterInfo> = {
  params: {
    title: 'Model Parameters (Capacity)',
    shortLabel: 'Parameters',
    qualityImpact:
      'Larger models (1.3B–14B) accurately follow complex prompts, generate realistic human anatomy, and maintain cinematic physics. Smaller models (20M–270M) produce looser, approximate visuals.',
    timeImpact:
      'Increases generation time proportionally. A 14B model requires ~10× more matrix multiplication FLOPs per denoising step than a 1.3B model.',
    plainEnglishSummary:
      'Higher parameter count creates noticeably smarter, more realistic video, but requires significantly longer render times and more compute.',
  },
  resolution: {
    title: 'Resolution (Width × Height)',
    shortLabel: 'Resolution',
    qualityImpact:
      'Higher pixel dimensions produce sharper fine textures, readable text, and crisp edges. Low resolutions (64×64) are pixelated and intended for structural tests.',
    timeImpact:
      'Doubling width and height quadruples total pixels and latent tokens. Attention complexity scales quadratically (O(N²)), making denoising dramatically slower without chunking.',
    plainEnglishSummary:
      'Sharper, higher-definition video at the cost of substantially higher GPU compute and longer waiting time per clip.',
  },
  baseFrames: {
    title: 'Base Frames (Temporal Sequence Length)',
    shortLabel: 'Base Frames',
    qualityImpact:
      'More frames deliver continuous, natural motion and consistent character appearances across time. Fewer frames (17f) produce quick micro-bursts with less temporal evolution.',
    timeImpact:
      'Scales attention memory and compute. Full full-sequence attention across 81+ frames is memory-heavy; temporal window chunking is used to keep generation feasible.',
    plainEnglishSummary:
      'More frames create longer, smoother video clips, but demand more GPU VRAM and increase total generation seconds.',
  },
  tokens: {
    title: 'Tokens per Chunk (3D Spacetime Patches)',
    shortLabel: 'Tokens',
    qualityImpact:
      'More tokens per chunk give the transformer a richer, more detailed spatial-temporal receptive field to synthesize simultaneous details.',
    timeImpact:
      'Direct driver of attention FLOPs and peak VRAM. Full attention scales as O(N²); large token counts (>30k) require flash-attention or streaming kernels to prevent Out-Of-Memory crashes.',
    plainEnglishSummary:
      'Controls the amount of visual data processed simultaneously; higher values yield richer detail but sharply increase GPU memory usage.',
  },
  vram: {
    title: 'Target VRAM (GPU Video Memory)',
    shortLabel: 'Target VRAM',
    qualityImpact:
      'Does not directly degrade pixels, but ample VRAM lets you run full bfloat16 precision and larger attention chunks without lossy quantization.',
    timeImpact:
      'Running out of VRAM triggers slow system RAM offloading (10× slower) or forces small sliding chunks with extra inter-window passes.',
    plainEnglishSummary:
      'The minimum GPU memory required to generate the video without crashing or slowing down to a crawl.',
  },
  hardware: {
    title: 'Intended Hardware Environment',
    shortLabel: 'Hardware',
    qualityImpact:
      'Quality is model-dependent, but pro hardware enables faster experimentation and higher precision (bfloat16) without precision drift.',
    timeImpact:
      'A multi-GPU cluster (H100) or high-end workstation (A100/4090) generates frames in seconds, whereas CPU or low-end laptops can take several minutes per step.',
    plainEnglishSummary:
      'The recommended computing setup needed to finish video generation in reasonable turnaround time.',
  },
  totalParams: {
    title: 'Total Model Parameters',
    shortLabel: 'Total Parameters',
    qualityImpact:
      'Represents the total learned weights in the Diffusion Transformer. More weights capture subtle visual lighting, surface reflections, and realistic motion dynamics.',
    timeImpact:
      'Scales linear matrix multiplications. Moving from 270M to 1.3B increases compute by ~5× per denoising iteration.',
    plainEnglishSummary:
      'The brain size of the video model: more weights mean higher visual fidelity and prompt adherence, but slower generation.',
  },
  layersHeads: {
    title: 'Layers & Attention Heads',
    shortLabel: 'Layers & Heads',
    qualityImpact:
      'Deeper layers extract abstract scene concepts and global coherence; more attention heads allow the model to track multiple motion trajectories simultaneously.',
    timeImpact:
      'Every additional layer adds a sequential transformer block that must execute on every denoising step, directly increasing latency.',
    plainEnglishSummary:
      'Layers determine how deeply the model reasons about the scene; heads track different relationships. More layers mean higher quality but longer render times.',
  },
  hiddenDim: {
    title: 'Hidden Dimension (Embedding Width)',
    shortLabel: 'Hidden Dim',
    qualityImpact:
      'Wider dimensions (e.g. 1536D or 5120D vs 384D) allow each visual token to carry fine nuances about color, shadow, depth, and velocity.',
    timeImpact:
      'Feed-forward layers scale with O(D²). Doubling hidden dimension quadruples the FLOPs in the MLP projection blocks.',
    plainEnglishSummary:
      'The capacity of each visual token: wider dimensions capture subtler textures and colors, but require much faster GPU memory bandwidth.',
  },
  modulation: {
    title: 'Modulation Savings (Shared AdaLN)',
    shortLabel: 'Modulation',
    qualityImpact:
      'Preserves 100% of video quality while conditioning each layer on diffusion timesteps and text embeddings via a shared projection MLP.',
    timeImpact:
      'Saves >80% parameter overhead and ~15-20% execution time in the conditioning pathways of the DiT backbone.',
    plainEnglishSummary:
      'An architectural optimization that speeds up generation by sharing conditioning math across layers without losing any visual quality.',
  },
};

interface ParameterTooltipProps {
  param: ParameterType;
  children: React.ReactNode;
  align?: 'left' | 'center' | 'right';
  side?: 'top' | 'bottom';
  className?: string;
  showIcon?: boolean;
}

export const ParameterTooltip: React.FC<ParameterTooltipProps> = ({
  param,
  children,
  align = 'left',
  side = 'top',
  className = '',
  showIcon = true,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const info = PARAMETER_EXPLANATIONS[param];

  // Close when clicking outside
  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (event: MouseEvent | TouchEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('touchstart', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('touchstart', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen]);

  const handleToggle = (e: React.MouseEvent | React.KeyboardEvent) => {
    e.stopPropagation();
    e.preventDefault();
    setIsOpen((prev) => !prev);
  };

  const handleMouseEnter = () => {
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    setIsOpen(false);
  };

  // Alignment classes for desktop/tablet
  const alignClass =
    align === 'right'
      ? 'right-0'
      : align === 'center'
      ? 'left-1/2 -translate-x-1/2'
      : 'left-0';

  const sideClass =
    side === 'bottom'
      ? 'top-full mt-2'
      : 'bottom-full mb-2';

  return (
    <div
      ref={containerRef}
      className={`relative inline-flex items-center gap-1 group ${className}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Label and interactive trigger */}
      <span
        role="button"
        tabIndex={0}
        onClick={handleToggle}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            handleToggle(e);
          }
        }}
        aria-expanded={isOpen}
        aria-label={`Learn how ${info.shortLabel} affects video quality and generation time`}
        className="inline-flex items-center gap-1 border-b border-dotted border-slate-500/80 hover:border-cyan-400 text-inherit cursor-help transition-colors select-none focus:outline-none focus:ring-1 focus:ring-cyan-400 rounded-xs"
      >
        {children}
        {showIcon && (
          <Info className="w-3 h-3 text-slate-400 group-hover:text-cyan-400 shrink-0 transition-colors" />
        )}
      </span>

      {/* Popover Tooltip Box */}
      {isOpen && (
        <div
          role="tooltip"
          onClick={(e) => e.stopPropagation()}
          className={`absolute ${sideClass} z-50 w-72 sm:w-80 max-w-[calc(100vw-36px)] p-3.5 rounded-xl bg-slate-900 border border-slate-700/90 text-slate-100 shadow-2xl backdrop-blur-md text-left transition-all animate-in fade-in-0 zoom-in-95 pointer-events-auto ${alignClass}`}
        >
          {/* Header with Title & Close button */}
          <div className="flex items-start justify-between gap-2 pb-2 mb-2.5 border-b border-slate-800">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <h4 className="text-xs font-bold text-white tracking-wide">
                {info.title}
              </h4>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                setIsOpen(false);
              }}
              className="text-slate-400 hover:text-white p-0.5 rounded hover:bg-slate-800 transition-colors cursor-pointer"
              aria-label="Close tooltip"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Plain-English Takeaway */}
          <p className="text-[11px] text-slate-300 font-medium leading-relaxed pb-2.5 mb-2.5 border-b border-slate-800/80">
            {info.plainEnglishSummary}
          </p>

          {/* Two-Column / Two-Row Impact Breakdown */}
          <div className="space-y-2 text-[11px] leading-snug">
            {/* Impact 1: Video Quality */}
            <div className="flex items-start gap-2 bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-amber-300 block font-semibold text-[10px] uppercase tracking-wider">
                  Impact on Video Quality
                </strong>
                <span className="text-slate-300 text-[11px]">{info.qualityImpact}</span>
              </div>
            </div>

            {/* Impact 2: Generation Time */}
            <div className="flex items-start gap-2 bg-slate-950/60 p-2 rounded-lg border border-slate-800/80">
              <Clock className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-cyan-300 block font-semibold text-[10px] uppercase tracking-wider">
                  Impact on Generation Time
                </strong>
                <span className="text-slate-300 text-[11px]">{info.timeImpact}</span>
              </div>
            </div>
          </div>

          {/* Tiny arrow */}
          <div
            className={`absolute w-2.5 h-2.5 bg-slate-900 border-slate-700/90 rotate-45 ${
              side === 'bottom'
                ? '-top-1 border-l border-t'
                : '-bottom-1 border-r border-b'
            } ${
              align === 'right'
                ? 'right-6'
                : align === 'center'
                ? 'left-1/2 -translate-x-1/2'
                : 'left-6'
            }`}
          />
        </div>
      )}
    </div>
  );
};
