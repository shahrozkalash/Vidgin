import React, { useState } from 'react';
import {
  Sliders,
  Sparkles,
  Link2,
  Unlink,
  Scissors,
  Trash2,
  Clock,
  Dice5,
  Copy,
  CheckCircle2,
  ShieldCheck,
  AlertTriangle,
  Film
} from 'lucide-react';
import { EditorClip } from '../../types/editor';

interface EditorInspectorProps {
  selectedClip: EditorClip | null;
  onUpdateClipPrompt: (clipId: string, prompt: string) => void;
  onSplitClip: (clipId: string) => void;
  onDeleteClip: (clipId: string) => void;
  fps?: number;
}

export const EditorInspector: React.FC<EditorInspectorProps> = ({
  selectedClip,
  onUpdateClipPrompt,
  onSplitClip,
  onDeleteClip,
  fps = 16,
}) => {
  const [isCopiedSeed, setIsCopiedSeed] = useState(false);

  if (!selectedClip) {
    return (
      <div className="w-80 md:w-84 bg-slate-900 border-l border-slate-800 flex flex-col h-full shrink-0 select-none p-5 overflow-y-auto">
        <div className="flex items-center gap-2 pb-3 border-b border-slate-800 text-slate-300 font-semibold text-xs">
          <Sliders className="w-4 h-4 text-slate-400" />
          <span>Clip Inspector</span>
        </div>

        <div className="flex-1 flex flex-col items-center justify-center text-center p-6 space-y-3">
          <div className="w-12 h-12 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-600">
            <Film className="w-6 h-6 stroke-[1.5]" />
          </div>
          <div className="space-y-1">
            <h4 className="text-xs font-semibold text-slate-300">No Clip Selected</h4>
            <p className="text-[11px] text-slate-500 leading-relaxed max-w-[220px]">
              Click any clip on the timeline below to inspect its diffusion prompt, seed, continuity metrics, and timing parameters.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const handleCopySeed = () => {
    navigator.clipboard.writeText(selectedClip.seed.toString());
    setIsCopiedSeed(true);
    setTimeout(() => setIsCopiedSeed(false), 1500);
  };

  const framesCount = Math.floor(selectedClip.duration * fps) + 1;

  return (
    <div className="w-80 md:w-84 bg-slate-900 border-l border-slate-800 flex flex-col h-full shrink-0 select-none overflow-y-auto">
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-semibold text-slate-200">Clip Inspector</h3>
        </div>
        <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
          Track {selectedClip.trackIndex === 0 ? 'Main' : `Branch ${selectedClip.trackIndex}`}
        </span>
      </div>

      <div className="p-4 space-y-4">
        {/* Title & Status */}
        <div>
          <label className="text-[11px] font-semibold text-slate-400 block mb-1">Title</label>
          <div className="text-xs font-medium text-slate-200 bg-slate-950 p-2.5 rounded-lg border border-slate-800 truncate">
            {selectedClip.title}
          </div>
        </div>

        {/* Prompt */}
        <div>
          <label className="text-[11px] font-semibold text-slate-400 block mb-1">Generation Prompt</label>
          <textarea
            value={selectedClip.prompt}
            onChange={(e) => onUpdateClipPrompt(selectedClip.id, e.target.value)}
            rows={3}
            className="w-full bg-slate-950 text-slate-100 text-xs p-2.5 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none leading-relaxed resize-none transition-all"
          />
        </div>

        {/* Seed */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-[11px] font-semibold text-slate-400">Random Seed</label>
            <span className="text-[10px] text-slate-500 font-mono">Euler ODE Deterministic</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 bg-slate-950 px-3 py-2 rounded-lg border border-slate-800 text-xs font-mono text-cyan-300 tabular-nums">
              {selectedClip.seed}
            </div>
            <button
              onClick={handleCopySeed}
              className="p-2 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white rounded-lg transition-colors"
              title="Copy seed"
            >
              {isCopiedSeed ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Timing Information */}
        <div className="space-y-1.5 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>Duration</span>
            </span>
            <span className="font-mono text-slate-200 tabular-nums">
              {selectedClip.duration.toFixed(2)}s ({framesCount} frames)
            </span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">Start Time</span>
            <span className="font-mono text-slate-300 tabular-nums">{selectedClip.startTime.toFixed(2)}s</span>
          </div>
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">End Time</span>
            <span className="font-mono text-slate-300 tabular-nums">
              {(selectedClip.startTime + selectedClip.duration).toFixed(2)}s
            </span>
          </div>
        </div>

        {/* Linked Continuity Section */}
        <div className="space-y-2">
          <label className="text-[11px] font-semibold text-slate-400 flex items-center gap-1.5">
            <Link2 className="w-3.5 h-3.5 text-cyan-400" />
            <span>Continuity &amp; Linkage</span>
          </label>

          {selectedClip.linkedToClipId ? (
            <div className="p-2.5 rounded-lg bg-cyan-950/30 border border-cyan-800/50 space-y-2">
              <div className="flex items-center justify-between text-[11px]">
                <span className="text-cyan-300 font-medium">Continued from prior clip</span>
                <span className="font-mono text-[10px] text-cyan-400 bg-cyan-950 px-1.5 py-0.5 rounded border border-cyan-800/80">
                  {selectedClip.linkedToClipId}
                </span>
              </div>

              {selectedClip.continuity && (
                <div className="space-y-1 pt-1 text-[11px]">
                  <div className="flex items-center justify-between text-slate-400">
                    <span>Seam Smoothness</span>
                    <span className="font-mono text-emerald-400 tabular-nums">
                      {selectedClip.continuity.seamSmoothness.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-400">
                    <span>Identity Similarity</span>
                    <span className="font-mono text-cyan-400 tabular-nums">
                      {selectedClip.continuity.identitySimilarity.toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-slate-400">
                    <span>Color &amp; Lighting Drift</span>
                    <span className="font-mono text-slate-300 tabular-nums">
                      {selectedClip.continuity.colorDrift.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-2.5 rounded-lg bg-slate-950/60 border border-slate-800 text-[11px] text-slate-400 flex items-center gap-2">
              <Unlink className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              <span>Independent root clip (starts from Gaussian noise $x_0$)</span>
            </div>
          )}
        </div>

        {/* Actions: Split / Delete */}
        <div className="pt-2 border-t border-slate-800 space-y-2">
          <button
            onClick={() => onSplitClip(selectedClip.id)}
            className="w-full py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-2 border border-slate-700/60 transition-colors"
          >
            <Scissors className="w-3.5 h-3.5 text-amber-400" />
            <span>Split at Playhead (S)</span>
          </button>

          <button
            onClick={() => onDeleteClip(selectedClip.id)}
            className="w-full py-2 px-3 rounded-lg bg-red-950/40 hover:bg-red-900/60 text-red-300 text-xs font-medium flex items-center justify-center gap-2 border border-red-800/50 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5 text-red-400" />
            <span>Delete Clip (Del)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
