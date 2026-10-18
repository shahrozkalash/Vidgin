import React, { useState } from 'react';
import {
  Sliders,
  Link2,
  Unlink,
  Scissors,
  Trash2,
  Clock,
  Copy,
  CheckCircle2,
  Film,
  X
} from 'lucide-react';
import { EditorClip } from '../../types/editor';

interface EditorInspectorProps {
  selectedClip: EditorClip | null;
  onUpdateClipPrompt: (clipId: string, prompt: string) => void;
  onSplitClip: (clipId: string) => void;
  onDeleteClip: (clipId: string) => void;
  fps?: number;
  onClose?: () => void;
}

export const EditorInspector: React.FC<EditorInspectorProps> = ({
  selectedClip,
  onUpdateClipPrompt,
  onSplitClip,
  onDeleteClip,
  fps = 16,
  onClose,
}) => {
  const [isCopiedSeed, setIsCopiedSeed] = useState(false);

  if (!selectedClip) {
    return (
      <div className="w-full h-full bg-slate-900 border-l border-slate-800 flex flex-col shrink-0 select-none p-4 overflow-y-auto">
        <div className="flex items-center justify-between pb-2 border-b border-slate-800 text-slate-300 font-semibold text-xs">
          <div className="flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-slate-400" />
            <span>Inspector</span>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 hover:bg-slate-800 text-slate-500 hover:text-slate-300 rounded"
              title="Close inspector"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <div className="flex-1 flex flex-col items-center justify-center text-center p-4 space-y-2">
          <div className="w-10 h-10 rounded-xl bg-slate-950 border border-slate-800 flex items-center justify-center text-slate-600">
            <Film className="w-5 h-5 stroke-[1.5]" />
          </div>
          <div className="space-y-0.5">
            <h4 className="text-xs font-semibold text-slate-300">No Clip Selected</h4>
            <p className="text-[10px] text-slate-500 leading-normal max-w-[180px]">
              Click a clip on the timeline to inspect its prompt, seed, continuity, and duration.
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
    <div className="w-full h-full bg-slate-900 border-l border-slate-800 flex flex-col shrink-0 select-none overflow-y-auto">
      {/* Header */}
      <div className="p-3 border-b border-slate-800 flex items-center justify-between gap-1">
        <div className="flex items-center gap-1.5 truncate">
          <Sliders className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <h3 className="text-xs font-semibold text-slate-200 truncate">Clip Inspector</h3>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <span className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded bg-cyan-950/80 text-cyan-400 border border-cyan-800/60">
            Track {selectedClip.trackIndex === 0 ? 'Main' : `Branch`}
          </span>
          {onClose && (
            <button
              onClick={onClose}
              className="p-1 hover:bg-slate-800 text-slate-400 hover:text-white rounded"
              title="Close inspector"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="p-3 space-y-3 text-xs">
        {/* Title */}
        <div>
          <label className="text-[10px] font-semibold text-slate-400 block mb-0.5">Title</label>
          <div className="text-xs font-medium text-slate-200 bg-slate-950 p-2 rounded-lg border border-slate-800 truncate">
            {selectedClip.title}
          </div>
        </div>

        {/* Prompt */}
        <div>
          <label className="text-[10px] font-semibold text-slate-400 block mb-0.5">Prompt</label>
          <textarea
            value={selectedClip.prompt}
            onChange={(e) => onUpdateClipPrompt(selectedClip.id, e.target.value)}
            rows={3}
            className="w-full bg-slate-950 text-slate-100 text-xs p-2 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none leading-relaxed resize-y"
          />
        </div>

        {/* Seed */}
        <div>
          <div className="flex items-center justify-between mb-0.5">
            <label className="text-[10px] font-semibold text-slate-400">Seed</label>
            <span className="text-[9px] text-slate-500 font-mono">Deterministic ODE</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="flex-1 bg-slate-950 px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs font-mono text-cyan-300 tabular-nums">
              {selectedClip.seed}
            </div>
            <button
              onClick={handleCopySeed}
              className="p-1.5 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-400 hover:text-white rounded-lg transition-colors"
              title="Copy seed"
            >
              {isCopiedSeed ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </div>
        </div>

        {/* Timing Information */}
        <div className="space-y-1 bg-slate-950/60 p-2.5 rounded-lg border border-slate-800/80 text-[11px]">
          <div className="flex items-center justify-between">
            <span className="text-slate-400 flex items-center gap-1">
              <Clock className="w-3 h-3 text-slate-500" />
              <span>Duration</span>
            </span>
            <span className="font-mono text-slate-200 tabular-nums">
              {selectedClip.duration.toFixed(2)}s ({framesCount}f)
            </span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>Start</span>
            <span className="font-mono text-slate-300 tabular-nums">{selectedClip.startTime.toFixed(2)}s</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span>End</span>
            <span className="font-mono text-slate-300 tabular-nums">
              {(selectedClip.startTime + selectedClip.duration).toFixed(2)}s
            </span>
          </div>
        </div>

        {/* Linked Continuity Section */}
        <div className="space-y-1.5">
          <label className="text-[10px] font-semibold text-slate-400 flex items-center gap-1">
            <Link2 className="w-3 h-3 text-cyan-400" />
            <span>Continuity &amp; Linkage</span>
          </label>

          {selectedClip.linkedToClipId ? (
            <div className="p-2 rounded-lg bg-cyan-950/30 border border-cyan-800/50 space-y-1 text-[11px]">
              <div className="flex items-center justify-between">
                <span className="text-cyan-300 font-medium">Linked to prior clip</span>
                <span className="font-mono text-[9px] text-cyan-400 bg-cyan-950 px-1 py-0.5 rounded border border-cyan-800/80">
                  {selectedClip.linkedToClipId}
                </span>
              </div>

              {selectedClip.continuity && (
                <div className="space-y-0.5 pt-1 text-[10px]">
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
                    <span>Color &amp; Light Drift</span>
                    <span className="font-mono text-slate-300 tabular-nums">
                      {selectedClip.continuity.colorDrift.toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800 text-[10px] text-slate-400 flex items-center gap-1.5">
              <Unlink className="w-3 h-3 text-slate-500 shrink-0" />
              <span>Independent root clip</span>
            </div>
          )}
        </div>

        {/* Actions: Split / Delete */}
        <div className="pt-1.5 border-t border-slate-800 space-y-1.5">
          <button
            onClick={() => onSplitClip(selectedClip.id)}
            className="w-full py-1.5 px-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-1.5 border border-slate-700/60 transition-colors"
          >
            <Scissors className="w-3 h-3 text-amber-400" />
            <span>Split at Playhead (S)</span>
          </button>

          <button
            onClick={() => onDeleteClip(selectedClip.id)}
            className="w-full py-1.5 px-2.5 rounded-lg bg-red-950/40 hover:bg-red-900/60 text-red-300 text-xs font-medium flex items-center justify-center gap-1.5 border border-red-800/50 transition-colors"
          >
            <Trash2 className="w-3 h-3 text-red-400" />
            <span>Delete Clip (Del)</span>
          </button>
        </div>
      </div>
    </div>
  );
};
