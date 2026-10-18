import React, { useState } from 'react';
import {
  Undo2,
  Redo2,
  Download,
  Film,
  Keyboard,
  SlidersHorizontal,
  Edit2,
  Check,
  Clock,
  HardDrive
} from 'lucide-react';

interface EditorTopBarProps {
  projectName: string;
  onProjectNameChange: (newName: string) => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onExport: () => void;
  onOpenShortcuts: () => void;
  onSwitchToOverview: () => void;
  totalDuration: number;
  fps?: number;
}

export const EditorTopBar: React.FC<EditorTopBarProps> = ({
  projectName,
  onProjectNameChange,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  onExport,
  onOpenShortcuts,
  onSwitchToOverview,
  totalDuration,
  fps = 16,
}) => {
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [tempTitle, setTempTitle] = useState(projectName);

  // Total frames calculated strictly adhering to Vidgen's T = 4k + 1 formula
  const frameCount = Math.floor(totalDuration * fps) + 1;
  // Estimated export size for 480p 16FPS H.264/WebM video stream (~0.55 MB per second)
  const estSizeMb = (totalDuration * 0.55).toFixed(1);

  const handleSaveTitle = () => {
    if (tempTitle.trim()) {
      onProjectNameChange(tempTitle.trim());
    } else {
      setTempTitle(projectName);
    }
    setIsEditingTitle(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveTitle();
    } else if (e.key === 'Escape') {
      setTempTitle(projectName);
      setIsEditingTitle(false);
    }
  };

  return (
    <header className="h-14 bg-slate-900/95 border-b border-slate-800 px-4 flex items-center justify-between gap-4 select-none shrink-0 z-20">
      {/* Brand & Project Name */}
      <div className="flex items-center gap-4 min-w-0">
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-600 via-cyan-500 to-sky-400 flex items-center justify-center shadow-lg shadow-cyan-950/50">
            <Film className="w-4 h-4 text-slate-950 stroke-[2.5]" />
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-bold tracking-tight text-white text-base">vidgen</span>
            <span className="text-[10px] font-semibold tracking-wider uppercase text-cyan-400 bg-cyan-950/80 border border-cyan-800/60 px-1.5 py-0.5 rounded">
              Timeline Editor
            </span>
          </div>
        </div>

        <div className="h-4 w-px bg-slate-800 shrink-0" />

        {/* Editable Project Name */}
        <div className="flex items-center gap-1.5 min-w-0">
          {isEditingTitle ? (
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={tempTitle}
                onChange={(e) => setTempTitle(e.target.value)}
                onKeyDown={handleKeyDown}
                onBlur={handleSaveTitle}
                autoFocus
                className="bg-slate-950 text-slate-100 text-xs px-2.5 py-1 rounded border border-cyan-500 outline-none w-48 sm:w-64 focus:ring-1 focus:ring-cyan-500"
                maxLength={60}
              />
              <button
                onClick={handleSaveTitle}
                className="p-1 hover:text-cyan-400 text-slate-400 transition-colors"
                title="Save project title"
              >
                <Check className="w-3.5 h-3.5" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setTempTitle(projectName);
                setIsEditingTitle(true);
              }}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-300 hover:text-white px-2 py-1 rounded hover:bg-slate-800/70 transition-colors group truncate max-w-[220px] sm:max-w-[320px]"
              title="Click to rename project"
            >
              <span className="truncate">{projectName}</span>
              <Edit2 className="w-3 h-3 text-slate-500 group-hover:text-cyan-400 opacity-60 group-hover:opacity-100 transition-opacity shrink-0" />
            </button>
          )}
        </div>
      </div>

      {/* Project Statistics Bar */}
      <div className="hidden lg:flex items-center gap-3 text-xs bg-slate-950/80 border border-slate-800/80 px-3.5 py-1.5 rounded-lg select-none">
        <div className="flex items-center gap-1.5" title="Total project timeline duration">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-500 font-medium">Duration:</span>
          <span className="font-mono text-slate-200 tabular-nums font-semibold">{totalDuration.toFixed(1)}s</span>
        </div>

        <span className="text-slate-700" aria-hidden="true">·</span>

        <div className="flex items-center gap-1.5" title="Total frames generated at 16 FPS (Vidgen T = 4k + 1)">
          <Film className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-slate-500 font-medium">Frames:</span>
          <span className="font-mono text-slate-200 tabular-nums font-semibold">{frameCount}f</span>
        </div>

        <span className="text-slate-700" aria-hidden="true">·</span>

        <div className="flex items-center gap-1.5" title="Estimated 480p H.264/WebM export payload size">
          <HardDrive className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-slate-500 font-medium">Est. Size:</span>
          <span className="font-mono text-slate-200 tabular-nums font-semibold">~{estSizeMb} MB</span>
        </div>
      </div>

      {/* Center Utility: Undo / Redo / Shortcuts */}
      <div className="flex items-center gap-1 bg-slate-950/60 p-1 rounded-lg border border-slate-800/80">
        <button
          onClick={onUndo}
          disabled={!canUndo}
          title="Undo (Ctrl+Z)"
          aria-label="Undo"
          className={`p-1.5 rounded transition-colors ${
            canUndo
              ? 'text-slate-300 hover:text-white hover:bg-slate-800 active:scale-95'
              : 'text-slate-600 cursor-not-allowed'
          }`}
        >
          <Undo2 className="w-4 h-4" />
        </button>
        <button
          onClick={onRedo}
          disabled={!canRedo}
          title="Redo (Ctrl+Shift+Z)"
          aria-label="Redo"
          className={`p-1.5 rounded transition-colors ${
            canRedo
              ? 'text-slate-300 hover:text-white hover:bg-slate-800 active:scale-95'
              : 'text-slate-600 cursor-not-allowed'
          }`}
        >
          <Redo2 className="w-4 h-4" />
        </button>
        <div className="h-3 w-px bg-slate-800 mx-1" />
        <button
          onClick={onOpenShortcuts}
          title="Keyboard Shortcuts"
          aria-label="Keyboard Shortcuts"
          className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
        >
          <Keyboard className="w-4 h-4" />
        </button>
      </div>

      {/* Right Controls: Overview Toggle & Export */}
      <div className="flex items-center gap-2.5 shrink-0">
        <button
          onClick={onSwitchToOverview}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-3 py-1.5 rounded-md bg-slate-800/70 hover:bg-slate-800 border border-slate-700/60 transition-colors"
          title="Switch to Architecture Overview"
        >
          <SlidersHorizontal className="w-3.5 h-3.5 text-cyan-400" />
          <span className="hidden sm:inline">Overview</span>
        </button>

        <button
          onClick={onExport}
          className="flex items-center gap-1.5 text-xs font-medium text-slate-950 bg-gradient-to-r from-cyan-400 to-sky-400 hover:from-cyan-300 hover:to-sky-300 px-3.5 py-1.5 rounded-md shadow-sm transition-all active:scale-95 font-sans"
        >
          <Download className="w-3.5 h-3.5 stroke-[2.2]" />
          <span>Export</span>
        </button>
      </div>
    </header>
  );
};
