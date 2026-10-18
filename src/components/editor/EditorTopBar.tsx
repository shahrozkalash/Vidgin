import React, { useState, useRef, useEffect } from 'react';
import {
  Undo2,
  Redo2,
  Download,
  Film,
  Keyboard,
  SlidersHorizontal,
  Edit2,
  Check,
  PanelRightClose,
  PanelRightOpen,
  MoreHorizontal
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
  isInspectorOpen: boolean;
  onToggleInspector: () => void;
  isCompactMode?: boolean;
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
  isInspectorOpen,
  onToggleInspector,
  isCompactMode = false,
}) => {
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [tempTitle, setTempTitle] = useState(projectName);
  const [isOverflowOpen, setIsOverflowOpen] = useState(false);
  const overflowRef = useRef<HTMLDivElement | null>(null);

  // Close overflow menu on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (overflowRef.current && !overflowRef.current.contains(e.target as Node)) {
        setIsOverflowOpen(false);
      }
    };
    if (isOverflowOpen) {
      window.addEventListener('mousedown', handleOutsideClick);
    }
    return () => window.removeEventListener('mousedown', handleOutsideClick);
  }, [isOverflowOpen]);

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
    <header className="w-full bg-slate-900 border-b border-slate-800 px-3 sm:px-4 py-2 flex items-center justify-between gap-2.5 sm:gap-4 select-none shrink-0 z-20 flex-wrap min-h-[48px]">
      {/* GROUP 1 (LEFT): Logo + Project Name (flex: 1, min-width: 0, single-line ellipsis) */}
      <div className="flex items-center gap-2 sm:gap-3 flex-1 min-w-0 shrink">
        {/* Logo */}
        <div className="flex items-center gap-1.5 shrink-0">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-600 via-cyan-500 to-sky-400 flex items-center justify-center shadow-md shadow-cyan-950/50">
            <Film className="w-3.5 h-3.5 text-slate-950 stroke-[2.5]" />
          </div>
          <span className="font-bold tracking-tight text-white text-sm sm:text-base shrink-0">vidgen</span>
        </div>

        <div className="h-3.5 w-px bg-slate-800 shrink-0" />

        {/* Project Name */}
        <div className="flex-1 min-w-0">
          {isEditingTitle ? (
            <div className="flex items-center gap-1 w-full">
              <input
                type="text"
                value={tempTitle}
                onChange={(e) => setTempTitle(e.target.value)}
                onKeyDown={handleKeyDown}
                onBlur={handleSaveTitle}
                autoFocus
                className="bg-slate-950 text-slate-100 text-xs px-2 py-0.5 rounded border border-cyan-500 outline-none w-full max-w-[260px] truncate focus:ring-1 focus:ring-cyan-500"
                maxLength={60}
              />
              <button
                onClick={handleSaveTitle}
                className="p-1 hover:text-cyan-400 text-slate-400 transition-colors shrink-0"
                title="Save project title"
              >
                <Check className="w-3 h-3" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => {
                setTempTitle(projectName);
                setIsEditingTitle(true);
              }}
              className="flex items-center gap-1.5 text-xs font-medium text-slate-300 hover:text-white px-1.5 py-1 rounded hover:bg-slate-800/70 transition-colors group w-full truncate text-left"
              title="Click to rename project"
            >
              <span className="truncate">{projectName}</span>
              <Edit2 className="w-2.5 h-2.5 text-slate-500 group-hover:text-cyan-400 opacity-60 group-hover:opacity-100 transition-opacity shrink-0" />
            </button>
          )}
        </div>
      </div>

      {/* GROUP 2 (MIDDLE): Undo, Redo */}
      <div className="flex items-center gap-1 shrink-0">
        <div className="flex items-center gap-0.5 bg-slate-950/60 p-0.5 rounded-lg border border-slate-800/80">
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
            <Undo2 className="w-3.5 h-3.5" />
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
            <Redo2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* GROUP 3 (RIGHT): Shortcuts button / overflow, Inspector Toggle, Overview, Export */}
      <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
        {/* Shortcuts button: visible when >= 1100px */}
        <div className="hidden lg:block">
          <button
            onClick={onOpenShortcuts}
            title="Keyboard Shortcuts"
            aria-label="Keyboard Shortcuts"
            className="flex items-center gap-1 p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors text-xs"
          >
            <Keyboard className="w-3.5 h-3.5" />
            <span className="hidden xl:inline text-[11px]">Shortcuts</span>
          </button>
        </div>

        {/* Overflow Menu below 1100px */}
        <div className="relative lg:hidden" ref={overflowRef}>
          <button
            onClick={() => setIsOverflowOpen(!isOverflowOpen)}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition-colors"
            title="More actions"
          >
            <MoreHorizontal className="w-4 h-4" />
          </button>

          {isOverflowOpen && (
            <div className="absolute right-0 top-full mt-1 w-44 bg-slate-900 border border-slate-800 rounded-lg shadow-xl p-1 z-50 text-xs animate-in fade-in duration-100">
              <button
                onClick={() => {
                  setIsOverflowOpen(false);
                  onOpenShortcuts();
                }}
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-slate-300 hover:text-white hover:bg-slate-800 rounded transition-colors"
              >
                <Keyboard className="w-3.5 h-3.5 text-cyan-400" />
                <span>Shortcuts</span>
              </button>
            </div>
          )}
        </div>

        {/* Inspector Toggle Button */}
        {!isCompactMode && (
          <button
            onClick={onToggleInspector}
            className={`flex items-center gap-1 text-xs px-2 sm:px-2.5 py-1.5 rounded-md border transition-colors ${
              isInspectorOpen
                ? 'bg-slate-800 text-cyan-400 border-slate-700'
                : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-800/60'
            }`}
            title={isInspectorOpen ? 'Collapse Inspector' : 'Expand Inspector'}
          >
            {isInspectorOpen ? (
              <PanelRightClose className="w-3.5 h-3.5" />
            ) : (
              <PanelRightOpen className="w-3.5 h-3.5" />
            )}
            <span className="hidden md:inline text-[11px]">Inspector</span>
          </button>
        )}

        {/* Overview Switch */}
        <button
          onClick={onSwitchToOverview}
          className="flex items-center gap-1 text-xs text-slate-300 hover:text-white px-2 sm:px-2.5 py-1.5 rounded-md bg-slate-800/70 hover:bg-slate-800 border border-slate-700/60 transition-colors shrink-0"
          title="Switch to Overview"
        >
          <SlidersHorizontal className="w-3 h-3 text-cyan-400" />
          <span className="hidden sm:inline text-[11px]">Overview</span>
        </button>

        {/* Export Button */}
        <button
          onClick={onExport}
          className="flex items-center gap-1.5 text-xs font-semibold text-slate-950 bg-gradient-to-r from-cyan-400 to-sky-400 hover:from-cyan-300 hover:to-sky-300 px-3 py-1.5 rounded-md shadow-sm transition-all active:scale-95 shrink-0"
        >
          <Download className="w-3.5 h-3.5 stroke-[2.2]" />
          <span>Export</span>
        </button>
      </div>
    </header>
  );
};
