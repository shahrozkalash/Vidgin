import React from 'react';
import { X, Keyboard } from 'lucide-react';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const KeyboardShortcutsModal: React.FC<KeyboardShortcutsModalProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  const shortcuts = [
    { key: 'Space', desc: 'Play / Pause timeline playback' },
    { key: 'S', desc: 'Split selected clip at current playhead position' },
    { key: 'Delete / Backspace', desc: 'Delete currently selected clip' },
    { key: 'Ctrl / ⌘ + Z', desc: 'Undo last edit' },
    { key: 'Ctrl / ⌘ + Shift + Z', desc: 'Redo previously undone edit' },
    { key: '← (Left Arrow)', desc: 'Step backward 1 frame (1/16 second)' },
    { key: '→ (Right Arrow)', desc: 'Step forward 1 frame (1/16 second)' },
    { key: 'Home', desc: 'Return playhead to 00:00' },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-xl shadow-2xl p-5 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Keyboard className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-semibold text-slate-100">Keyboard Shortcuts</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded-md transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-2 text-xs">
          {shortcuts.map((item, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between py-1.5 px-2.5 rounded-lg bg-slate-950/60 border border-slate-800/80"
            >
              <span className="text-slate-300">{item.desc}</span>
              <kbd className="px-2 py-1 bg-slate-900 border border-slate-700/80 text-cyan-300 font-mono text-[11px] rounded shadow-sm">
                {item.key}
              </kbd>
            </div>
          ))}
        </div>

        <div className="pt-2 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-200 transition-colors"
          >
            Got it
          </button>
        </div>
      </div>
    </div>
  );
};
