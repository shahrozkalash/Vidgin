import React, { useState } from 'react';
import {
  Sparkles,
  Brain,
  Film,
  Lock,
  Unlock,
  Dice5,
  Pin,
  Clock,
  Palette,
  Trees,
  User,
  CheckCircle2,
  Copy
} from 'lucide-react';
import { LeftPanelTab, EditorMemoryItem, EditorClip } from '../../types/editor';

interface EditorLeftPanelProps {
  activeTab: LeftPanelTab;
  onTabChange: (tab: LeftPanelTab) => void;
  prompt: string;
  onPromptChange: (newPrompt: string) => void;
  seed: number;
  onSeedChange: (newSeed: number) => void;
  isSeedLocked: boolean;
  onToggleSeedLock: () => void;
  duration: number; // in seconds
  onDurationChange: (duration: number) => void;
  onGenerateClick: () => void;
  memoryItems: EditorMemoryItem[];
  onToggleMemoryLock: (id: string) => void;
  onToggleMemoryPin: (id: string) => void;
  clips: EditorClip[];
  onSelectClip: (clipId: string) => void;
  selectedClipId: string | null;
  hideTabBar?: boolean; // if rendered inside compact drawer tab bar
}

export const EditorLeftPanel: React.FC<EditorLeftPanelProps> = ({
  activeTab,
  onTabChange,
  prompt,
  onPromptChange,
  seed,
  onSeedChange,
  isSeedLocked,
  onToggleSeedLock,
  duration,
  onDurationChange,
  onGenerateClick,
  memoryItems,
  onToggleMemoryLock,
  onToggleMemoryPin,
  clips,
  onSelectClip,
  selectedClipId,
  hideTabBar = false,
}) => {
  const [memoryFilter, setMemoryFilter] = useState<'all' | 'character' | 'environment' | 'style'>('all');
  const [copiedSeed, setCopiedSeed] = useState(false);
  const [selectedPresetVal, setSelectedPresetVal] = useState('');

  const presetOptions = [
    { label: '-- Select a Quick Preset --', value: '' },
    {
      label: 'Red fox in misty morning forest',
      value: 'A cinematic red fox exploring misty birch woods at dawn, volumetric morning sunlight, photorealistic 8k',
    },
    {
      label: 'Futuristic rover in neon rain',
      value: 'A futuristic cyber rover navigating wet neon alleys in rain, steady tracking camera shot, cinematic reflections',
    },
    {
      label: 'Sailboat in turquoise ocean',
      value: 'A vintage sailboat cutting through turquoise ocean waves, aerial cinematic pan at sunset, high dynamic range',
    },
    {
      label: 'Hummingbird orchid macro',
      value: 'An amber hummingbird hovering beside violet orchids, ultra slow motion 4k macro shot, crystal clear focus',
    },
  ];

  const handleRandomizeSeed = () => {
    if (!isSeedLocked) {
      onSeedChange(Math.floor(Math.random() * 900000) + 100000);
    }
  };

  const handleCopySeed = () => {
    navigator.clipboard.writeText(seed.toString());
    setCopiedSeed(true);
    setTimeout(() => setCopiedSeed(false), 1500);
  };

  const handlePresetSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedPresetVal(val);
    if (val) {
      onPromptChange(val);
    }
  };

  const filteredMemory = memoryItems.filter(
    (item) => memoryFilter === 'all' || item.category === memoryFilter
  );

  return (
    <div className="w-full h-full bg-slate-900 flex flex-col select-none overflow-hidden border-r border-slate-800">
      {/* Top Tab Switcher */}
      {!hideTabBar && (
        <div className="grid grid-cols-3 border-b border-slate-800 bg-slate-950/80 p-1 gap-1 shrink-0">
          <button
            onClick={() => onTabChange('generate')}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
              activeTab === 'generate'
                ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
            <span className="truncate">Generate</span>
          </button>

          <button
            onClick={() => onTabChange('memory')}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
              activeTab === 'memory'
                ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Brain className="w-3.5 h-3.5 text-sky-400" />
            <span className="truncate">Memory</span>
          </button>

          <button
            onClick={() => onTabChange('clips')}
            className={`flex items-center justify-center gap-1.5 py-1.5 px-2 rounded-md text-xs font-medium transition-all ${
              activeTab === 'clips'
                ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Film className="w-3.5 h-3.5 text-amber-400" />
            <span className="truncate">Clips ({clips.length})</span>
          </button>
        </div>
      )}

      {/* Tab Body Scrollable Container */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3.5">
        {/* ================= TAB 1: GENERATE ================= */}
        {activeTab === 'generate' && (
          <div className="space-y-3">
            {/* Quick Presets Dropdown */}
            <div className="space-y-1">
              <label htmlFor="quick-preset-select" className="text-[11px] font-semibold text-slate-400 block">
                Quick Presets
              </label>
              <select
                id="quick-preset-select"
                value={selectedPresetVal}
                onChange={handlePresetSelect}
                className="w-full bg-slate-950 text-slate-200 text-xs py-1.5 px-2 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none cursor-pointer"
              >
                {presetOptions.map((opt, i) => (
                  <option key={i} value={opt.value} className="bg-slate-900 text-slate-200">
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Prompt Box: 3 rows and auto-growing */}
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <label className="text-[11px] font-semibold text-slate-300">Prompt</label>
                <span className="text-[10px] text-slate-500 font-mono">Vidgen DiT 480p</span>
              </div>
              <textarea
                value={prompt}
                onChange={(e) => onPromptChange(e.target.value)}
                placeholder="Describe your video scene..."
                rows={3}
                className="w-full min-h-[76px] bg-slate-950 text-slate-100 placeholder-slate-500 text-xs p-2.5 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none resize-y leading-relaxed"
              />
            </div>

            {/* Duration Selector (5s chunks, T = 4k + 1) */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <label className="font-semibold text-slate-300 flex items-center gap-1">
                  <Clock className="w-3 h-3 text-slate-400" />
                  <span>Duration</span>
                </label>
                <span className="text-cyan-400 font-mono text-[10px]">
                  {duration === 5 ? '81 frames (k=20)' : duration === 10 ? '161 frames (k=40)' : '241 frames (k=60)'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-1.5">
                {[5, 10, 15].map((d) => (
                  <button
                    key={d}
                    onClick={() => onDurationChange(d)}
                    className={`py-1.5 px-1 rounded-lg text-xs font-medium border text-center transition-all ${
                      duration === d
                        ? 'bg-cyan-950/70 text-cyan-300 border-cyan-700/80 shadow-sm'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="font-semibold">{d}s</div>
                    <div className="text-[9px] opacity-75">{d * 16 + 1}f</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Seed Control with Lock */}
            <div className="space-y-1">
              <div className="flex items-center justify-between text-[11px]">
                <label className="font-semibold text-slate-300">Seed</label>
                <button
                  onClick={onToggleSeedLock}
                  className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                    isSeedLocked
                      ? 'text-amber-300 bg-amber-950/60 border-amber-800/70'
                      : 'text-slate-400 bg-slate-950 border-slate-800 hover:text-slate-200'
                  }`}
                  title={isSeedLocked ? 'Seed locked (reproducible)' : 'Seed unlocked'}
                >
                  {isSeedLocked ? <Lock className="w-2.5 h-2.5 text-amber-400" /> : <Unlock className="w-2.5 h-2.5" />}
                  <span>{isSeedLocked ? 'Locked' : 'Unlocked'}</span>
                </button>
              </div>

              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => onSeedChange(parseInt(e.target.value) || 0)}
                  disabled={isSeedLocked}
                  className="flex-1 bg-slate-950 text-slate-100 text-xs px-2.5 py-1.5 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none font-mono tabular-nums disabled:opacity-60"
                />
                <button
                  onClick={handleRandomizeSeed}
                  disabled={isSeedLocked}
                  className="p-1.5 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-lg disabled:opacity-40 transition-colors"
                  title="Randomize seed"
                >
                  <Dice5 className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={handleCopySeed}
                  className="p-1.5 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-lg transition-colors"
                  title="Copy seed"
                >
                  {copiedSeed ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>

            {/* Architecture note */}
            <div className="p-2 rounded bg-slate-950/60 border border-slate-800/70 text-[10px] text-slate-400 leading-normal">
              Causal 3D VAE with 16 latent channels, 3D RoPE spatial-temporal embeddings, and rectified flow ODE sampler.
            </div>
          </div>
        )}

        {/* ================= TAB 2: MEMORY ================= */}
        {activeTab === 'memory' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-slate-200">Consistency Memory</h3>
                <p className="text-[10px] text-slate-500">Lock subjects &amp; style across clips</p>
              </div>
              <span className="text-[10px] text-cyan-400 font-mono">
                {memoryItems.filter((m) => m.locked).length} locked
              </span>
            </div>

            {/* Filter Tabs */}
            <div className="flex items-center gap-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-[10px]">
              {(['all', 'character', 'environment', 'style'] as const).map((cat) => (
                <button
                  key={cat}
                  onClick={() => setMemoryFilter(cat)}
                  className={`flex-1 py-1 text-center capitalize rounded transition-colors ${
                    memoryFilter === cat
                      ? 'bg-slate-800 text-cyan-300 font-medium'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Memory items */}
            <div className="space-y-2">
              {filteredMemory.map((item) => (
                <div
                  key={item.id}
                  className={`p-2 rounded-lg border text-xs transition-all ${
                    item.locked
                      ? 'bg-slate-950/90 border-cyan-800/80 shadow-sm'
                      : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between gap-1 mb-1">
                    <div className="flex items-center gap-1.5 truncate">
                      <div
                        className="w-4 h-4 rounded flex items-center justify-center shrink-0"
                        style={{ backgroundColor: item.avatarColor }}
                      >
                        {item.category === 'character' ? (
                          <User className="w-2.5 h-2.5 text-slate-950" />
                        ) : item.category === 'environment' ? (
                          <Trees className="w-2.5 h-2.5 text-slate-950" />
                        ) : (
                          <Palette className="w-2.5 h-2.5 text-slate-950" />
                        )}
                      </div>
                      <span className="font-medium text-slate-200 truncate">{item.name}</span>
                    </div>

                    <div className="flex items-center gap-1 shrink-0">
                      <button
                        onClick={() => onToggleMemoryPin(item.id)}
                        className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                          item.pinned ? 'text-amber-400' : 'text-slate-600 hover:text-slate-300'
                        }`}
                        title={item.pinned ? 'Pinned to prompt' : 'Pin to prompt'}
                      >
                        <Pin className="w-3 h-3" />
                      </button>
                      <button
                        onClick={() => onToggleMemoryLock(item.id)}
                        className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                          item.locked ? 'text-cyan-400' : 'text-slate-600 hover:text-slate-300'
                        }`}
                        title={item.locked ? 'Locked consistency' : 'Click to lock'}
                      >
                        {item.locked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                      </button>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-400 leading-relaxed">{item.description}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ================= TAB 3: CLIPS ================= */}
        {activeTab === 'clips' && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-slate-200">Timeline Clips</span>
              <span className="text-[10px] text-slate-500 font-mono">{clips.length} items</span>
            </div>

            <div className="space-y-1.5">
              {clips.map((clip, index) => (
                <div
                  key={clip.id}
                  onClick={() => onSelectClip(clip.id)}
                  className={`p-2 rounded-lg border cursor-pointer transition-all ${
                    selectedClipId === clip.id
                      ? 'bg-slate-800/90 border-cyan-500 ring-1 ring-cyan-500'
                      : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs mb-0.5">
                    <span className="font-medium text-slate-200 truncate">
                      {index + 1}. {clip.title}
                    </span>
                    <span className="text-[10px] text-cyan-400 font-mono tabular-nums shrink-0 ml-1">
                      {clip.duration.toFixed(1)}s
                    </span>
                  </div>
                  <p className="text-[10px] text-slate-400 line-clamp-1 leading-normal">
                    {clip.prompt}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Sticky Footer with Generate Button (always visible when Generate tab is active) */}
      {activeTab === 'generate' && (
        <div className="p-3 border-t border-slate-800 bg-slate-900/95 sticky bottom-0 mt-auto shrink-0 z-10">
          <button
            onClick={onGenerateClick}
            className="w-full py-2.5 px-3 rounded-lg bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-slate-950 font-semibold text-xs tracking-wide shadow-md shadow-cyan-950/40 flex items-center justify-center gap-1.5 active:scale-[0.98] transition-all cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 stroke-[2.2]" />
            <span>Generate Clip ({duration}s)</span>
          </button>
        </div>
      )}
    </div>
  );
};
