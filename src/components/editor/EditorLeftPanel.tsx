import React, { useState } from 'react';
import {
  Sparkles,
  Brain,
  Film,
  Lock,
  Unlock,
  Dice5,
  Plus,
  Pin,
  Clock,
  Layers,
  Sliders,
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
}) => {
  const [memoryFilter, setMemoryFilter] = useState<'all' | 'character' | 'environment' | 'style'>('all');
  const [copiedSeed, setCopiedSeed] = useState(false);

  const samplePrompts = [
    'A cinematic red fox exploring misty birch woods at dawn, cinematic lighting, photorealistic 8k',
    'A futuristic cyber rover navigating wet neon alleys in rain, steady tracking camera shot',
    'A vintage sailboat cutting through turquoise ocean waves, aerial cinematic pan at sunset',
    'An amber hummingbird hovering beside violet orchids, ultra slow motion 4k macro shot',
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

  const filteredMemory = memoryItems.filter(
    (item) => memoryFilter === 'all' || item.category === memoryFilter
  );

  return (
    <div className="w-80 md:w-88 lg:w-96 bg-slate-900 border-r border-slate-800 flex flex-col h-full shrink-0 select-none overflow-hidden">
      {/* Top Tab Switcher */}
      <div className="grid grid-cols-3 border-b border-slate-800 bg-slate-950/70 p-1 gap-1">
        <button
          onClick={() => onTabChange('generate')}
          className={`flex items-center justify-center gap-1.5 py-2 px-2 rounded-md text-xs font-medium transition-all ${
            activeTab === 'generate'
              ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5 text-cyan-400" />
          <span>Generate</span>
        </button>

        <button
          onClick={() => onTabChange('memory')}
          className={`flex items-center justify-center gap-1.5 py-2 px-2 rounded-md text-xs font-medium transition-all ${
            activeTab === 'memory'
              ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Brain className="w-3.5 h-3.5 text-sky-400" />
          <span>Memory</span>
        </button>

        <button
          onClick={() => onTabChange('clips')}
          className={`flex items-center justify-center gap-1.5 py-2 px-2 rounded-md text-xs font-medium transition-all ${
            activeTab === 'clips'
              ? 'bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/60'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
          }`}
        >
          <Film className="w-3.5 h-3.5 text-amber-400" />
          <span>Clips ({clips.length})</span>
        </button>
      </div>

      {/* Tab Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* ================= TAB 1: GENERATE ================= */}
        {activeTab === 'generate' && (
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-xs font-semibold text-slate-300">Prompt</label>
                <span className="text-[10px] text-slate-500 font-mono">Vidgen Flow Matching</span>
              </div>
              <textarea
                value={prompt}
                onChange={(e) => onPromptChange(e.target.value)}
                placeholder="Describe the video scene to generate..."
                rows={3}
                className="w-full bg-slate-950 text-slate-100 placeholder-slate-500 text-xs p-3 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none resize-none transition-all leading-relaxed"
              />

              {/* Sample Prompts Pills */}
              <div className="mt-2 space-y-1">
                <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">Quick Presets:</span>
                <div className="flex flex-col gap-1">
                  {samplePrompts.slice(0, 2).map((sp, idx) => (
                    <button
                      key={idx}
                      onClick={() => onPromptChange(sp)}
                      className="text-left text-[11px] text-slate-400 hover:text-cyan-300 bg-slate-950/50 hover:bg-slate-800/80 p-1.5 rounded border border-slate-800/60 truncate transition-colors"
                      title={sp}
                    >
                      {sp}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Duration Selector (5s chunks, strictly T = 4k + 1) */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  <span>Duration</span>
                </label>
                <span className="text-[10px] text-cyan-400 font-mono">
                  {duration === 5 ? '81 frames (k=20)' : duration === 10 ? '161 frames (k=40)' : '241 frames (k=60)'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {[5, 10, 15].map((d) => (
                  <button
                    key={d}
                    onClick={() => onDurationChange(d)}
                    className={`py-2 px-2 rounded-lg text-xs font-medium border text-center transition-all ${
                      duration === d
                        ? 'bg-cyan-950/70 text-cyan-300 border-cyan-700/80 shadow-sm'
                        : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200 hover:bg-slate-800/50'
                    }`}
                  >
                    <div className="font-semibold">{d}s</div>
                    <div className="text-[10px] opacity-75">{d * 16 + 1}f</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Seed Control with Lock */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label className="text-xs font-semibold text-slate-300">Seed</label>
                <div className="flex items-center gap-2">
                  <button
                    onClick={onToggleSeedLock}
                    className={`flex items-center gap-1 text-[11px] px-2 py-0.5 rounded border transition-colors ${
                      isSeedLocked
                        ? 'text-amber-300 bg-amber-950/60 border-amber-800/70'
                        : 'text-slate-400 bg-slate-950 border-slate-800 hover:text-slate-200'
                    }`}
                    title={isSeedLocked ? 'Seed locked (reproducible)' : 'Seed unlocked (randomizes per generation)'}
                  >
                    {isSeedLocked ? <Lock className="w-3 h-3 text-amber-400" /> : <Unlock className="w-3 h-3" />}
                    <span>{isSeedLocked ? 'Locked' : 'Unlocked'}</span>
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="relative flex-1">
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => onSeedChange(parseInt(e.target.value) || 0)}
                    disabled={isSeedLocked}
                    className="w-full bg-slate-950 text-slate-100 text-xs px-3 py-2 rounded-lg border border-slate-800 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 outline-none font-mono tabular-nums disabled:opacity-60 disabled:cursor-not-allowed"
                  />
                </div>
                <button
                  onClick={handleRandomizeSeed}
                  disabled={isSeedLocked}
                  className="p-2 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  title="Randomize seed"
                >
                  <Dice5 className="w-4 h-4" />
                </button>
                <button
                  onClick={handleCopySeed}
                  className="p-2 bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white rounded-lg transition-colors"
                  title="Copy seed"
                >
                  {copiedSeed ? <CheckCircle2 className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {/* Model Architecture Note */}
            <div className="p-2.5 rounded-lg bg-slate-950/70 border border-slate-800/80 text-[11px] text-slate-400 space-y-1">
              <div className="flex items-center justify-between text-slate-300 font-medium">
                <span>Model Backend</span>
                <span className="text-cyan-400 font-mono">Vidgen DiT 480p</span>
              </div>
              <p className="text-[10px] text-slate-500 leading-normal">
                4x8x8 Causal 3D VAE with 16 latent channels, 3D RoPE position embeddings, and rectified flow ODE sampler.
              </p>
            </div>

            {/* Big Primary Generate Button */}
            <button
              onClick={onGenerateClick}
              className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-cyan-500 to-sky-500 hover:from-cyan-400 hover:to-sky-400 text-slate-950 font-semibold text-xs tracking-wide shadow-md shadow-cyan-950/40 flex items-center justify-center gap-2 active:scale-[0.98] transition-all cursor-pointer"
            >
              <Sparkles className="w-4 h-4 stroke-[2.2]" />
              <span>Generate Clip (5s)</span>
            </button>
          </div>
        )}

        {/* ================= TAB 2: MEMORY ================= */}
        {activeTab === 'memory' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-xs font-semibold text-slate-200">Consistency Memory</h3>
                <p className="text-[10px] text-slate-400">Lock subjects and style across continued clips</p>
              </div>
              <span className="text-[10px] text-cyan-400 font-mono">
                {memoryItems.filter((m) => m.locked).length} locked
              </span>
            </div>

            {/* Memory Filter Tabs */}
            <div className="flex items-center gap-1 p-0.5 bg-slate-950 rounded-lg border border-slate-800 text-[11px]">
              {(['all', 'character', 'environment', 'style'] as const).map((cat) => (
                <button
                  key={cat}
                  onClick={() => setMemoryFilter(cat)}
                  className={`flex-1 py-1 text-center capitalize rounded transition-colors ${
                    memoryFilter === cat
                      ? 'bg-slate-800 text-cyan-300 font-medium shadow-sm'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>

            {/* Memory Items List */}
            <div className="space-y-2">
              {filteredMemory.map((item) => (
                <div
                  key={item.id}
                  className={`p-2.5 rounded-lg border transition-all ${
                    item.locked
                      ? 'bg-slate-950/90 border-cyan-800/80 shadow-sm shadow-cyan-950/30'
                      : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold text-slate-950"
                        style={{ backgroundColor: item.avatarColor }}
                      >
                        {item.category === 'character' ? (
                          <User className="w-3 h-3 text-slate-950" />
                        ) : item.category === 'environment' ? (
                          <Trees className="w-3 h-3 text-slate-950" />
                        ) : (
                          <Palette className="w-3 h-3 text-slate-950" />
                        )}
                      </div>
                      <div>
                        <div className="text-xs font-medium text-slate-200">{item.name}</div>
                        <div className="text-[10px] text-slate-500 capitalize">{item.category}</div>
                      </div>
                    </div>

                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => onToggleMemoryPin(item.id)}
                        className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                          item.pinned ? 'text-amber-400' : 'text-slate-600 hover:text-slate-300'
                        }`}
                        title={item.pinned ? 'Pinned to prompt' : 'Pin to prompt'}
                      >
                        <Pin className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => onToggleMemoryLock(item.id)}
                        className={`p-1 rounded hover:bg-slate-800 transition-colors ${
                          item.locked ? 'text-cyan-400' : 'text-slate-600 hover:text-slate-300'
                        }`}
                        title={item.locked ? 'Locked consistency' : 'Click to lock'}
                      >
                        {item.locked ? <Lock className="w-3.5 h-3.5" /> : <Unlock className="w-3.5 h-3.5" />}
                      </button>
                    </div>
                  </div>

                  <p className="text-[11px] text-slate-400 leading-relaxed mb-2">{item.description}</p>

                  <div className="flex flex-wrap gap-1">
                    {item.tags.map((tag, idx) => (
                      <span
                        key={idx}
                        className="text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ================= TAB 3: CLIPS ================= */}
        {activeTab === 'clips' && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-200">Project Timeline Clips</h3>
              <span className="text-[10px] text-slate-400">{clips.length} segments</span>
            </div>

            <div className="space-y-2">
              {clips.map((clip, index) => (
                <div
                  key={clip.id}
                  onClick={() => onSelectClip(clip.id)}
                  className={`p-2.5 rounded-lg border cursor-pointer transition-all ${
                    selectedClipId === clip.id
                      ? 'bg-slate-800/90 border-cyan-500 ring-1 ring-cyan-500'
                      : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-800/50 hover:border-slate-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs font-medium text-slate-200 truncate max-w-[170px]">
                      {index + 1}. {clip.title}
                    </span>
                    <span className="text-[10px] text-cyan-400 font-mono tabular-nums">
                      {clip.duration.toFixed(1)}s
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed mb-1.5">
                    {clip.prompt}
                  </p>
                  <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
                    <span>Seed: {clip.seed}</span>
                    <span>Start: {clip.startTime.toFixed(1)}s</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
