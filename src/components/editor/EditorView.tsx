import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  EditorClip,
  AudioClip,
  CaptionClip,
  EditorMemoryItem,
  LeftPanelTab,
  EditorHistorySnapshot
} from '../../types/editor';
import { EditorTopBar } from './EditorTopBar';
import { EditorLeftPanel } from './EditorLeftPanel';
import { EditorPreviewPlayer } from './EditorPreviewPlayer';
import { EditorInspector } from './EditorInspector';
import { EditorTimeline } from './EditorTimeline';
import { KeyboardShortcutsModal } from './KeyboardShortcutsModal';

interface EditorViewProps {
  onSwitchToOverview: () => void;
}

export const EditorView: React.FC<EditorViewProps> = ({ onSwitchToOverview }) => {
  const FPS = 16; // Vidgen standard 16 fps

  // Project state
  const [projectName, setProjectName] = useState('Nordic Wilderness - Ep 1');
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [zoom, setZoom] = useState(60); // px per second
  const [selectedClipId, setSelectedClipId] = useState<string | null>('clip-1');
  const [isShortcutsOpen, setIsShortcutsOpen] = useState(false);

  // Left Panel State
  const [activeTab, setActiveTab] = useState<LeftPanelTab>('generate');
  const [prompt, setPrompt] = useState(
    'A cinematic red fox exploring misty birch woods at dawn, cinematic lighting, photorealistic 8k'
  );
  const [seed, setSeed] = useState(42109);
  const [isSeedLocked, setIsSeedLocked] = useState(true);
  const [duration, setDuration] = useState(5); // 5s = 81 frames

  // Initial Clips Data
  const initialClips: EditorClip[] = [
    {
      id: 'clip-1',
      title: 'Scene 1: Morning Mist Entrance',
      prompt: 'A cinematic red fox emerging through misty pine forest at sunrise, soft volumetric sunlight',
      seed: 42109,
      startTime: 0,
      duration: 5.0,
      status: 'ready',
      progress: 100,
      trackIndex: 0,
      continuity: {
        seamSmoothness: 98.2,
        identitySimilarity: 96.5,
        colorDrift: 1.4,
      },
    },
    {
      id: 'clip-2',
      title: 'Scene 2: Forest Trail Panning',
      prompt: 'Red fox turns head toward distant river, soft morning light filtering through birch trees',
      seed: 42109,
      startTime: 5.0,
      duration: 5.0,
      linkedToClipId: 'clip-1',
      status: 'ready',
      progress: 100,
      trackIndex: 0,
      continuity: {
        seamSmoothness: 96.4,
        identitySimilarity: 94.8,
        colorDrift: 1.9,
      },
    },
    {
      id: 'clip-3',
      title: 'Scene 3: River Gorge Reveal',
      prompt: 'Aerial drone camera pulls back showing misty river gorge and morning light sweeping over conifers',
      seed: 78234,
      startTime: 10.0,
      duration: 5.0,
      linkedToClipId: 'clip-2',
      status: 'ready',
      progress: 100,
      trackIndex: 0,
      continuity: {
        seamSmoothness: 93.1,
        identitySimilarity: 91.2,
        colorDrift: 2.8,
      },
    },
  ];

  const initialAudio: AudioClip[] = [
    {
      id: 'audio-1',
      name: 'Ambient Forest & Dawn Wind (48kHz)',
      startTime: 0,
      duration: 15.0,
      volume: 0.8,
      muted: false,
    },
  ];

  const initialCaptions: CaptionClip[] = [
    {
      id: 'cap-1',
      text: 'Morning breaks over the northern pines...',
      startTime: 0,
      duration: 5.0,
    },
    {
      id: 'cap-2',
      text: 'A red fox pauses at the scent of running water.',
      startTime: 5.0,
      duration: 5.0,
    },
    {
      id: 'cap-3',
      text: 'The misty gorge stretches toward the horizon.',
      startTime: 10.0,
      duration: 5.0,
    },
  ];

  const initialMemory: EditorMemoryItem[] = [
    {
      id: 'mem-1',
      category: 'character',
      name: 'Ruby (Red Fox)',
      description: 'Adult vixen with vibrant amber fur, snowy white chest bib, and bushy dark tail tip.',
      locked: true,
      pinned: true,
      avatarColor: '#f97316',
      tags: ['subject', 'red-fox', 'protagonist'],
    },
    {
      id: 'mem-2',
      category: 'environment',
      name: 'Nordic Morning Woods',
      description: 'Dense subarctic birch and pine forest with ground moss, dew drops, and rising ground mist.',
      locked: true,
      pinned: false,
      avatarColor: '#10b981',
      tags: ['setting', 'forest', 'mist'],
    },
    {
      id: 'mem-3',
      category: 'style',
      name: '35mm Cinematic Film',
      description: 'Natural directional sunlight, subtle anamorphic lens flares, rich dynamic shadows.',
      locked: true,
      pinned: false,
      avatarColor: '#06b6d4',
      tags: ['lighting', 'cinematic', '35mm'],
    },
  ];

  const [clips, setClips] = useState<EditorClip[]>(initialClips);
  const [audioClips, setAudioClips] = useState<AudioClip[]>(initialAudio);
  const [captionClips, setCaptionClips] = useState<CaptionClip[]>(initialCaptions);
  const [memoryItems, setMemoryItems] = useState<EditorMemoryItem[]>(initialMemory);

  // Undo / Redo History Stack
  const [history, setHistory] = useState<EditorHistorySnapshot[]>([
    {
      clips: initialClips,
      audioClips: initialAudio,
      captionClips: initialCaptions,
      projectName: 'Nordic Wilderness - Ep 1',
    },
  ]);
  const [historyIndex, setHistoryIndex] = useState(0);

  // Push to history helper
  const pushHistory = (newClips: EditorClip[], newAudio?: AudioClip[], newCaptions?: CaptionClip[], newTitle?: string) => {
    const nextSnapshot: EditorHistorySnapshot = {
      clips: newClips,
      audioClips: newAudio ?? audioClips,
      captionClips: newCaptions ?? captionClips,
      projectName: newTitle ?? projectName,
    };
    const updatedHistory = history.slice(0, historyIndex + 1);
    updatedHistory.push(nextSnapshot);
    setHistory(updatedHistory);
    setHistoryIndex(updatedHistory.length - 1);
  };

  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      const prev = history[historyIndex - 1];
      setClips(prev.clips);
      setAudioClips(prev.audioClips);
      setCaptionClips(prev.captionClips);
      setProjectName(prev.projectName);
      setHistoryIndex(historyIndex - 1);
    }
  }, [historyIndex, history]);

  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      const next = history[historyIndex + 1];
      setClips(next.clips);
      setAudioClips(next.audioClips);
      setCaptionClips(next.captionClips);
      setProjectName(next.projectName);
      setHistoryIndex(historyIndex + 1);
    }
  }, [historyIndex, history]);

  // Compute total timeline duration
  const totalDuration = Math.max(
    15,
    ...clips.map((c) => c.startTime + c.duration),
    ...audioClips.map((a) => a.startTime + a.duration)
  );

  // Find active clip at current playhead time
  const activeClip = clips.find(
    (c) => currentTime >= c.startTime && currentTime < c.startTime + c.duration
  ) || null;

  // Selected clip object
  const selectedClip = clips.find((c) => c.id === selectedClipId) || null;

  // Playback animation loop
  const animFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number | null>(null);

  useEffect(() => {
    if (!isPlaying) {
      lastTimeRef.current = null;
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      return;
    }

    const loop = (timestamp: number) => {
      if (lastTimeRef.current !== null) {
        const deltaSec = (timestamp - lastTimeRef.current) / 1000;
        setCurrentTime((prev) => {
          const next = prev + deltaSec;
          if (next >= totalDuration) {
            setIsPlaying(false);
            return 0; // Return to start upon finish
          }
          return next;
        });
      }
      lastTimeRef.current = timestamp;
      animFrameRef.current = requestAnimationFrame(loop);
    };

    animFrameRef.current = requestAnimationFrame(loop);
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [isPlaying, totalDuration]);

  // Step frame forward or backward
  const handleStepFrame = useCallback((direction: -1 | 1) => {
    setIsPlaying(false);
    const frameDuration = 1 / FPS;
    setCurrentTime((prev) => Math.max(0, Math.min(totalDuration, prev + direction * frameDuration)));
  }, [totalDuration]);

  // Toggle Play / Pause
  const handleTogglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Split selected clip at playhead
  const handleSplitClip = useCallback((clipId: string) => {
    const target = clips.find((c) => c.id === clipId);
    if (!target) return;

    // Check if playhead is strictly inside the clip duration
    if (currentTime <= target.startTime + 0.1 || currentTime >= target.startTime + target.duration - 0.1) {
      return;
    }

    const splitOffset = currentTime - target.startTime;
    const durationPart1 = splitOffset;
    const durationPart2 = target.duration - splitOffset;

    const clipPart1: EditorClip = {
      ...target,
      duration: durationPart1,
      title: `${target.title} (Part 1)`,
    };

    const clipPart2: EditorClip = {
      ...target,
      id: `clip-${Date.now()}`,
      startTime: currentTime,
      duration: durationPart2,
      title: `${target.title} (Part 2)`,
      linkedToClipId: target.id,
    };

    const newClips = clips.flatMap((c) => (c.id === clipId ? [clipPart1, clipPart2] : [c]));
    setClips(newClips);
    setSelectedClipId(clipPart2.id);
    pushHistory(newClips);
  }, [clips, currentTime]);

  // Delete selected clip
  const handleDeleteClip = useCallback((clipId: string) => {
    const newClips = clips.filter((c) => c.id !== clipId);
    setClips(newClips);
    setSelectedClipId(newClips.length > 0 ? newClips[0].id : null);
    pushHistory(newClips);
  }, [clips]);

  // Update clip prompt
  const handleUpdateClipPrompt = (clipId: string, newPrompt: string) => {
    const newClips = clips.map((c) => (c.id === clipId ? { ...c, prompt: newPrompt } : c));
    setClips(newClips);
    pushHistory(newClips);
  };

  // Toggle Memory Lock
  const handleToggleMemoryLock = (id: string) => {
    setMemoryItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, locked: !item.locked } : item))
    );
  };

  // Toggle Memory Pin
  const handleToggleMemoryPin = (id: string) => {
    setMemoryItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, pinned: !item.pinned } : item))
    );
  };

  // Generate button click (Batch 1 shell trigger)
  const handleGenerateClick = () => {
    // Adds a clip block to the Video track
    const newClipId = `clip-${Date.now()}`;
    const newStartTime = Math.max(0, ...clips.map((c) => c.startTime + c.duration));
    const newClip: EditorClip = {
      id: newClipId,
      title: `Generated Scene #${clips.length + 1}`,
      prompt: prompt,
      seed: seed,
      startTime: newStartTime,
      duration: duration,
      status: 'ready',
      progress: 100,
      trackIndex: 0,
      continuity: {
        seamSmoothness: 95.5,
        identitySimilarity: 94.0,
        colorDrift: 2.1,
      },
    };

    const newClips = [...clips, newClip];
    setClips(newClips);
    setSelectedClipId(newClipId);
    pushHistory(newClips);

    if (!isSeedLocked) {
      setSeed(Math.floor(Math.random() * 900000) + 100000);
    }
  };

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input or textarea
      const target = e.target as HTMLElement;
      if (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable
      ) {
        return;
      }

      // Space: Play / Pause
      if (e.code === 'Space') {
        e.preventDefault();
        handleTogglePlay();
      }

      // S: Split at playhead
      if (e.key === 's' || e.key === 'S') {
        if (!e.ctrlKey && !e.metaKey && selectedClipId) {
          e.preventDefault();
          handleSplitClip(selectedClipId);
        }
      }

      // Delete / Backspace: Remove clip
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selectedClipId) {
          e.preventDefault();
          handleDeleteClip(selectedClipId);
        }
      }

      // Left Arrow: Step backward 1 frame
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        handleStepFrame(-1);
      }

      // Right Arrow: Step forward 1 frame
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        handleStepFrame(1);
      }

      // Home: Go to 0
      if (e.key === 'Home') {
        e.preventDefault();
        setIsPlaying(false);
        setCurrentTime(0);
      }

      // Undo: Ctrl/Cmd + Z
      if ((e.ctrlKey || e.metaKey) && !e.shiftKey && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        handleUndo();
      }

      // Redo: Ctrl/Cmd + Shift + Z or Ctrl/Cmd + Y
      if (
        ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'z' || e.key === 'Z')) ||
        ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y'))
      ) {
        e.preventDefault();
        handleRedo();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [
    handleTogglePlay,
    selectedClipId,
    handleSplitClip,
    handleDeleteClip,
    handleStepFrame,
    handleUndo,
    handleRedo,
  ]);

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-950 text-slate-100 font-sans overflow-hidden select-none">
      {/* Top Bar */}
      <EditorTopBar
        projectName={projectName}
        onProjectNameChange={(name) => {
          setProjectName(name);
          pushHistory(clips, undefined, undefined, name);
        }}
        canUndo={historyIndex > 0}
        canRedo={historyIndex < history.length - 1}
        onUndo={handleUndo}
        onRedo={handleRedo}
        onExport={() => alert(`Exporting "${projectName}" (Vidgen 480p 16FPS Render)...`)}
        onOpenShortcuts={() => setIsShortcutsOpen(true)}
        onSwitchToOverview={onSwitchToOverview}
        totalDuration={totalDuration}
        fps={FPS}
      />

      {/* Center 3-Column Studio Area */}
      <div className="flex-1 flex overflow-hidden min-h-0">
        {/* Left Panel: Tabs (Generate, Memory, Clips) */}
        <EditorLeftPanel
          activeTab={activeTab}
          onTabChange={setActiveTab}
          prompt={prompt}
          onPromptChange={setPrompt}
          seed={seed}
          onSeedChange={setSeed}
          isSeedLocked={isSeedLocked}
          onToggleSeedLock={() => setIsSeedLocked(!isSeedLocked)}
          duration={duration}
          onDurationChange={setDuration}
          onGenerateClick={handleGenerateClick}
          memoryItems={memoryItems}
          onToggleMemoryLock={handleToggleMemoryLock}
          onToggleMemoryPin={handleToggleMemoryPin}
          clips={clips}
          onSelectClip={(id) => {
            setSelectedClipId(id);
            const target = clips.find((c) => c.id === id);
            if (target) setCurrentTime(target.startTime);
          }}
          selectedClipId={selectedClipId}
        />

        {/* Center: Canvas Video Preview Player */}
        <EditorPreviewPlayer
          currentTime={currentTime}
          totalDuration={totalDuration}
          isPlaying={isPlaying}
          onTogglePlay={handleTogglePlay}
          onSeek={(time) => {
            setIsPlaying(false);
            setCurrentTime(time);
          }}
          onStepFrame={handleStepFrame}
          onResetTime={() => {
            setIsPlaying(false);
            setCurrentTime(0);
          }}
          activeClip={activeClip}
          fps={FPS}
        />

        {/* Right Panel: Clip Inspector */}
        <EditorInspector
          selectedClip={selectedClip}
          onUpdateClipPrompt={handleUpdateClipPrompt}
          onSplitClip={handleSplitClip}
          onDeleteClip={handleDeleteClip}
          fps={FPS}
        />
      </div>

      {/* Bottom Panel: Multi-track Timeline */}
      <EditorTimeline
        currentTime={currentTime}
        totalDuration={totalDuration}
        zoom={zoom}
        onZoomChange={setZoom}
        onSeek={(time) => {
          setIsPlaying(false);
          setCurrentTime(time);
        }}
        clips={clips}
        selectedClipId={selectedClipId}
        onSelectClip={(id) => setSelectedClipId(id)}
        onSplitClip={handleSplitClip}
        onDeleteClip={handleDeleteClip}
        audioClips={audioClips}
        captionClips={captionClips}
        fps={FPS}
      />

      {/* Keyboard Shortcuts Dialog */}
      <KeyboardShortcutsModal
        isOpen={isShortcutsOpen}
        onClose={() => setIsShortcutsOpen(false)}
      />
    </div>
  );
};
