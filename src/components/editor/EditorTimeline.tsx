import React, { useRef, useState, useEffect } from 'react';
import {
  Scissors,
  Trash2,
  ZoomIn,
  ZoomOut,
  Volume2,
  VolumeX,
  Type,
  Video,
  Music,
  Link2,
  Layers,
  ChevronRight,
  Maximize
} from 'lucide-react';
import { EditorClip, AudioClip, CaptionClip } from '../../types/editor';

interface EditorTimelineProps {
  currentTime: number;
  totalDuration: number;
  zoom: number; // Pixels per second (e.g. 40 - 160)
  onZoomChange: (newZoom: number) => void;
  onSeek: (newTime: number) => void;
  clips: EditorClip[];
  selectedClipId: string | null;
  onSelectClip: (clipId: string | null) => void;
  onSplitClip: (clipId: string) => void;
  onDeleteClip: (clipId: string) => void;
  audioClips: AudioClip[];
  captionClips: CaptionClip[];
  fps?: number;
}

export const EditorTimeline: React.FC<EditorTimelineProps> = ({
  currentTime,
  totalDuration,
  zoom,
  onZoomChange,
  onSeek,
  clips,
  selectedClipId,
  onSelectClip,
  onSplitClip,
  onDeleteClip,
  audioClips,
  captionClips,
  fps = 16,
}) => {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [isAudioMuted, setIsAudioMuted] = useState(false);

  // Timeline track total width in pixels (give extra padding at end)
  const timelineContentWidth = Math.max(800, (totalDuration + 5) * zoom);

  // Format seconds to MM:SS
  const formatRulerTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  // Convert pixel offset on timeline track to timestamp in seconds
  const handleTimelineMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!scrollContainerRef.current) return;
    const rect = scrollContainerRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left + scrollContainerRef.current.scrollLeft;
    const newTime = Math.max(0, Math.min(totalDuration, clickX / zoom));
    onSeek(newTime);
    setIsDraggingPlayhead(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingPlayhead || !scrollContainerRef.current) return;
      const rect = scrollContainerRef.current.getBoundingClientRect();
      const clickX = e.clientX - rect.left + scrollContainerRef.current.scrollLeft;
      const newTime = Math.max(0, Math.min(totalDuration, clickX / zoom));
      onSeek(newTime);
    };

    const handleMouseUp = () => {
      if (isDraggingPlayhead) {
        setIsDraggingPlayhead(false);
      }
    };

    if (isDraggingPlayhead) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingPlayhead, zoom, totalDuration, onSeek]);

  // Generate ruler tick marks based on zoom
  const tickInterval = zoom < 50 ? 5 : zoom < 90 ? 2 : 1; // seconds between major ticks
  const rulerTicks = [];
  for (let sec = 0; sec <= totalDuration + 5; sec += tickInterval) {
    rulerTicks.push(sec);
  }

  const selectedClip = clips.find((c) => c.id === selectedClipId);

  return (
    <div className="h-72 bg-slate-900 border-t border-slate-800 flex flex-col shrink-0 select-none overflow-hidden">
      {/* Timeline Controls & Action Bar */}
      <div className="h-10 bg-slate-950/80 border-b border-slate-800 px-4 flex items-center justify-between gap-4 shrink-0 text-xs text-slate-400">
        {/* Left Actions: Split & Delete */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => selectedClipId && onSplitClip(selectedClipId)}
            disabled={!selectedClipId}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
              selectedClipId
                ? 'bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white'
                : 'text-slate-600 bg-slate-900 cursor-not-allowed'
            }`}
            title="Split selected clip at current playhead (S)"
          >
            <Scissors className="w-3.5 h-3.5 text-amber-400" />
            <span>Split (S)</span>
          </button>

          <button
            onClick={() => selectedClipId && onDeleteClip(selectedClipId)}
            disabled={!selectedClipId}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors ${
              selectedClipId
                ? 'bg-red-950/40 text-red-300 hover:bg-red-900/60 border border-red-900/40'
                : 'text-slate-600 bg-slate-900 cursor-not-allowed'
            }`}
            title="Delete selected clip (Del)"
          >
            <Trash2 className="w-3.5 h-3.5 text-red-400" />
            <span>Delete (Del)</span>
          </button>

          <div className="h-3.5 w-px bg-slate-800 mx-1" />

          {selectedClip && (
            <span className="text-[11px] text-slate-400 truncate max-w-[200px]">
              Selected: <strong className="text-slate-200">{selectedClip.title}</strong> ({selectedClip.duration.toFixed(1)}s)
            </span>
          )}
        </div>

        {/* Right: Zoom Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onZoomChange(Math.max(30, zoom - 15))}
            className="p-1 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          <input
            type="range"
            min={30}
            max={160}
            value={zoom}
            onChange={(e) => onZoomChange(parseInt(e.target.value))}
            className="w-24 h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
            title={`Timeline zoom: ${zoom}px / second`}
          />

          <button
            onClick={() => onZoomChange(Math.min(160, zoom + 15))}
            className="p-1 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          <span className="text-[10px] text-slate-500 font-mono w-10 text-right">{zoom}px/s</span>
        </div>
      </div>

      {/* Main Timeline Workspace: Left Track Headers + Right Scrollable Tracks */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Track Headers (Fixed Width) */}
        <div className="w-28 sm:w-32 bg-slate-950 border-r border-slate-800 flex flex-col shrink-0 z-10 select-none">
          {/* Header Spacer matching Ruler */}
          <div className="h-6 border-b border-slate-800 bg-slate-950 flex items-center px-2 text-[10px] font-mono text-slate-500">
            TRACKS
          </div>

          {/* Track 1: Video Header */}
          <div className="h-18 border-b border-slate-800/80 px-2.5 flex items-center justify-between text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5">
              <Video className="w-3.5 h-3.5 text-cyan-400" />
              <span>Video</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">1</span>
          </div>

          {/* Track 2: Audio Header */}
          <div className="h-14 border-b border-slate-800/80 px-2.5 flex items-center justify-between text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5">
              <Music className="w-3.5 h-3.5 text-sky-400" />
              <span>Audio</span>
            </div>
            <button
              onClick={() => setIsAudioMuted(!isAudioMuted)}
              className="p-1 hover:text-slate-100 text-slate-500 transition-colors"
              title={isAudioMuted ? 'Unmute' : 'Mute'}
            >
              {isAudioMuted ? <VolumeX className="w-3.5 h-3.5 text-amber-400" /> : <Volume2 className="w-3.5 h-3.5" />}
            </button>
          </div>

          {/* Track 3: Captions Header */}
          <div className="h-12 border-b border-slate-800/80 px-2.5 flex items-center justify-between text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5">
              <Type className="w-3.5 h-3.5 text-amber-400" />
              <span>Captions</span>
            </div>
            <span className="text-[10px] text-slate-500 font-mono">CC</span>
          </div>
        </div>

        {/* Right: Scrollable Timeline Tracks Container */}
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-x-auto overflow-y-hidden relative bg-slate-950 scrollbar-thin scrollbar-thumb-slate-800 select-none"
        >
          <div
            className="relative h-full"
            style={{ width: `${timelineContentWidth}px` }}
            onMouseDown={handleTimelineMouseDown}
          >
            {/* Top Time Ruler */}
            <div className="h-6 bg-slate-950 border-b border-slate-800 relative cursor-pointer select-none">
              {rulerTicks.map((sec) => (
                <div
                  key={sec}
                  className="absolute top-0 bottom-0 border-l border-slate-800 flex items-end pb-0.5 pl-1"
                  style={{ left: `${sec * zoom}px` }}
                >
                  <span className="text-[9px] font-mono text-slate-500 tabular-nums">
                    {formatRulerTime(sec)}
                  </span>
                </div>
              ))}
            </div>

            {/* Vertical Second Grid Lines across tracks */}
            {rulerTicks.map((sec) => (
              <div
                key={`grid-${sec}`}
                className="absolute top-6 bottom-0 border-l border-slate-900/60 pointer-events-none"
                style={{ left: `${sec * zoom}px` }}
              />
            ))}

            {/* TRACK 1: Video Track */}
            <div className="h-18 border-b border-slate-800/80 relative flex items-center bg-slate-900/20">
              {clips.map((clip) => {
                const clipLeft = clip.startTime * zoom;
                const clipWidth = Math.max(30, clip.duration * zoom);
                const isSelected = selectedClipId === clip.id;

                return (
                  <div
                    key={clip.id}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectClip(clip.id);
                    }}
                    className={`absolute top-1 bottom-1 rounded-md overflow-hidden cursor-pointer transition-all border ${
                      isSelected
                        ? 'border-cyan-400 bg-cyan-950/70 shadow-lg shadow-cyan-950/40 ring-2 ring-cyan-400/50 z-10'
                        : 'border-slate-700 bg-slate-800/90 hover:border-slate-600 hover:bg-slate-800'
                    }`}
                    style={{
                      left: `${clipLeft}px`,
                      width: `${clipWidth}px`,
                    }}
                  >
                    {/* Filmstrip simulation stripes */}
                    <div className="absolute inset-0 opacity-15 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:12px_12px] pointer-events-none" />

                    {/* Clip content label */}
                    <div className="relative h-full p-2 flex flex-col justify-between z-10">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs font-semibold text-white truncate max-w-[80%]">
                          {clip.title}
                        </span>
                        {clip.linkedToClipId && (
                          <span
                            className="p-0.5 bg-cyan-900/80 text-cyan-300 rounded border border-cyan-700/60 shrink-0"
                            title="Linked continuity from previous clip"
                          >
                            <Link2 className="w-3 h-3" />
                          </span>
                        )}
                      </div>

                      <div className="flex items-center justify-between text-[10px] font-mono text-cyan-300 tabular-nums">
                        <span>{clip.duration.toFixed(1)}s</span>
                        <span className="text-slate-400">{Math.floor(clip.duration * fps) + 1}f</span>
                      </div>
                    </div>

                    {/* Trim Handles on ends */}
                    <div className="absolute left-0 top-0 bottom-0 w-1.5 bg-cyan-500/40 hover:bg-cyan-400 cursor-ew-resize opacity-0 group-hover:opacity-100" />
                    <div className="absolute right-0 top-0 bottom-0 w-1.5 bg-cyan-500/40 hover:bg-cyan-400 cursor-ew-resize opacity-0 group-hover:opacity-100" />
                  </div>
                );
              })}
            </div>

            {/* TRACK 2: Audio Track (Placeholder) */}
            <div className="h-14 border-b border-slate-800/80 relative flex items-center bg-slate-900/10">
              {audioClips.map((audio) => {
                const audioLeft = audio.startTime * zoom;
                const audioWidth = audio.duration * zoom;

                return (
                  <div
                    key={audio.id}
                    className="absolute top-1 bottom-1 rounded-md bg-sky-950/40 border border-sky-800/50 overflow-hidden flex items-center px-2 select-none"
                    style={{
                      left: `${audioLeft}px`,
                      width: `${audioWidth}px`,
                      opacity: isAudioMuted ? 0.4 : 1,
                    }}
                  >
                    {/* Simulated Waveform SVG */}
                    <div className="flex-1 flex items-center gap-0.5 h-6 opacity-60">
                      {Array.from({ length: Math.min(80, Math.floor(audioWidth / 4)) }).map((_, idx) => {
                        const h = 4 + (Math.sin(idx * 0.4) * 0.5 + 0.5) * 16;
                        return (
                          <div
                            key={idx}
                            className="w-0.5 bg-sky-400 rounded-full"
                            style={{ height: `${h}px` }}
                          />
                        );
                      })}
                    </div>
                    <span className="text-[10px] font-medium text-sky-300 ml-2 shrink-0">
                      {audio.name}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* TRACK 3: Captions Track */}
            <div className="h-12 border-b border-slate-800/80 relative flex items-center bg-slate-900/10">
              {captionClips.map((cap) => {
                const capLeft = cap.startTime * zoom;
                const capWidth = cap.duration * zoom;

                return (
                  <div
                    key={cap.id}
                    className="absolute top-1 bottom-1 rounded-md bg-amber-950/40 border border-amber-800/50 px-2 flex items-center truncate text-[11px] text-amber-200 font-sans shadow-sm"
                    style={{
                      left: `${capLeft}px`,
                      width: `${capWidth}px`,
                    }}
                  >
                    <span className="truncate">{cap.text}</span>
                  </div>
                );
              })}
            </div>

            {/* DRAGGABLE PLAYHEAD NEEDLE */}
            <div
              className="absolute top-0 bottom-0 pointer-events-none z-30 flex flex-col items-center"
              style={{
                left: `${currentTime * zoom}px`,
                transform: 'translateX(-50%)',
              }}
            >
              {/* Playhead Marker Head */}
              <div className="w-3.5 h-4 bg-cyan-400 border border-cyan-200 rounded-b shadow-lg shadow-cyan-500/50 flex items-center justify-center">
                <div className="w-1 h-1 bg-slate-950 rounded-full" />
              </div>

              {/* Vertical Laser Needle */}
              <div className="w-0.5 flex-1 bg-cyan-400 shadow-[0_0_8px_rgba(56,189,248,0.8)]" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
