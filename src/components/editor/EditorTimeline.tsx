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
  Maximize2
} from 'lucide-react';
import { EditorClip, AudioClip, CaptionClip } from '../../types/editor';

interface EditorTimelineProps {
  currentTime: number;
  totalDuration: number;
  zoom: number; // Pixels per second (e.g. 30 - 160)
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
  height: number;
  onResizeStart?: (e: React.MouseEvent) => void;
  isCompactMode?: boolean;
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
  height,
  onResizeStart,
  isCompactMode = false,
}) => {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [isAudioMuted, setIsAudioMuted] = useState(false);

  // Timeline track total width in pixels (minimum 1000px so it always scrolls smoothly)
  const timelineContentWidth = Math.max(900, (totalDuration + 5) * zoom);

  const formatRulerTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

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

  // Fit to screen calculation
  const handleFitToScreen = () => {
    if (!scrollContainerRef.current) return;
    const availableWidth = scrollContainerRef.current.clientWidth - 40;
    if (availableWidth > 0 && totalDuration > 0) {
      const fittedZoom = Math.max(25, Math.min(160, Math.floor(availableWidth / (totalDuration + 1))));
      onZoomChange(fittedZoom);
    }
  };

  // Generate ruler tick marks based on zoom
  const tickInterval = zoom < 45 ? 5 : zoom < 85 ? 2 : 1;
  const rulerTicks = [];
  for (let sec = 0; sec <= totalDuration + 6; sec += tickInterval) {
    rulerTicks.push(sec);
  }

  const selectedClip = clips.find((c) => c.id === selectedClipId);

  return (
    <div
      className="bg-slate-900 border-t border-slate-800 flex flex-col shrink-0 select-none overflow-hidden relative w-full"
      style={{ height: `${height}px` }}
    >
      {/* Top Resize Drag Handle */}
      {onResizeStart && (
        <div
          onMouseDown={onResizeStart}
          className="h-1.5 w-full bg-slate-800/60 hover:bg-cyan-500/80 cursor-ns-resize transition-colors absolute top-0 left-0 right-0 z-30 flex items-center justify-center group"
          title="Drag up or down to resize timeline"
        >
          <div className="w-12 h-0.5 bg-slate-600 group-hover:bg-cyan-300 rounded-full" />
        </div>
      )}

      {/* Timeline Controls & Action Bar (Height 36px) */}
      <div className="h-9 bg-slate-950/80 border-b border-slate-800 px-3 flex items-center justify-between gap-2 shrink-0 text-xs text-slate-400">
        {/* Left Actions: Split & Delete */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => selectedClipId && onSplitClip(selectedClipId)}
            disabled={!selectedClipId}
            className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs transition-colors ${
              selectedClipId
                ? 'bg-slate-800 text-slate-200 hover:bg-slate-700 hover:text-white'
                : 'text-slate-600 bg-slate-900 cursor-not-allowed'
            }`}
            title="Split selected clip at playhead (S)"
          >
            <Scissors className="w-3 h-3 text-amber-400" />
            <span className="hidden sm:inline">Split (S)</span>
          </button>

          <button
            onClick={() => selectedClipId && onDeleteClip(selectedClipId)}
            disabled={!selectedClipId}
            className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs transition-colors ${
              selectedClipId
                ? 'bg-red-950/40 text-red-300 hover:bg-red-900/60 border border-red-900/40'
                : 'text-slate-600 bg-slate-900 cursor-not-allowed'
            }`}
            title="Delete selected clip (Del)"
          >
            <Trash2 className="w-3 h-3 text-red-400" />
            <span className="hidden sm:inline">Delete</span>
          </button>

          {selectedClip && (
            <span className="hidden md:inline text-[11px] text-slate-400 truncate max-w-[160px] lg:max-w-[240px] ml-1">
              <strong className="text-slate-200">{selectedClip.title}</strong>
            </span>
          )}
        </div>

        {/* Right Actions: Fit-to-screen + Zoom Controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={handleFitToScreen}
            className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-cyan-300 px-2 py-0.5 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 transition-colors"
            title="Fit timeline to screen width"
          >
            <Maximize2 className="w-3 h-3 text-cyan-400" />
            <span className="hidden sm:inline">Fit</span>
          </button>

          <div className="h-3 w-px bg-slate-800 mx-0.5" />

          <button
            onClick={() => onZoomChange(Math.max(25, zoom - 15))}
            className="p-1 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
            title="Zoom out"
          >
            <ZoomOut className="w-3 h-3" />
          </button>

          <input
            type="range"
            min={25}
            max={160}
            value={zoom}
            onChange={(e) => onZoomChange(parseInt(e.target.value))}
            className="w-16 sm:w-20 h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
            title={`Timeline zoom: ${zoom}px / s`}
          />

          <button
            onClick={() => onZoomChange(Math.min(160, zoom + 15))}
            className="p-1 hover:text-slate-200 hover:bg-slate-800 rounded transition-colors"
            title="Zoom in"
          >
            <ZoomIn className="w-3 h-3" />
          </button>

          <span className="text-[10px] text-slate-500 font-mono w-8 text-right hidden sm:inline">{zoom}px</span>
        </div>
      </div>

      {/* Main Timeline Workspace: Left Track Labels Column (narrow ~96px) + Right Scrollable Tracks */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Track Labels Column (~96px wide, icons only below 1100px) */}
        <div className="w-20 lg:w-24 bg-slate-950 border-r border-slate-800 flex flex-col shrink-0 z-10 select-none">
          {/* Header Spacer */}
          <div className="h-6 border-b border-slate-800 bg-slate-950 flex items-center justify-center lg:justify-start lg:px-2 text-[9px] font-mono text-slate-500">
            TRACKS
          </div>

          {/* Video Track Label */}
          <div className="h-16 border-b border-slate-800/80 flex items-center justify-center lg:justify-between px-2 text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5" title="Video Track 1">
              <Video className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
              <span className="hidden lg:inline text-xs">Video</span>
            </div>
            <span className="hidden lg:inline text-[9px] text-slate-500 font-mono">1</span>
          </div>

          {/* Audio Track Label */}
          <div className="h-12 border-b border-slate-800/80 flex items-center justify-center lg:justify-between px-2 text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5" title="Audio Track">
              <Music className="w-3.5 h-3.5 text-sky-400 shrink-0" />
              <span className="hidden lg:inline text-xs">Audio</span>
            </div>
            <button
              onClick={() => setIsAudioMuted(!isAudioMuted)}
              className="p-0.5 hover:text-slate-100 text-slate-500 transition-colors"
              title={isAudioMuted ? 'Unmute' : 'Mute'}
            >
              {isAudioMuted ? <VolumeX className="w-3 h-3 text-amber-400" /> : <Volume2 className="w-3 h-3" />}
            </button>
          </div>

          {/* Captions Track Label */}
          <div className="h-10 border-b border-slate-800/80 flex items-center justify-center lg:justify-between px-2 text-xs font-medium text-slate-300 bg-slate-900/50">
            <div className="flex items-center gap-1.5" title="Captions Track">
              <Type className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="hidden lg:inline text-xs">Captions</span>
            </div>
            <span className="hidden lg:inline text-[9px] text-slate-500 font-mono">CC</span>
          </div>
        </div>

        {/* Right: Scrollable Timeline Tracks Container with Visible Horizontal Scrollbar */}
        <div
          ref={scrollContainerRef}
          className="flex-1 overflow-x-scroll overflow-y-hidden relative bg-slate-950 select-none [scrollbar-color:#334155_#020617] [scrollbar-width:thin]"
          style={{ scrollbarGutter: 'stable' }}
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

            {/* TRACK 1: Video Track (Height 64px) */}
            <div className="h-16 border-b border-slate-800/80 relative flex items-center bg-slate-900/20">
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
                        ? 'border-cyan-400 bg-cyan-950/70 shadow-lg shadow-cyan-950/40 ring-1 ring-cyan-400 z-10'
                        : 'border-slate-700 bg-slate-800/90 hover:border-slate-600 hover:bg-slate-800'
                    }`}
                    style={{
                      left: `${clipLeft}px`,
                      width: `${clipWidth}px`,
                    }}
                  >
                    <div className="absolute inset-0 opacity-15 bg-[radial-gradient(#38bdf8_1px,transparent_1px)] [background-size:12px_12px] pointer-events-none" />

                    <div className="relative h-full p-1.5 flex flex-col justify-between z-10">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-[11px] font-semibold text-white truncate max-w-[85%]">
                          {clip.title}
                        </span>
                        {clip.linkedToClipId && (
                          <span
                            className="p-0.5 bg-cyan-900/80 text-cyan-300 rounded border border-cyan-700/60 shrink-0"
                            title="Linked continuity from previous clip"
                          >
                            <Link2 className="w-2.5 h-2.5" />
                          </span>
                        )}
                      </div>

                      <div className="flex items-center justify-between text-[9px] font-mono text-cyan-300 tabular-nums">
                        <span>{clip.duration.toFixed(1)}s</span>
                        <span className="text-slate-400">{Math.floor(clip.duration * fps) + 1}f</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* TRACK 2: Audio Track (Height 48px) */}
            <div className="h-12 border-b border-slate-800/80 relative flex items-center bg-slate-900/10">
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
                    <div className="flex-1 flex items-center gap-0.5 h-5 opacity-60">
                      {Array.from({ length: Math.min(80, Math.floor(audioWidth / 4)) }).map((_, idx) => {
                        const h = 3 + (Math.sin(idx * 0.4) * 0.5 + 0.5) * 12;
                        return (
                          <div
                            key={idx}
                            className="w-0.5 bg-sky-400 rounded-full"
                            style={{ height: `${h}px` }}
                          />
                        );
                      })}
                    </div>
                    <span className="text-[9px] font-medium text-sky-300 ml-1.5 shrink-0 truncate">
                      {audio.name}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* TRACK 3: Captions Track (Height 40px) */}
            <div className="h-10 border-b border-slate-800/80 relative flex items-center bg-slate-900/10">
              {captionClips.map((cap) => {
                const capLeft = cap.startTime * zoom;
                const capWidth = cap.duration * zoom;

                return (
                  <div
                    key={cap.id}
                    className="absolute top-1 bottom-1 rounded-md bg-amber-950/40 border border-amber-800/50 px-2 flex items-center truncate text-[10px] text-amber-200 font-sans shadow-sm"
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
              <div className="w-3 h-3.5 bg-cyan-400 border border-cyan-200 rounded-b shadow-lg shadow-cyan-500/50 flex items-center justify-center">
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
