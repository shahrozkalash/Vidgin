/**
 * Types and interfaces for the VidGen CapCut-style Desktop Timeline Video Editor
 */

export interface EditorClipContinuity {
  seamSmoothness: number;      // e.g. 96 (percentage)
  identitySimilarity: number;  // e.g. 94 (percentage)
  colorDrift: number;          // e.g. 2.4 (percentage)
  isWarning?: boolean;
}

export interface EditorClip {
  id: string;
  title: string;
  prompt: string;
  seed: number;
  startTime: number;           // Seconds from project origin
  duration: number;            // Length in seconds (e.g. 5.0)
  linkedToClipId?: string;     // ID of parent clip continued from
  linkedNextClipId?: string;   // ID of child clip continued to
  continuity?: EditorClipContinuity;
  status: 'ready' | 'generating' | 'queued';
  progress: number;            // 0 - 100
  trackIndex: number;          // 0 = primary video track, 1 = alternate branch
  accentColor?: string;
  thumbnails?: string[];       // Snapshot previews
}

export interface AudioClip {
  id: string;
  name: string;
  startTime: number;
  duration: number;
  volume: number;              // 0.0 - 1.0
  muted: boolean;
}

export interface CaptionClip {
  id: string;
  text: string;
  startTime: number;
  duration: number;
}

export type MemoryCategory = 'character' | 'environment' | 'style';

export interface EditorMemoryItem {
  id: string;
  category: MemoryCategory;
  name: string;
  description: string;
  locked: boolean;
  pinned: boolean;
  avatarColor: string;
  tags: string[];
}

export type LeftPanelTab = 'generate' | 'memory' | 'clips';

export interface EditorHistorySnapshot {
  clips: EditorClip[];
  audioClips: AudioClip[];
  captionClips: CaptionClip[];
  projectName: string;
}
