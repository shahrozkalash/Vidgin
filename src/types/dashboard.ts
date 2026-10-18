/**
 * Data contracts and models for the Vidgen Projects and Progress Dashboard
 */

export type ClipStatus = 'queued' | 'generating' | 'done' | 'failed';

export interface ClipContinuity {
  seam: number;        // e.g. 96.5 (% seam smoothness across temporal join)
  identity: number;    // e.g. 94.2 (% DINO visual identity preservation)
  colorDrift: number;  // e.g. 1.8 (% color and illumination delta)
}

export interface Clip {
  id: string;
  projectId: string;
  parentId?: string;   // Previous clip ID if continued
  prompt: string;
  seed: number;
  durationSec: number; // e.g. 5.0
  status: ClipStatus;
  progress: number;    // 0.0 to 1.0
  createdAt: number;   // Epoch ms
  continuity?: ClipContinuity;
  favorite?: boolean;
  thumbnailUrl?: string;
  videoDataUrl?: string;
  startTime?: number;  // Position on timeline
}

export interface Project {
  id: string;
  name: string;
  targetSeconds: number; // e.g. 30 (default 30s target length)
  createdAt: number;     // Epoch ms
  updatedAt: number;     // Epoch ms
  clips: Clip[];
  coverClipId?: string;
}

export type JobType = 'generate' | 'continue' | 'regenerate';

export type JobStatus =
  | 'queued'
  | 'generating'
  | 'finishing'
  | 'done'
  | 'failed'
  | 'canceled';

export interface Job {
  id: string;
  clipId: string;
  projectId: string;
  type: JobType;
  status: JobStatus;
  progress: number;      // 0.0 to 1.0
  step: number;          // Current ODE sampling step (e.g. 23)
  totalSteps: number;    // Total ODE steps (e.g. 50)
  chunkIndex: number;    // 1-indexed (e.g. 1)
  chunkCount: number;    // Total 5s chunks (e.g. 6 for 30s)
  etaSeconds: number;    // Smoothed estimated seconds remaining
  startedAt?: number;
  finishedAt?: number;
  error?: string;
  priority?: number;     // Higher integer = higher priority (default 0)
  contextFrames?: number;// Default 24 pixel frames (6 latent frames)
  simulateFailures?: boolean;
}

export interface JobProgressEvent {
  jobId: string;
  clipId: string;
  status: JobStatus;
  progress: number;
  step: number;
  totalSteps: number;
  chunkIndex: number;
  chunkCount: number;
  etaSeconds: number;
  message?: string;
  error?: string;
}

export interface GenerateClipParams {
  projectId: string;
  prompt: string;
  seed: number;
  durationSec: number;
  priority?: number;
  simulateFailures?: boolean;
  steps?: number;
}

export interface ContinueClipParams {
  projectId: string;
  prevClipId: string;
  prompt: string;
  seed: number;
  durationSec: number;
  contextFrames?: number; // Defaults to 24 pixel frames (6 latent frames)
  priority?: number;
  simulateFailures?: boolean;
  steps?: number;
}

export interface RegenerateClipParams {
  projectId: string;
  clipId: string;
  prompt?: string;
  seed?: number;
  contextFrames?: number;
  priority?: number;
  steps?: number;
}

export interface UsageRecord {
  monthKey: string;             // "YYYY-MM"
  generationsCount: number;
  secondsGenerated: number;
  estimatedComputeMinutes: number;
  lastResetAt: number;
}

export interface DashboardStorageState {
  version: number;
  projects: Project[];
  activeProjectId: string | null;
  jobs: Job[];
  usage: UsageRecord;
  settings: {
    concurrency: number;
    simulateFailures: boolean;
    defaultTargetSeconds: number;
    defaultContextFrames: number;
  };
}
