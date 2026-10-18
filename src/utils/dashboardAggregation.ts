/**
 * Progress aggregation functions and usage rollover utilities for Vidgen Dashboard
 */

import { Project, Clip, Job, UsageRecord, ClipStatus } from '../types/dashboard';

export interface ProjectProgressSummary {
  totalSeconds: number;        // Total seconds across all clips in project
  doneSeconds: number;         // Seconds of clips with status === 'done'
  targetSeconds: number;       // Project target length (e.g. 30s)
  progressRatio: number;       // 0.0 to 1.0 (clamped)
  progressPercent: number;     // 0 to 100
  isTargetMet: boolean;
  totalClipsCount: number;
  doneClipsCount: number;
}

export interface SegmentedProgressItem {
  status: ClipStatus;
  count: number;
  seconds: number;
  percent: number;             // Percentage of total duration or target
  colorClass: string;          // Tailwind color class
}

export interface ContinuityAverages {
  seam: number;                // e.g. 96.5%
  identity: number;            // e.g. 94.2%
  colorDrift: number;          // e.g. 1.8%
  sampleCount: number;
  qualityGrade: 'excellent' | 'good' | 'fair' | 'needs-attention';
}

export interface MonthlyUsageSummary {
  monthKey: string;            // "YYYY-MM"
  generationsThisMonth: number;
  secondsGeneratedThisMonth: number;
  estimatedComputeMinutes: number;
  hasRolledOver: boolean;
}

/**
 * Calculates overall project progress against target seconds
 */
export function calculateProjectProgress(project: Project): ProjectProgressSummary {
  const clips = project.clips || [];
  const targetSeconds = Math.max(1, project.targetSeconds || 30);

  let totalSeconds = 0;
  let doneSeconds = 0;
  let doneClipsCount = 0;

  for (const clip of clips) {
    totalSeconds += clip.durationSec;
    if (clip.status === 'done') {
      doneSeconds += clip.durationSec;
      doneClipsCount++;
    }
  }

  const progressRatio = Math.min(1.0, doneSeconds / targetSeconds);
  const progressPercent = Math.min(100, Math.round(progressRatio * 100));

  return {
    totalSeconds,
    doneSeconds,
    targetSeconds,
    progressRatio,
    progressPercent,
    isTargetMet: doneSeconds >= targetSeconds,
    totalClipsCount: clips.length,
    doneClipsCount,
  };
}

/**
 * Calculates segmented progress breakdown by clip status
 * (amber for queued, cyan for generating, green for done, red for failed)
 */
export function calculateSegmentedProgress(clips: Clip[]): SegmentedProgressItem[] {
  const buckets: Record<ClipStatus, { count: number; seconds: number }> = {
    queued: { count: 0, seconds: 0 },
    generating: { count: 0, seconds: 0 },
    done: { count: 0, seconds: 0 },
    failed: { count: 0, seconds: 0 },
  };

  let totalSeconds = 0;
  for (const clip of clips) {
    const sec = clip.durationSec || 5;
    totalSeconds += sec;
    if (buckets[clip.status]) {
      buckets[clip.status].count++;
      buckets[clip.status].seconds += sec;
    }
  }

  const safeTotal = Math.max(1, totalSeconds);

  return [
    {
      status: 'done',
      count: buckets.done.count,
      seconds: buckets.done.seconds,
      percent: Math.round((buckets.done.seconds / safeTotal) * 100),
      colorClass: 'bg-emerald-400',
    },
    {
      status: 'generating',
      count: buckets.generating.count,
      seconds: buckets.generating.seconds,
      percent: Math.round((buckets.generating.seconds / safeTotal) * 100),
      colorClass: 'bg-cyan-400',
    },
    {
      status: 'queued',
      count: buckets.queued.count,
      seconds: buckets.queued.seconds,
      percent: Math.round((buckets.queued.seconds / safeTotal) * 100),
      colorClass: 'bg-amber-400',
    },
    {
      status: 'failed',
      count: buckets.failed.count,
      seconds: buckets.failed.seconds,
      percent: Math.round((buckets.failed.seconds / safeTotal) * 100),
      colorClass: 'bg-rose-400',
    },
  ];
}

/**
 * Estimates total remaining time across active and queued jobs
 */
export function estimateTimeRemaining(jobs: Job[]): number {
  let totalRemainingSec = 0;

  for (const job of jobs) {
    if (job.status === 'generating' || job.status === 'finishing') {
      totalRemainingSec += job.etaSeconds || 2;
    } else if (job.status === 'queued') {
      // Estimated default duration for a queued job: totalSteps * chunks * 0.08
      const steps = job.totalSteps || 30;
      const chunks = job.chunkCount || 1;
      totalRemainingSec += steps * chunks * 0.08;
    }
  }

  return Math.round(totalRemainingSec * 10) / 10;
}

/**
 * Calculates average continuity scores across clips
 */
export function calculateAverageContinuity(clips: Clip[]): ContinuityAverages {
  const linkedClips = clips.filter((c) => c.continuity && c.parentId);

  if (linkedClips.length === 0) {
    // If no links or single clip, baseline nominal scores
    return {
      seam: 98.0,
      identity: 96.0,
      colorDrift: 1.2,
      sampleCount: 0,
      qualityGrade: 'excellent',
    };
  }

  let totalSeam = 0;
  let totalIdentity = 0;
  let totalColorDrift = 0;

  for (const clip of linkedClips) {
    if (clip.continuity) {
      totalSeam += clip.continuity.seam;
      totalIdentity += clip.continuity.identity;
      totalColorDrift += clip.continuity.colorDrift;
    }
  }

  const n = linkedClips.length;
  const avgSeam = Math.round((totalSeam / n) * 10) / 10;
  const avgIdentity = Math.round((totalIdentity / n) * 10) / 10;
  const avgColorDrift = Math.round((totalColorDrift / n) * 10) / 10;

  let qualityGrade: ContinuityAverages['qualityGrade'] = 'excellent';
  if (avgIdentity < 80 || avgSeam < 85 || avgColorDrift > 6.0) {
    qualityGrade = 'needs-attention';
  } else if (avgIdentity < 90 || avgSeam < 92 || avgColorDrift > 3.5) {
    qualityGrade = 'fair';
  } else if (avgIdentity < 94 || avgSeam < 95) {
    qualityGrade = 'good';
  }

  return {
    seam: avgSeam,
    identity: avgIdentity,
    colorDrift: avgColorDrift,
    sampleCount: n,
    qualityGrade,
  };
}

/**
 * Calculates monthly usage and handles month-boundary rollovers
 */
export function processMonthlyUsage(
  currentRecord: UsageRecord | null,
  currentDate: Date = new Date()
): { usage: UsageRecord; hasRolledOver: boolean } {
  const year = currentDate.getFullYear();
  const month = String(currentDate.getMonth() + 1).padStart(2, '0');
  const activeMonthKey = `${year}-${month}`;

  if (!currentRecord || currentRecord.monthKey !== activeMonthKey) {
    // New month rollover
    const freshRecord: UsageRecord = {
      monthKey: activeMonthKey,
      generationsCount: 0,
      secondsGenerated: 0,
      estimatedComputeMinutes: 0,
      lastResetAt: currentDate.getTime(),
    };
    return { usage: freshRecord, hasRolledOver: true };
  }

  return { usage: currentRecord, hasRolledOver: false };
}

/**
 * Updates usage metrics when a clip finishes generation
 */
export function recordGenerationUsage(
  currentUsage: UsageRecord,
  clipDurationSec: number,
  steps: number = 30
): UsageRecord {
  // Approximate compute minutes: (seconds * steps * resolutionFactor) / 60
  // For 480p at 16fps: factor ≈ 0.45 relative to single-device H100 compute
  const addedComputeMin = Math.round(((clipDurationSec * steps * 0.45) / 60) * 10) / 10;

  return {
    ...currentUsage,
    generationsCount: currentUsage.generationsCount + 1,
    secondsGenerated: currentUsage.secondsGenerated + clipDurationSec,
    estimatedComputeMinutes: Math.round((currentUsage.estimatedComputeMinutes + addedComputeMin) * 10) / 10,
  };
}
