/**
 * GeneratorBackend interface and MockBackend implementation
 * Provides realistic rectified-flow ODE generation simulation with progress events,
 * chunking, smoothed ETA, and continuity modeling.
 */

import {
  Clip,
  Job,
  JobProgressEvent,
  GenerateClipParams,
  ContinueClipParams,
  RegenerateClipParams
} from '../types/dashboard';

export interface GeneratorBackend {
  generateClip(params: GenerateClipParams): { job: Job; clip: Clip };
  continueClip(params: ContinueClipParams): { job: Job; clip: Clip };
  regenerateClip(params: RegenerateClipParams, existingClip: Clip): { job: Job; clip: Clip };
  executeJob(
    job: Job,
    clip: Clip,
    onProgress: (event: JobProgressEvent) => void,
    isCanceled?: () => boolean
  ): Promise<Clip>;
}

export const DEFAULT_CONTINUE_CONTEXT_PIXEL_FRAMES = 24; // 6 latent frames under 4x VAE temporal downsampling

/**
 * Exponential Moving Average (EMA) smoother for step durations
 */
export class ETASmoother {
  private alpha: number;
  private avgStepDurationSec: number | null = null;
  private lastStepTimestamp: number | null = null;

  constructor(alpha: number = 0.25, initialStepDurationSec: number = 0.08) {
    this.alpha = alpha;
    this.avgStepDurationSec = initialStepDurationSec;
  }

  public recordStep(currentTimestamp: number): void {
    if (this.lastStepTimestamp !== null) {
      const stepDuration = Math.max(0.001, (currentTimestamp - this.lastStepTimestamp) / 1000);
      if (this.avgStepDurationSec === null) {
        this.avgStepDurationSec = stepDuration;
      } else {
        this.avgStepDurationSec =
          this.alpha * stepDuration + (1 - this.alpha) * this.avgStepDurationSec;
      }
    }
    this.lastStepTimestamp = currentTimestamp;
  }

  public getEstimatedTimeRemaining(remainingSteps: number): number {
    const stepDuration = this.avgStepDurationSec ?? 0.08;
    return Math.max(0, Math.round(remainingSteps * stepDuration * 10) / 10);
  }

  public getAverageStepDuration(): number {
    return this.avgStepDurationSec ?? 0.08;
  }
}

export class MockBackend implements GeneratorBackend {
  private defaultSteps: number;
  private simulateFailures: boolean;
  private failureRate: number; // 0.0 to 1.0 (default 0.15 when simulateFailures is on)

  constructor(options?: {
    defaultSteps?: number;
    simulateFailures?: boolean;
    failureRate?: number;
  }) {
    this.defaultSteps = options?.defaultSteps ?? 30;
    this.simulateFailures = options?.simulateFailures ?? false;
    this.failureRate = options?.failureRate ?? 0.15;
  }

  public setSimulateFailures(enabled: boolean): void {
    this.simulateFailures = enabled;
  }

  public isSimulateFailuresEnabled(): boolean {
    return this.simulateFailures;
  }

  public generateClip(params: GenerateClipParams): { job: Job; clip: Clip } {
    const clipId = `clip-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    const jobId = `job-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    const totalSteps = params.steps ?? this.defaultSteps;
    const chunkCount = Math.max(1, Math.round(params.durationSec / 5));

    const clip: Clip = {
      id: clipId,
      projectId: params.projectId,
      prompt: params.prompt,
      seed: params.seed,
      durationSec: params.durationSec,
      status: 'queued',
      progress: 0,
      createdAt: Date.now(),
      continuity: {
        seam: 98.0,
        identity: 96.0,
        colorDrift: 1.2,
      },
    };

    const job: Job = {
      id: jobId,
      clipId: clipId,
      projectId: params.projectId,
      type: 'generate',
      status: 'queued',
      progress: 0,
      step: 0,
      totalSteps,
      chunkIndex: 1,
      chunkCount,
      etaSeconds: Math.round(totalSteps * chunkCount * 0.08),
      priority: params.priority ?? 0,
      simulateFailures: params.simulateFailures ?? this.simulateFailures,
    };

    return { job, clip };
  }

  public continueClip(params: ContinueClipParams): { job: Job; clip: Clip } {
    const clipId = `clip-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    const jobId = `job-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    const totalSteps = params.steps ?? this.defaultSteps;
    const chunkCount = Math.max(1, Math.round(params.durationSec / 5));
    const contextFrames =
      params.contextFrames ?? DEFAULT_CONTINUE_CONTEXT_PIXEL_FRAMES;

    // High continuity for continued clips with anchor context
    const clip: Clip = {
      id: clipId,
      projectId: params.projectId,
      parentId: params.prevClipId,
      prompt: params.prompt,
      seed: params.seed,
      durationSec: params.durationSec,
      status: 'queued',
      progress: 0,
      createdAt: Date.now(),
      continuity: {
        seam: 96.4,
        identity: 94.8,
        colorDrift: 1.9,
      },
    };

    const job: Job = {
      id: jobId,
      clipId: clipId,
      projectId: params.projectId,
      type: 'continue',
      status: 'queued',
      progress: 0,
      step: 0,
      totalSteps,
      chunkIndex: 1,
      chunkCount,
      etaSeconds: Math.round(totalSteps * chunkCount * 0.08),
      priority: params.priority ?? 0,
      contextFrames,
      simulateFailures: params.simulateFailures ?? this.simulateFailures,
    };

    return { job, clip };
  }

  public regenerateClip(
    params: RegenerateClipParams,
    existingClip: Clip
  ): { job: Job; clip: Clip } {
    const jobId = `job-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    const totalSteps = params.steps ?? this.defaultSteps;
    const chunkCount = Math.max(1, Math.round(existingClip.durationSec / 5));

    const updatedClip: Clip = {
      ...existingClip,
      prompt: params.prompt ?? existingClip.prompt,
      seed: params.seed ?? existingClip.seed,
      status: 'queued',
      progress: 0,
      continuity: {
        seam: Math.min(99.5, (existingClip.continuity?.seam ?? 95.0) + 2.5),
        identity: Math.min(99.0, (existingClip.continuity?.identity ?? 93.0) + 2.5),
        colorDrift: Math.max(0.5, (existingClip.continuity?.colorDrift ?? 2.5) - 0.8),
      },
    };

    const job: Job = {
      id: jobId,
      clipId: existingClip.id,
      projectId: params.projectId,
      type: 'regenerate',
      status: 'queued',
      progress: 0,
      step: 0,
      totalSteps,
      chunkIndex: 1,
      chunkCount,
      etaSeconds: Math.round(totalSteps * chunkCount * 0.08),
      priority: params.priority ?? 1, // Regenerations get slightly higher priority by default
      contextFrames: params.contextFrames ?? DEFAULT_CONTINUE_CONTEXT_PIXEL_FRAMES,
      simulateFailures: this.simulateFailures,
    };

    return { job, clip: updatedClip };
  }

  public async executeJob(
    job: Job,
    clip: Clip,
    onProgress: (event: JobProgressEvent) => void,
    isCanceled?: () => boolean
  ): Promise<Clip> {
    const smoother = new ETASmoother(0.3, 0.06);
    const totalStepsPerChunk = job.totalSteps;
    const totalChunks = job.chunkCount;
    const overallSteps = totalStepsPerChunk * totalChunks;

    // Fail early or halfway if simulated failure is randomly triggered
    const shouldSimulateFailure =
      (job.simulateFailures ?? this.simulateFailures) && Math.random() < this.failureRate;
    const failureStepTarget = shouldSimulateFailure
      ? Math.max(3, Math.floor(Math.random() * (overallSteps - 4)))
      : -1;

    let currentStepCount = 0;
    job.startedAt = Date.now();
    job.status = 'generating';

    for (let c = 1; c <= totalChunks; c++) {
      for (let s = 1; s <= totalStepsPerChunk; s++) {
        if (isCanceled && isCanceled()) {
          job.status = 'canceled';
          throw new Error('Job canceled by user');
        }

        currentStepCount++;
        smoother.recordStep(Date.now());

        // Check for simulated failure
        if (shouldSimulateFailure && currentStepCount >= failureStepTarget) {
          job.status = 'failed';
          job.error = 'Simulation failure: Flow trajectory divergence at step ' + currentStepCount;
          job.finishedAt = Date.now();
          onProgress({
            jobId: job.id,
            clipId: clip.id,
            status: 'failed',
            progress: currentStepCount / overallSteps,
            step: s,
            totalSteps: totalStepsPerChunk,
            chunkIndex: c,
            chunkCount: totalChunks,
            etaSeconds: 0,
            error: job.error,
          });
          throw new Error(job.error);
        }

        const remainingSteps = overallSteps - currentStepCount;
        const etaSeconds = smoother.getEstimatedTimeRemaining(remainingSteps);
        const progress = Math.min(0.98, currentStepCount / overallSteps);

        job.step = s;
        job.chunkIndex = c;
        job.progress = progress;
        job.etaSeconds = etaSeconds;

        onProgress({
          jobId: job.id,
          clipId: clip.id,
          status: 'generating',
          progress,
          step: s,
          totalSteps: totalStepsPerChunk,
          chunkIndex: c,
          chunkCount: totalChunks,
          etaSeconds,
          message: `Chunk ${c}/${totalChunks} · Step ${s}/${totalStepsPerChunk}`,
        });

        // Fast asynchronous tick to allow UI re-renders and smooth ETA tracking
        await new Promise((resolve) => setTimeout(resolve, 35));
      }
    }

    // Finishing phase
    job.status = 'finishing';
    job.progress = 0.99;
    job.etaSeconds = 0.2;
    onProgress({
      jobId: job.id,
      clipId: clip.id,
      status: 'finishing',
      progress: 0.99,
      step: totalStepsPerChunk,
      totalSteps: totalStepsPerChunk,
      chunkIndex: totalChunks,
      chunkCount: totalChunks,
      etaSeconds: 0.1,
      message: 'Compiling 3D VAE video frames...',
    });

    await new Promise((resolve) => setTimeout(resolve, 60));

    // Completed
    job.status = 'done';
    job.progress = 1.0;
    job.etaSeconds = 0;
    job.finishedAt = Date.now();

    const updatedClip: Clip = {
      ...clip,
      status: 'done',
      progress: 1.0,
      thumbnailUrl: `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180"><rect width="320" height="180" fill="%230f172a"/><circle cx="160" cy="90" r="30" fill="%2338bdf8"/></svg>`,
    };

    onProgress({
      jobId: job.id,
      clipId: clip.id,
      status: 'done',
      progress: 1.0,
      step: totalStepsPerChunk,
      totalSteps: totalStepsPerChunk,
      chunkIndex: totalChunks,
      chunkCount: totalChunks,
      etaSeconds: 0,
      message: 'Generation complete',
    });

    return updatedClip;
  }
}
