/**
 * QueueEngine: Concurrency scheduler, priority management, pause/resume,
 * cancellation, and retry management for Vidgen generation jobs.
 */

import { Job, Clip, JobProgressEvent } from '../types/dashboard';
import { GeneratorBackend } from './generatorBackend';

export type QueueEventListener = (jobs: Job[]) => void;
export type JobEventListener = (event: JobProgressEvent) => void;
export type ClipFinishedListener = (clip: Clip) => void;

export class QueueEngine {
  private backend: GeneratorBackend;
  private concurrency: number;
  private isPausedState: boolean = false;
  private jobs: Map<string, Job> = new Map();
  private clips: Map<string, Clip> = new Map();
  private activeJobIds: Set<string> = new Set();
  private canceledJobIds: Set<string> = new Set();

  private queueListeners: Set<QueueEventListener> = new Set();
  private progressListeners: Set<JobEventListener> = new Set();
  private clipFinishedListeners: Set<ClipFinishedListener> = new Set();

  constructor(backend: GeneratorBackend, concurrency: number = 1) {
    this.backend = backend;
    this.concurrency = Math.max(1, concurrency);
  }

  public setConcurrency(concurrency: number): void {
    this.concurrency = Math.max(1, concurrency);
    this.processNext();
  }

  public getConcurrency(): number {
    return this.concurrency;
  }

  public pause(): void {
    this.isPausedState = true;
    this.notifyQueueChanged();
  }

  public resume(): void {
    this.isPausedState = false;
    this.notifyQueueChanged();
    this.processNext();
  }

  public isPaused(): boolean {
    return this.isPausedState;
  }

  public addJob(job: Job, clip: Clip): void {
    this.jobs.set(job.id, { ...job, status: 'queued', progress: 0 });
    this.clips.set(clip.id, { ...clip, status: 'queued', progress: 0 });
    this.canceledJobIds.delete(job.id);
    this.notifyQueueChanged();
    this.processNext();
  }

  public cancelJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) return false;

    if (job.status === 'done' || job.status === 'failed' || job.status === 'canceled') {
      return false;
    }

    this.canceledJobIds.add(jobId);
    job.status = 'canceled';
    job.finishedAt = Date.now();
    this.activeJobIds.delete(jobId);

    const clip = this.clips.get(job.clipId);
    if (clip && clip.status !== 'done') {
      clip.status = 'failed';
    }

    this.notifyProgress({
      jobId,
      clipId: job.clipId,
      status: 'canceled',
      progress: job.progress,
      step: job.step,
      totalSteps: job.totalSteps,
      chunkIndex: job.chunkIndex,
      chunkCount: job.chunkCount,
      etaSeconds: 0,
      message: 'Job canceled',
    });

    this.notifyQueueChanged();
    this.processNext();
    return true;
  }

  public retryJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job) return false;

    const clip = this.clips.get(job.clipId);
    if (!clip) return false;

    // Reset job state
    job.status = 'queued';
    job.progress = 0;
    job.step = 0;
    job.error = undefined;
    job.startedAt = undefined;
    job.finishedAt = undefined;
    this.canceledJobIds.delete(jobId);
    this.activeJobIds.delete(jobId);

    clip.status = 'queued';
    clip.progress = 0;

    this.notifyQueueChanged();
    this.processNext();
    return true;
  }

  public prioritizeJob(jobId: string): boolean {
    const job = this.jobs.get(jobId);
    if (!job || job.status !== 'queued') return false;

    // Raise priority above current max
    const maxPriority = Math.max(0, ...Array.from(this.jobs.values()).map((j) => j.priority ?? 0));
    job.priority = maxPriority + 1;
    this.notifyQueueChanged();
    this.processNext();
    return true;
  }

  public getAllJobs(): Job[] {
    return Array.from(this.jobs.values());
  }

  public getJob(jobId: string): Job | undefined {
    return this.jobs.get(jobId);
  }

  public getClip(clipId: string): Clip | undefined {
    return this.clips.get(clipId);
  }

  public clearFinishedJobs(): void {
    for (const [id, job] of this.jobs.entries()) {
      if (job.status === 'done' || job.status === 'canceled' || job.status === 'failed') {
        this.jobs.delete(id);
      }
    }
    this.notifyQueueChanged();
  }

  public subscribeQueue(listener: QueueEventListener): () => void {
    this.queueListeners.add(listener);
    listener(this.getAllJobs());
    return () => this.queueListeners.delete(listener);
  }

  public subscribeProgress(listener: JobEventListener): () => void {
    this.progressListeners.add(listener);
    return () => this.progressListeners.delete(listener);
  }

  public subscribeClipFinished(listener: ClipFinishedListener): () => void {
    this.clipFinishedListeners.add(listener);
    return () => this.clipFinishedListeners.delete(listener);
  }

  private notifyQueueChanged(): void {
    const jobList = this.getAllJobs();
    for (const listener of this.queueListeners) {
      try {
        listener(jobList);
      } catch (err) {
        console.error('Queue listener error:', err);
      }
    }
  }

  private notifyProgress(event: JobProgressEvent): void {
    for (const listener of this.progressListeners) {
      try {
        listener(event);
      } catch (err) {
        console.error('Progress listener error:', err);
      }
    }
  }

  private notifyClipFinished(clip: Clip): void {
    for (const listener of this.clipFinishedListeners) {
      try {
        listener(clip);
      } catch (err) {
        console.error('Clip finished listener error:', err);
      }
    }
  }

  private getNextQueuedJob(): Job | null {
    if (this.isPausedState) return null;
    if (this.activeJobIds.size >= this.concurrency) return null;

    // Filter queued jobs
    const queued = Array.from(this.jobs.values()).filter((j) => j.status === 'queued');
    if (queued.length === 0) return null;

    // Sort by priority descending (higher integer first), then by started/added
    queued.sort((a, b) => {
      const pDiff = (b.priority ?? 0) - (a.priority ?? 0);
      if (pDiff !== 0) return pDiff;
      return a.id.localeCompare(b.id);
    });

    return queued[0];
  }

  private async processNext(): Promise<void> {
    if (this.isPausedState) return;
    if (this.activeJobIds.size >= this.concurrency) return;

    const nextJob = this.getNextQueuedJob();
    if (!nextJob) return;

    const clip = this.clips.get(nextJob.clipId);
    if (!clip) return;

    this.activeJobIds.add(nextJob.id);
    this.notifyQueueChanged();

    const isJobCanceled = () => this.canceledJobIds.has(nextJob.id);

    try {
      const updatedClip = await this.backend.executeJob(
        nextJob,
        clip,
        (progressEvent) => {
          // Update in-memory job
          nextJob.status = progressEvent.status;
          nextJob.progress = progressEvent.progress;
          nextJob.step = progressEvent.step;
          nextJob.totalSteps = progressEvent.totalSteps;
          nextJob.chunkIndex = progressEvent.chunkIndex;
          nextJob.chunkCount = progressEvent.chunkCount;
          nextJob.etaSeconds = progressEvent.etaSeconds;

          // Update clip
          if (progressEvent.status === 'generating') {
            clip.status = 'generating';
            clip.progress = progressEvent.progress;
          }

          this.notifyProgress(progressEvent);
          this.notifyQueueChanged();
        },
        isJobCanceled
      );

      this.clips.set(updatedClip.id, updatedClip);
      this.notifyClipFinished(updatedClip);
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      if (this.canceledJobIds.has(nextJob.id)) {
        nextJob.status = 'canceled';
      } else {
        nextJob.status = 'failed';
        nextJob.error = errorMsg;
        clip.status = 'failed';
      }
      nextJob.finishedAt = Date.now();
      this.notifyQueueChanged();
    } finally {
      this.activeJobIds.delete(nextJob.id);
      this.notifyQueueChanged();
      // Pump queue for remaining jobs
      setTimeout(() => this.processNext(), 10);
    }
  }
}
