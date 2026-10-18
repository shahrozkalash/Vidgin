/**
 * Unit tests for Vidgen Dashboard and Progress Engine
 * Tests status transitions, concurrency scheduling, ETA smoothing,
 * progress aggregation, monthly rollover, and storage contracts.
 */

import { MockBackend, ETASmoother, DEFAULT_CONTINUE_CONTEXT_PIXEL_FRAMES } from '../generatorBackend';
import { QueueEngine } from '../queueEngine';
import {
  calculateProjectProgress,
  calculateSegmentedProgress,
  calculateAverageContinuity,
  processMonthlyUsage,
  recordGenerationUsage,
  estimateTimeRemaining
} from '../dashboardAggregation';
import {
  exportProjectToJson,
  importProjectFromJson,
  createDefaultStorageState
} from '../dashboardStorage';
import { Project, Clip, Job, UsageRecord } from '../../types/dashboard';

export async function runDashboardEngineTests(): Promise<{
  passed: number;
  failed: number;
  results: { name: string; passed: boolean; message?: string }[];
}> {
  const results: { name: string; passed: boolean; message?: string }[] = [];
  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, testName: string, detail?: string) {
    if (condition) {
      passed++;
      results.push({ name: testName, passed: true });
    } else {
      failed++;
      results.push({ name: testName, passed: false, message: detail ?? 'Assertion failed' });
    }
  }

  // ----------------------------------------------------
  // TEST 1: ETA Smoothing Exponential Moving Average
  // ----------------------------------------------------
  try {
    const smoother = new ETASmoother(0.5, 0.1);
    const t0 = 1000;
    smoother.recordStep(t0);
    smoother.recordStep(t0 + 100); // 100ms step
    smoother.recordStep(t0 + 200); // 100ms step

    const eta10 = smoother.getEstimatedTimeRemaining(10);
    assert(eta10 > 0.8 && eta10 < 1.2, 'ETA Smoother calculates stable remaining time', `Got ${eta10}`);
    assert(DEFAULT_CONTINUE_CONTEXT_PIXEL_FRAMES === 24, 'Default Continue context is 24 pixel frames (6 latent frames)');
  } catch (err) {
    assert(false, 'ETA Smoother error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 2: Progress Aggregation (Done vs Target)
  // ----------------------------------------------------
  try {
    const sampleProject: Project = {
      id: 'p-test',
      name: 'Test Project',
      targetSeconds: 30,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      clips: [
        { id: 'c1', projectId: 'p-test', prompt: 'Scene 1', seed: 1, durationSec: 5, status: 'done', progress: 1, createdAt: 1 },
        { id: 'c2', projectId: 'p-test', prompt: 'Scene 2', seed: 2, durationSec: 5, status: 'done', progress: 1, createdAt: 2 },
        { id: 'c3', projectId: 'p-test', prompt: 'Scene 3', seed: 3, durationSec: 5, status: 'done', progress: 1, createdAt: 3 },
        { id: 'c4', projectId: 'p-test', prompt: 'Scene 4', seed: 4, durationSec: 5, status: 'generating', progress: 0.5, createdAt: 4 },
        { id: 'c5', projectId: 'p-test', prompt: 'Scene 5', seed: 5, durationSec: 5, status: 'queued', progress: 0, createdAt: 5 },
      ],
    };

    const summary = calculateProjectProgress(sampleProject);
    assert(summary.doneSeconds === 15, 'Project progress counts exactly done seconds (15s)', `Got ${summary.doneSeconds}`);
    assert(summary.targetSeconds === 30, 'Target seconds is 30s');
    assert(summary.progressPercent === 50, 'Progress percentage is 50%', `Got ${summary.progressPercent}`);
    assert(summary.isTargetMet === false, 'Target met is false at 15s / 30s');

    // Add remaining 15s done
    sampleProject.clips.push(
      { id: 'c6', projectId: 'p-test', prompt: 'Scene 6', seed: 6, durationSec: 15, status: 'done', progress: 1, createdAt: 6 }
    );
    const summary2 = calculateProjectProgress(sampleProject);
    assert(summary2.isTargetMet === true, 'Target met is true at 30s / 30s');
    assert(summary2.progressPercent === 100, 'Progress percentage caps cleanly at 100%');
  } catch (err) {
    assert(false, 'Progress Aggregation error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 3: Segmented Progress Breakdown
  // ----------------------------------------------------
  try {
    const clips: Clip[] = [
      { id: '1', projectId: 'p', prompt: 'A', seed: 1, durationSec: 5, status: 'done', progress: 1, createdAt: 1 },
      { id: '2', projectId: 'p', prompt: 'B', seed: 2, durationSec: 5, status: 'generating', progress: 0.4, createdAt: 2 },
      { id: '3', projectId: 'p', prompt: 'C', seed: 3, durationSec: 5, status: 'queued', progress: 0, createdAt: 3 },
      { id: '4', projectId: 'p', prompt: 'D', seed: 4, durationSec: 5, status: 'failed', progress: 0, createdAt: 4 },
    ];

    const segmented = calculateSegmentedProgress(clips);
    const doneItem = segmented.find((s) => s.status === 'done');
    const genItem = segmented.find((s) => s.status === 'generating');
    const queueItem = segmented.find((s) => s.status === 'queued');
    const failItem = segmented.find((s) => s.status === 'failed');

    assert(doneItem?.count === 1 && doneItem.seconds === 5, 'Segmented status tracks done clips');
    assert(genItem?.count === 1 && genItem.seconds === 5, 'Segmented status tracks generating clips');
    assert(queueItem?.count === 1 && queueItem.seconds === 5, 'Segmented status tracks queued clips');
    assert(failItem?.count === 1 && failItem.seconds === 5, 'Segmented status tracks failed clips');
  } catch (err) {
    assert(false, 'Segmented progress error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 4: Continuity Averages
  // ----------------------------------------------------
  try {
    const clipsWithContinuity: Clip[] = [
      { id: 'c1', projectId: 'p', prompt: 'A', seed: 1, durationSec: 5, status: 'done', progress: 1, createdAt: 1 },
      { id: 'c2', projectId: 'p', parentId: 'c1', prompt: 'B', seed: 2, durationSec: 5, status: 'done', progress: 1, createdAt: 2, continuity: { seam: 96.0, identity: 94.0, colorDrift: 1.5 } },
      { id: 'c3', projectId: 'p', parentId: 'c2', prompt: 'C', seed: 3, durationSec: 5, status: 'done', progress: 1, createdAt: 3, continuity: { seam: 94.0, identity: 92.0, colorDrift: 2.5 } },
    ];

    const cont = calculateAverageContinuity(clipsWithContinuity);
    assert(cont.sampleCount === 2, 'Counts linked clips only (2 links)');
    assert(cont.seam === 95.0, 'Average seam smoothness calculated correctly (95.0%)', `Got ${cont.seam}`);
    assert(cont.identity === 93.0, 'Average identity similarity calculated correctly (93.0%)', `Got ${cont.identity}`);
    assert(cont.colorDrift === 2.0, 'Average color drift calculated correctly (2.0%)', `Got ${cont.colorDrift}`);
    assert(cont.qualityGrade === 'good', 'Quality grade resolves to good');
  } catch (err) {
    assert(false, 'Continuity averages error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 5: Monthly Usage Rollover
  // ----------------------------------------------------
  try {
    const prevMonthRecord: UsageRecord = {
      monthKey: '2026-09',
      generationsCount: 42,
      secondsGenerated: 210,
      estimatedComputeMinutes: 75.5,
      lastResetAt: 1725148800000,
    };

    // Simulate current date in October 2026
    const octDate = new Date(2026, 9, 18); // Month index 9 = October
    const rollover = processMonthlyUsage(prevMonthRecord, octDate);

    assert(rollover.hasRolledOver === true, 'Detects new calendar month and triggers rollover');
    assert(rollover.usage.monthKey === '2026-10', 'Sets active monthKey to 2026-10');
    assert(rollover.usage.generationsCount === 0, 'Resets generationsCount to 0');
    assert(rollover.usage.secondsGenerated === 0, 'Resets secondsGenerated to 0');

    // Record a generation in October
    const updated = recordGenerationUsage(rollover.usage, 5.0, 30);
    assert(updated.generationsCount === 1, 'Increments generations count to 1');
    assert(updated.secondsGenerated === 5.0, 'Increments secondsGenerated by 5s');
    assert(updated.estimatedComputeMinutes > 0, 'Calculates compute minutes');

    // Calling processMonthlyUsage on same month preserves records
    const noRollover = processMonthlyUsage(updated, octDate);
    assert(noRollover.hasRolledOver === false, 'Same month preserves existing usage counts');
    assert(noRollover.usage.generationsCount === 1, 'Maintains accumulated generations');
  } catch (err) {
    assert(false, 'Monthly usage rollover error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 6: Queue Engine Concurrency & Priority Scheduling
  // ----------------------------------------------------
  try {
    const backend = new MockBackend({ defaultSteps: 4, simulateFailures: false });
    const queue = new QueueEngine(backend, 1);

    const { job: jobLow, clip: clipLow } = backend.generateClip({
      projectId: 'p',
      prompt: 'Low priority',
      seed: 1,
      durationSec: 5,
      priority: 0,
      steps: 4,
    });

    const { job: jobHigh, clip: clipHigh } = backend.generateClip({
      projectId: 'p',
      prompt: 'High priority',
      seed: 2,
      durationSec: 5,
      priority: 10,
      steps: 4,
    });

    // Pause queue to inspect priority sorting before execution
    queue.pause();
    assert(queue.isPaused() === true, 'Queue pause state verified');

    queue.addJob(jobLow, clipLow);
    queue.addJob(jobHigh, clipHigh);

    const allJobs = queue.getAllJobs();
    assert(allJobs.length === 2, 'Queue holds both added jobs');

    // Prioritize low job
    queue.prioritizeJob(jobLow.id);
    const updatedJobLow = queue.getJob(jobLow.id);
    assert((updatedJobLow?.priority ?? 0) > (jobHigh.priority ?? 0), 'prioritizeJob elevates priority above max');

    // Cancel job
    const canceled = queue.cancelJob(jobHigh.id);
    assert(canceled === true, 'cancelJob succeeds on queued job');
    assert(queue.getJob(jobHigh.id)?.status === 'canceled', 'Canceled job status is canceled');

    // Retry job
    const retried = queue.retryJob(jobHigh.id);
    assert(retried === true, 'retryJob succeeds on canceled job');
    assert(queue.getJob(jobHigh.id)?.status === 'queued', 'Retried job status returns to queued');
  } catch (err) {
    assert(false, 'Queue Engine error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 7: MockBackend Execution & Status Transitions
  // ----------------------------------------------------
  try {
    const backend = new MockBackend({ defaultSteps: 3, simulateFailures: false });
    const { job, clip } = backend.generateClip({
      projectId: 'p',
      prompt: 'Fast scene',
      seed: 42,
      durationSec: 5,
      steps: 3,
    });

    const progressStatuses: string[] = [];
    const updatedClip = await backend.executeJob(job, clip, (event) => {
      progressStatuses.push(event.status);
    });

    assert(progressStatuses.includes('generating'), 'Emits generating progress events');
    assert(progressStatuses.includes('finishing'), 'Emits finishing progress events');
    assert(progressStatuses.includes('done'), 'Emits done completion progress event');
    assert(job.status === 'done', 'Final job status is done');
    assert(updatedClip.status === 'done', 'Final clip status is done');
    assert(job.progress === 1.0, 'Final progress reaches 1.0');
  } catch (err) {
    assert(false, 'Execution status transitions error: ' + String(err));
  }

  // ----------------------------------------------------
  // TEST 8: JSON Export and Import
  // ----------------------------------------------------
  try {
    const defaultState = createDefaultStorageState();
    const projectToExport = defaultState.projects[0];
    const exportedJson = exportProjectToJson(projectToExport);

    assert(typeof exportedJson === 'string' && exportedJson.length > 50, 'exportProjectToJson generates JSON string');

    const importedProject = importProjectFromJson(exportedJson);
    assert(importedProject.name.includes(projectToExport.name), 'importProjectFromJson preserves project name');
    assert(importedProject.clips.length === projectToExport.clips.length, 'importProjectFromJson preserves all clips');
    assert(importedProject.targetSeconds === projectToExport.targetSeconds, 'importProjectFromJson preserves target seconds');
  } catch (err) {
    assert(false, 'JSON Export/Import error: ' + String(err));
  }

  return { passed, failed, results };
}
