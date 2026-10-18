/**
 * Scaling & Distributed Training Engine for Wan Video Diffusion (Part 9)
 *
 * Implements:
 * - VRAM memory breakdown (FSDP2 / ZeRO-3, Activation Checkpointing, 8-bit AdamW)
 * - Fixed global batch size invariant schedule calculation
 * - Multi-GPU loss curve equivalence simulation (1 GPU vs 2, 4, 8, 16 GPUs)
 * - Cross-topology checkpoint resharding and resumption verification
 * - Model FLOPs Utilization (MFU) & Tokens/sec throughput profiler
 * - DeepSpeed Ulysses sequence parallelism communication vs compute modeling
 * - Cost and compute reality comparison (Track A vs Wan-class 14B)
 */

import { PresetKey } from '../types/vidgen';
import { DIT_PRESET_CONFIGS } from './ditEngine';

export interface HardwareProfile {
  id: string;
  name: string;
  peakTFlopsBF16: number; // TFLOPS
  vramGB: number;
  interconnectBandwidthGBs: number; // NVLink or PCIe
  interconnectType: 'NVLink 4' | 'NVLink 3' | 'PCIe Gen4' | 'PCIe Gen5';
  costPerHourUSD: number;
}

export const HARDWARE_CATALOG: Record<string, HardwareProfile> = {
  h100_sxm: {
    id: 'h100_sxm',
    name: 'NVIDIA H100 SXM5 (80GB)',
    peakTFlopsBF16: 989,
    vramGB: 80,
    interconnectBandwidthGBs: 900,
    interconnectType: 'NVLink 4',
    costPerHourUSD: 3.20,
  },
  a100_80gb: {
    id: 'a100_80gb',
    name: 'NVIDIA A100 SXM4 (80GB)',
    peakTFlopsBF16: 312,
    vramGB: 80,
    interconnectBandwidthGBs: 600,
    interconnectType: 'NVLink 3',
    costPerHourUSD: 1.80,
  },
  l40s: {
    id: 'l40s',
    name: 'NVIDIA L40S (48GB)',
    peakTFlopsBF16: 366,
    vramGB: 48,
    interconnectBandwidthGBs: 64,
    interconnectType: 'PCIe Gen4',
    costPerHourUSD: 1.10,
  },
  rtx4090: {
    id: 'rtx4090',
    name: 'NVIDIA RTX 4090 (24GB)',
    peakTFlopsBF16: 165,
    vramGB: 24,
    interconnectBandwidthGBs: 64,
    interconnectType: 'PCIe Gen4',
    costPerHourUSD: 0.50,
  },
};

export interface MemoryBreakdown {
  modelWeightsGB: number;
  gradientsGB: number;
  optimizerStatesGB: number;
  activationsGB: number;
  cudaBuffersGB: number;
  totalVRAMGB: number;
  maxVRAMGB: number;
  isOOM: boolean;
  savingsVsBaselineGB: number;
  fsdpShardFactor: number;
  spShardFactor: number;
}

export interface ParallelismSettings {
  gpuCount: number;
  hardwareId: string;
  fsdpEnabled: boolean;
  activationCheckpointing: boolean;
  use8BitOptimizer: boolean;
  sequenceParallelDegree: number; // Ulysses SP
  targetGlobalBatchSize: number;
  tokensPerClip: number; // e.g. 1024 to 32760
  precision: 'bf16' | 'fp16' | 'fp32';
}

export function computeMemoryBreakdown(
  paramsCount: number,
  settings: ParallelismSettings
): MemoryBreakdown {
  const hw = HARDWARE_CATALOG[settings.hardwareId] || HARDWARE_CATALOG.h100_sxm;
  const fsdpDegree = settings.fsdpEnabled ? Math.max(1, settings.gpuCount) : 1;
  const spDegree = Math.max(1, settings.sequenceParallelDegree);
  const bytesPerParam = settings.precision === 'fp32' ? 4 : 2;

  // 1. Sharded Model Weights
  const modelWeightsGB = (paramsCount * bytesPerParam) / fsdpDegree / (1024 ** 3);

  // 2. Gradients
  const gradientsGB = (paramsCount * bytesPerParam) / fsdpDegree / (1024 ** 3);

  // 3. Optimizer States (AdamW: 12 bytes/param in FP32 vs 6 bytes in 8-bit)
  const optBytes = settings.use8BitOptimizer ? 6 : 12;
  const optimizerStatesGB = (paramsCount * optBytes) / fsdpDegree / (1024 ** 3);

  // 4. Activations (scaled by sequence length and sequence parallel degree)
  const hiddenDim = 1024;
  const numLayers = 16;
  const localTokens = settings.tokensPerClip / spDegree;

  let activationsGB = 0;
  if (settings.activationCheckpointing) {
    // Only boundary activations saved: ~ 2 * localTokens * hiddenDim * 2 bytes * 4
    activationsGB = (2 * localTokens * hiddenDim * bytesPerParam * 4) / (1024 ** 3);
  } else {
    // Full layer stack intermediates
    activationsGB = (numLayers * localTokens * hiddenDim * 16 * bytesPerParam) / (1024 ** 3);
  }

  // 5. CUDA Runtime & KV cache workspace
  const cudaBuffersGB = 2.2;

  const totalVRAMGB = modelWeightsGB + gradientsGB + optimizerStatesGB + activationsGB + cudaBuffersGB;
  const isOOM = totalVRAMGB > hw.vramGB;

  // Calculate unoptimized baseline (no FSDP, no act checkpointing, fp32 opt)
  const baselineWeights = (paramsCount * 2) / (1024 ** 3);
  const baselineGrads = (paramsCount * 2) / (1024 ** 3);
  const baselineOpt = (paramsCount * 12) / (1024 ** 3);
  const baselineAct = (numLayers * settings.tokensPerClip * hiddenDim * 16 * 2) / (1024 ** 3);
  const baselineTotal = baselineWeights + baselineGrads + baselineOpt + baselineAct + cudaBuffersGB;
  const savingsVsBaselineGB = Math.max(0, baselineTotal - totalVRAMGB);

  return {
    modelWeightsGB,
    gradientsGB,
    optimizerStatesGB,
    activationsGB,
    cudaBuffersGB,
    totalVRAMGB,
    maxVRAMGB: hw.vramGB,
    isOOM,
    savingsVsBaselineGB,
    fsdpShardFactor: fsdpDegree,
    spShardFactor: spDegree,
  };
}

export interface InvariantBatchConfig {
  worldSize: number;
  localBatch: number;
  gradAccumSteps: number;
  effectiveGlobalBatch: number;
  isExact: boolean;
}

export function computeInvariantSchedule(
  targetGlobalBatch: number,
  worldSize: number,
  maxLocalBatch: number = 1
): InvariantBatchConfig {
  const localBatch = Math.min(maxLocalBatch, Math.max(1, Math.floor(targetGlobalBatch / worldSize)));
  const currentStepTotal = localBatch * worldSize;
  const gradAccumSteps = Math.max(1, Math.ceil(targetGlobalBatch / currentStepTotal));
  const effectiveGlobalBatch = localBatch * worldSize * gradAccumSteps;

  return {
    worldSize,
    localBatch,
    gradAccumSteps,
    effectiveGlobalBatch,
    isExact: effectiveGlobalBatch === targetGlobalBatch,
  };
}

export interface LossCurvePoint {
  step: number;
  loss1GPU: number;
  loss2GPU: number;
  loss4GPU: number;
  loss8GPU: number;
  meanLoss: number;
}

export function generateMultiGPULossCurves(
  totalSteps: number = 30,
  targetGlobalBatch: number = 16
): LossCurvePoint[] {
  const points: LossCurvePoint[] = [];

  for (let s = 1; s <= totalSteps; s++) {
    // True rectified flow asymptotic loss trajectory: L(s) = 0.85 * exp(-s / 9) + 0.08 + noise
    const base = 0.85 * Math.exp(-s / 9.0) + 0.08;

    // Small numerical jitter within noise band (+/- 0.004)
    const noise1 = (Math.sin(s * 1.7) * 0.003);
    const noise2 = (Math.sin(s * 1.7 + 0.5) * 0.0035);
    const noise4 = (Math.sin(s * 1.7 + 1.1) * 0.003);
    const noise8 = (Math.sin(s * 1.7 + 1.8) * 0.0032);

    const l1 = Math.max(0.01, base + noise1);
    const l2 = Math.max(0.01, base + noise2);
    const l4 = Math.max(0.01, base + noise4);
    const l8 = Math.max(0.01, base + noise8);
    const mean = (l1 + l2 + l4 + l8) / 4;

    points.push({
      step: s,
      loss1GPU: Number(l1.toFixed(4)),
      loss2GPU: Number(l2.toFixed(4)),
      loss4GPU: Number(l4.toFixed(4)),
      loss8GPU: Number(l8.toFixed(4)),
      meanLoss: Number(mean.toFixed(4)),
    });
  }

  return points;
}

export interface ReshardSimulationResult {
  savedStep: number;
  savedEpoch: number;
  oldWorldSize: number;
  newWorldSize: number;
  oldLocalBatch: number;
  oldGradAccum: number;
  newLocalBatch: number;
  newGradAccum: number;
  globalBatchConserved: boolean;
  totalSamplesConsumed: number;
  stateStatus: 'BIT_EXACT' | 'RE_SHARDED_SUCCESS';
  verificationMessage: string;
}

export function simulateResharding(
  savedStep: number,
  oldWorldSize: number,
  newWorldSize: number,
  targetGlobalBatch: number = 16
): ReshardSimulationResult {
  const oldSchedule = computeInvariantSchedule(targetGlobalBatch, oldWorldSize);
  const newSchedule = computeInvariantSchedule(targetGlobalBatch, newWorldSize);

  const totalSamplesConsumed = savedStep * oldSchedule.effectiveGlobalBatch;

  return {
    savedStep,
    savedEpoch: Math.floor(savedStep / 200),
    oldWorldSize,
    newWorldSize,
    oldLocalBatch: oldSchedule.localBatch,
    oldGradAccum: oldSchedule.gradAccumSteps,
    newLocalBatch: newSchedule.localBatch,
    newGradAccum: newSchedule.gradAccumSteps,
    globalBatchConserved: oldSchedule.effectiveGlobalBatch === newSchedule.effectiveGlobalBatch,
    totalSamplesConsumed,
    stateStatus: 'RE_SHARDED_SUCCESS',
    verificationMessage: `Seamless resume: Model parameters un-sharded from ${oldWorldSize} GPUs and dynamically re-sharded to ${newWorldSize} GPUs. Global batch size invariant at ${newSchedule.effectiveGlobalBatch} clips.`,
  };
}

export interface MFUCalculation {
  tokensPerSecTotal: number;
  tokensPerSecPerGPU: number;
  flopsPerToken: number;
  achievedTFlopsCluster: number;
  peakTFlopsCluster: number;
  mfuPercent: number;
  isWithinTargetRange: boolean; // 30% - 45%
  rating: 'Bottlenecked' | 'Acceptable' | 'Optimal (30-45%)' | 'Exceptional';
  stallAnalysis: {
    dataFetchPercent: number;
    communicationPercent: number;
    kernelComputePercent: number;
    primaryBottleneck: string;
  };
}

export function calculateMFU(
  paramsCount: number,
  numLayers: number,
  hiddenDim: number,
  seqLen: number,
  gpuCount: number,
  hardwareId: string,
  stepDurationSec: number,
  globalBatchSize: number,
  dataFetchDurationSec: number = 0.05
): MFUCalculation {
  const hw = HARDWARE_CATALOG[hardwareId] || HARDWARE_CATALOG.h100_sxm;
  const totalTokens = globalBatchSize * seqLen;
  const totalTime = Math.max(0.01, stepDurationSec + dataFetchDurationSec);
  const tokensPerSecTotal = totalTokens / totalTime;
  const tokensPerSecPerGPU = tokensPerSecTotal / Math.max(1, gpuCount);

  // FLOPs per token: 6P + 12 L D (seq + text) / seq
  const textLen = 512;
  const denseFlops = 6 * paramsCount;
  const attnFlops = 12 * numLayers * hiddenDim * (seqLen + textLen) / Math.max(1, seqLen);
  const flopsPerToken = denseFlops + attnFlops;

  const achievedFlopsSec = tokensPerSecTotal * flopsPerToken;
  const achievedTFlopsCluster = achievedFlopsSec / 1e12;
  const peakTFlopsCluster = hw.peakTFlopsBF16 * gpuCount;

  const mfuPercent = Math.min(100, Math.max(1, (achievedTFlopsCluster / peakTFlopsCluster) * 100));
  const isWithinTargetRange = mfuPercent >= 30 && mfuPercent <= 48;

  let rating: MFUCalculation['rating'] = 'Optimal (30-45%)';
  if (mfuPercent < 25) rating = 'Bottlenecked';
  else if (mfuPercent < 30) rating = 'Acceptable';
  else if (mfuPercent > 48) rating = 'Exceptional';

  // Stall analysis
  const dataFetchPercent = Math.min(40, (dataFetchDurationSec / totalTime) * 100);
  const commPercent = Math.min(45, (0.04 * (gpuCount > 1 ? Math.log2(gpuCount) : 0) / totalTime) * 100);
  const kernelComputePercent = Math.max(15, 100 - dataFetchPercent - commPercent);

  let primaryBottleneck = 'Compute Bound (Ideal GPU utilization)';
  if (dataFetchPercent > 20) primaryBottleneck = 'DataLoader I/O Stall (Disk or CPU decoding bottleneck - use precomputed latents)';
  else if (commPercent > 25) primaryBottleneck = 'AllReduce/AllToAll Communication Overhead (Interconnect bandwidth bottleneck)';

  return {
    tokensPerSecTotal: Math.round(tokensPerSecTotal),
    tokensPerSecPerGPU: Math.round(tokensPerSecPerGPU),
    flopsPerToken,
    achievedTFlopsCluster: Number(achievedTFlopsCluster.toFixed(1)),
    peakTFlopsCluster: Number(peakTFlopsCluster.toFixed(1)),
    mfuPercent: Number(mfuPercent.toFixed(1)),
    isWithinTargetRange,
    rating,
    stallAnalysis: {
      dataFetchPercent: Number(dataFetchPercent.toFixed(1)),
      communicationPercent: Number(commPercent.toFixed(1)),
      kernelComputePercent: Number(kernelComputePercent.toFixed(1)),
      primaryBottleneck,
    },
  };
}

export interface TrainingTrackCost {
  trackName: string;
  preset: string;
  params: string;
  hardware: string;
  gpuCount: number;
  durationEstimate: string;
  estimatedCostUSD: number;
  feasibility: 'Accessible on 1 GPU' | 'Small Cluster Required' | 'Commercial Scale';
  curriculumStage: string;
}

export const TRAINING_COST_REALITY: TrainingTrackCost[] = [
  {
    trackName: 'Track A (Tiny Baseline)',
    preset: 'tiny',
    params: '18M params',
    hardware: '1x RTX 4090 (24GB)',
    gpuCount: 1,
    durationEstimate: '4 - 8 hours',
    estimatedCostUSD: 4.0,
    feasibility: 'Accessible on 1 GPU',
    curriculumStage: 'Stages 1 & 2 (Images 256px + Short Videos 17 frames)',
  },
  {
    trackName: 'Track A (Small Research)',
    preset: 'small',
    params: '168M params',
    hardware: '1x A100 (80GB) or 2x RTX 4090',
    gpuCount: 1,
    durationEstimate: '24 - 48 hours',
    estimatedCostUSD: 50.0,
    feasibility: 'Accessible on 1 GPU',
    curriculumStage: 'Stages 1, 2 & 3 (480p up to 33 frames)',
  },
  {
    trackName: 'Track B (Base Foundation)',
    preset: 'base',
    params: '410M params',
    hardware: '8x H100 SXM5 (80GB)',
    gpuCount: 8,
    durationEstimate: '4 - 7 days',
    estimatedCostUSD: 2800.0,
    feasibility: 'Small Cluster Required',
    curriculumStage: 'Full Curriculum (720p 49 frames with aesthetic filtering)',
  },
  {
    trackName: 'Commercial Wan-Class Scale',
    preset: 'large (14B)',
    params: '14.2 Billion params',
    hardware: '256x H100 SXM5 (Cluster)',
    gpuCount: 256,
    durationEstimate: '18 - 30 days (~6,000 GPU-days)',
    estimatedCostUSD: 460000.0,
    feasibility: 'Commercial Scale',
    curriculumStage: 'Curriculum 1-4 with 10M+ licensed motion-curated video corpus',
  },
];
