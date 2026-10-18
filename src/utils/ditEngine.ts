/**
 * DiT Backbone Utility Engine for Vidgen Video Latent Diffusion
 * Simulates Vidgen-style block flow, parameter breakdowns, adaLN-Zero modulation,
 * overfit training curve on synthetic clips, and precision comparisons.
 */

import { PresetKey } from '../types/vidgen';
import { PRESETS } from './bucketEngine';

export interface DiTBlockConfig {
  numLayers: number;
  hiddenDim: number;
  numHeads: number;
  headDim: number;
  ffnDim: number;
  textDim: number;
  patchSize: [number, number, number];
  inChannels: number;
  outChannels: number;
}

export interface DiTParamBreakdown {
  patchEmbed: number;
  timestepEmbedder: number;
  textProjector: number;
  selfAttnPerBlock: number;
  crossAttnPerBlock: number;
  ffnPerBlock: number;
  blockModPerBlock: number;
  totalPerBlock: number;
  allBlocks: number;
  outputHead: number;
  totalParams: number;
  targetPresetName: string;
  nominalTargetParams: string;
  diffPercent: number;
  sharedModSavingsPercent: number;
}

export const DIT_PRESET_CONFIGS: Record<PresetKey, DiTBlockConfig> = {
  tiny: {
    numLayers: 8,
    hiddenDim: 384,
    numHeads: 6,
    headDim: 64,
    ffnDim: 1536,
    textDim: 384,
    patchSize: [1, 2, 2],
    inChannels: 16,
    outChannels: 16,
  },
  small: {
    numLayers: 16,
    hiddenDim: 1024,
    numHeads: 16,
    headDim: 64,
    ffnDim: 4096,
    textDim: 1024,
    patchSize: [1, 2, 2],
    inChannels: 16,
    outChannels: 16,
  },
  vidgen480p: {
    numLayers: 30,
    hiddenDim: 1536,
    numHeads: 12,
    headDim: 128,
    ffnDim: 8960,
    textDim: 4096,
    patchSize: [1, 2, 2],
    inChannels: 16,
    outChannels: 16,
  },
  vidgen14b: {
    numLayers: 40,
    hiddenDim: 5120,
    numHeads: 40,
    headDim: 128,
    ffnDim: 13824,
    textDim: 4096,
    patchSize: [1, 2, 2],
    inChannels: 16,
    outChannels: 16,
  },
};

/**
 * Calculate rigorous parameter count for Vidgen DiT architecture.
 */
export function calculateDiTParameters(cfg: DiTBlockConfig, presetKey: PresetKey): DiTParamBreakdown {
  const { numLayers, hiddenDim, numHeads, headDim, ffnDim, textDim, patchSize, inChannels, outChannels } = cfg;
  const pt = patchSize[0];
  const ph = patchSize[1];
  const pw = patchSize[2];

  // 1. Patch Embed: Conv3d(16, hiddenDim, (pt, ph, pw))
  const patchEmbed = inChannels * pt * ph * pw * hiddenDim + hiddenDim;

  // 2. Timestep Embedder: Sinusoidal (256) -> MLP: Linear(256, D) + Linear(D, D) + shared Linear(D, 6*D)
  const timeDim = 256;
  const timeMlp = (timeDim * hiddenDim + hiddenDim) + (hiddenDim * hiddenDim + hiddenDim);
  const sharedProj = hiddenDim * (6 * hiddenDim) + (6 * hiddenDim);
  const timestepEmbedder = timeMlp + sharedProj;

  // 3. Text Projector: Linear(textDim, D) + Linear(D, D)
  const textProjector = (textDim * hiddenDim + hiddenDim) + (hiddenDim * hiddenDim + hiddenDim);

  // 4. Per Block:
  // (a) Self Attention: q, k, v (3 * D * D), out_proj (D * D), QK-RMSNorm (2 * headDim)
  const selfAttnPerBlock = (3 * hiddenDim * hiddenDim) + (hiddenDim * hiddenDim) + (2 * headDim);

  // (b) Text Cross Attention: q, k, v (3 * D * D), out_proj (D * D), QK-RMSNorm (2 * headDim), norm_cross (2 * D)
  const crossAttnPerBlock = (3 * hiddenDim * hiddenDim) + (hiddenDim * hiddenDim) + (2 * headDim) + (2 * hiddenDim);

  // (c) Feed Forward: Linear(D, ffnDim) + Linear(ffnDim, D)
  const ffnPerBlock = (hiddenDim * ffnDim + ffnDim) + (ffnDim * hiddenDim + hiddenDim);

  // (d) Learnable modulation table offset per block (6 * D)
  const blockModPerBlock = 6 * hiddenDim;

  const totalPerBlock = selfAttnPerBlock + crossAttnPerBlock + ffnPerBlock + blockModPerBlock;
  const allBlocks = totalPerBlock * numLayers;

  // 5. Output Head: final norm (2 * D) + mod_proj (D * 2D + 2D) + Linear(D, outChannels*pt*ph*pw)
  const patchElements = outChannels * pt * ph * pw;
  const outputHead = (2 * hiddenDim) + (hiddenDim * 2 * hiddenDim + 2 * hiddenDim) + (hiddenDim * patchElements + patchElements);

  const totalParams = patchEmbed + timestepEmbedder + textProjector + allBlocks + outputHead;

  // Savings calculation: standard per-block MLP vs Vidgen shared projection
  const standardMlpMod = numLayers * (hiddenDim * 6 * hiddenDim + 6 * hiddenDim);
  const vidgenMod = sharedProj + (numLayers * blockModPerBlock);
  const sharedModSavingsPercent = Math.max(0, ((standardMlpMod - vidgenMod) / standardMlpMod) * 100);

  const nominals: Record<PresetKey, { label: string; count: number }> = {
    tiny: { label: '~20M', count: 20_000_000 },
    small: { label: '~270M', count: 270_000_000 },
    vidgen480p: { label: '~1.3B', count: 1_350_000_000 },
    vidgen14b: { label: '~14B', count: 14_000_000_000 },
  };

  const nom = nominals[presetKey];
  const diffPercent = ((totalParams - nom.count) / nom.count) * 100;

  return {
    patchEmbed,
    timestepEmbedder,
    textProjector,
    selfAttnPerBlock,
    crossAttnPerBlock,
    ffnPerBlock,
    blockModPerBlock,
    totalPerBlock,
    allBlocks,
    outputHead,
    totalParams,
    targetPresetName: PRESETS[presetKey].name,
    nominalTargetParams: nom.label,
    diffPercent,
    sharedModSavingsPercent,
  };
}

/**
 * Compute sinusoidal timestep vector values for visualization.
 */
export function computeSinusoidalEmbeddings(timestep: number, dim: number = 32): number[] {
  const scaledT = timestep * 1000.0;
  const half = dim / 2;
  const vals: number[] = [];
  const maxPeriod = 10000.0;

  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-Math.log(maxPeriod) * i) / half);
    const arg = scaledT * freq;
    vals.push(Math.cos(arg));
  }
  for (let i = 0; i < half; i++) {
    const freq = Math.exp((-Math.log(maxPeriod) * i) / half);
    const arg = scaledT * freq;
    vals.push(Math.sin(arg));
  }

  return vals;
}

/**
 * Simulate the 6 modulation curves across timesteps [0.0 to 1.0].
 */
export function computeModulationValues(timestep: number) {
  // Simulates learned MLP response for (scale1, shift1, gate1, scale2, shift2, gate2)
  const t = timestep;
  const scale1 = 0.45 * Math.sin(Math.PI * t) + 0.1 * Math.cos(2 * Math.PI * t);
  const shift1 = 0.35 * Math.cos(Math.PI * t * 0.5);
  // Gates are adaLN-zero initialized at step 0 of training, becoming subtle multipliers
  const gate1 = 0.25 * (1.0 - Math.exp(-3 * t));
  const scale2 = 0.5 * Math.sin(Math.PI * t * 1.2);
  const shift2 = -0.3 * Math.sin(Math.PI * t);
  const gate2 = 0.28 * (1.0 - Math.exp(-3 * t));

  return {
    scale1: Number(scale1.toFixed(4)),
    shift1: Number(shift1.toFixed(4)),
    gate1: Number(gate1.toFixed(4)),
    scale2: Number(scale2.toFixed(4)),
    shift2: Number(shift2.toFixed(4)),
    gate2: Number(gate2.toFixed(4)),
  };
}

/**
 * Overfit test simulator for synthetic clips.
 * Generates synthetic loss curves demonstrating monotonic drop to near zero (< 0.05).
 */
export interface OverfitStepRecord {
  step: number;
  loss: number;
  gradNorm: number;
  predError: number;
}

export function simulateOverfitRun(numClips: number = 2, totalSteps: number = 30): OverfitStepRecord[] {
  const records: OverfitStepRecord[] = [];
  let currentLoss = 0.88 + (numClips * 0.04);
  let gradNorm = 3.2;

  for (let s = 1; s <= totalSteps; s++) {
    // Exponential decay with realistic noise
    const decayFactor = 0.84 - (0.01 * (s / totalSteps));
    const noise = (Math.random() - 0.48) * 0.02 * currentLoss;
    currentLoss = Math.max(0.012, currentLoss * decayFactor + noise);

    gradNorm = Math.max(0.05, gradNorm * 0.88 + (Math.random() * 0.04));
    const predError = currentLoss * 1.15;

    records.push({
      step: s,
      loss: Number(currentLoss.toFixed(5)),
      gradNorm: Number(gradNorm.toFixed(4)),
      predError: Number(predError.toFixed(5)),
    });
  }

  return records;
}

/**
 * Compare Full 3D Spacetime Attention vs Factorized Ablation
 */
export function getAttentionComplexityComparison(gridT: number, gridH: number, gridW: number) {
  const N = gridT * gridH * gridW;
  const full3D_FLOPs = 2 * N * N; // O(N^2)
  const factorized_FLOPs = 2 * gridT * (gridH * gridW) * (gridH * gridW) + 2 * (gridH * gridW) * gridT * gridT;

  const ratio = (full3D_FLOPs / factorized_FLOPs).toFixed(1);

  return {
    N,
    full3D_FLOPs,
    factorized_FLOPs,
    ratio: Number(ratio),
    full3D_formula: `O(N²) = O((${gridT} × ${gridH} × ${gridW})²) = ${full3D_FLOPs.toLocaleString()} ops/head`,
    factorized_formula: `O(T × (HW)² + HW × T²) = ${factorized_FLOPs.toLocaleString()} ops/head`,
  };
}
