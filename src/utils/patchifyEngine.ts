import { PatchifyCalcResult, RoPEPartitionSpec, PresetKey } from '../types/vidgen';

/**
 * Calculates spacetime tokens given raw video dimensions and patch size.
 * Vidgen 4x temporal VAE: t = (T - 1) // 4 + 1
 * Vidgen 8x spatial VAE:  h = H // 8, w = W // 8
 * Patchify: grid_t = t // pt, grid_h = h // ph, grid_w = w // pw
 * Tokens N = grid_t * grid_h * grid_w
 */
export function calculatePatchTokens(
  frames: number,
  height: number,
  width: number,
  patchSize: [number, number, number] = [1, 2, 2]
): PatchifyCalcResult {
  const [pt, ph, pw] = patchSize;
  const latentT = Math.floor((frames - 1) / 4) + 1;
  const latentH = Math.floor(height / 8);
  const latentW = Math.floor(width / 8);

  const gridT = Math.floor(latentT / pt);
  const gridH = Math.floor(latentH / ph);
  const gridW = Math.floor(latentW / pw);

  const totalTokens = gridT * gridH * gridW;
  const isTarget = frames === 81 && height === 480 && width === 832 && totalTokens === 32760;

  return {
    frames,
    height,
    width,
    patchSize,
    latentT,
    latentH,
    latentW,
    gridT,
    gridH,
    gridW,
    totalTokens,
    isVidgen480pTarget: isTarget,
  };
}

/**
 * Partition head_dim into (dt, dh, dw) with all even components.
 * Vidgen standard: head_dim = 128 -> (44, 42, 42)
 * Tiny preset:  head_dim = 64  -> (22, 22, 20)
 */
export function getRoPEPartition(headDim: number = 128): RoPEPartitionSpec {
  let dt: number;
  let dh: number;
  let dw: number;

  if (headDim === 128) {
    dt = 44;
    dh = 42;
    dw = 42;
  } else if (headDim === 64) {
    dt = 22;
    dh = 22;
    dw = 20;
  } else {
    const base = Math.floor(Math.floor(headDim / 3) / 2) * 2;
    const rem = headDim - 2 * base;
    dt = rem;
    dh = base;
    dw = base;
  }

  return {
    headDim,
    dt,
    dh,
    dw,
    theta: 10000.0,
  };
}

/**
 * Simulates 3D RoPE dot-product attention score between two 3D token coordinates.
 * Proves relative-position invariance:
 * score(pos1, pos2) === score(pos1 + delta, pos2 + delta)
 */
export function computeRoPEAttentionScore(
  pos1: [number, number, number],
  pos2: [number, number, number],
  headDim: number = 128,
  theta: number = 10000.0
): { score: number; deltaT: number; deltaH: number; deltaW: number } {
  const { dt, dh, dw } = getRoPEPartition(headDim);

  const deltaT = pos1[0] - pos2[0];
  const deltaH = pos1[1] - pos2[1];
  const deltaW = pos1[2] - pos2[2];

  // In Rotary Embedding, the inner product between query at pos1 and key at pos2
  // simplifies to the sum over all rotary frequencies of cos((pos1 - pos2) * omega_i).
  let sumCos = 0;

  // Time component (dt // 2 frequencies)
  const halfDt = dt / 2;
  for (let i = 0; i < halfDt; i++) {
    const omega = 1.0 / Math.pow(theta, i / halfDt);
    sumCos += Math.cos(deltaT * omega);
  }

  // Height component (dh // 2 frequencies)
  const halfDh = dh / 2;
  for (let i = 0; i < halfDh; i++) {
    const omega = 1.0 / Math.pow(theta, i / halfDh);
    sumCos += Math.cos(deltaH * omega);
  }

  // Width component (dw // 2 frequencies)
  const halfDw = dw / 2;
  for (let i = 0; i < halfDw; i++) {
    const omega = 1.0 / Math.pow(theta, i / halfDw);
    sumCos += Math.cos(deltaW * omega);
  }

  const score = sumCos / (headDim / 2);
  return {
    score: Number(score.toFixed(6)),
    deltaT,
    deltaH,
    deltaW,
  };
}
