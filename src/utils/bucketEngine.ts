import { BucketSpec, PresetKey, PresetConfig, VideoClip, SamplerState, LatentStats } from '../types/vidgen';

export const PRESETS: Record<PresetKey, PresetConfig> = {
  tiny: {
    key: 'tiny',
    name: 'Tiny',
    tagline: 'Track A Prototyping & Verification',
    resolution: '64x64',
    baseFrames: 17, // k = 4 -> 4*4 + 1
    tokens: 80,
    ditConfig: '8L / 384D / 6H / 1536 FFN',
    params: '~20M',
    targetTokensPerBatch: 160,
    patchSize: [1, 2, 2],
  },
  small: {
    key: 'small',
    name: 'Small',
    tagline: 'Medium Scale Verification',
    resolution: '256x256',
    baseFrames: 33, // k = 8 -> 4*8 + 1
    tokens: 2304,
    ditConfig: '16L / 1024D / 16H / 4096 FFN',
    params: '~270M',
    targetTokensPerBatch: 4608,
    patchSize: [1, 2, 2],
  },
  wan480p: {
    key: 'wan480p',
    name: 'Wan 480p',
    tagline: 'Standard Production DiT',
    resolution: '480x832',
    baseFrames: 81, // k = 20 -> 4*20 + 1
    tokens: 32760,
    ditConfig: '30L / 1536D / 12H / 8960 FFN',
    params: '~1.3B',
    targetTokensPerBatch: 65520,
    patchSize: [1, 2, 2],
  },
  wan14b: {
    key: 'wan14b',
    name: 'Wan 14B',
    tagline: 'Flagship Cinematic DiT',
    resolution: '720x1280',
    baseFrames: 81,
    tokens: 75600,
    ditConfig: '40L / 5120D / 40H / 13824 FFN',
    params: '~14B',
    targetTokensPerBatch: 151200,
    patchSize: [1, 2, 2],
  },
};

/**
 * Calculates token count after 4x temporal and 8x spatial downsampling,
 * followed by (pt, ph, pw) patchify.
 */
export function calculateTokens(
  frames: number,
  height: number,
  width: number,
  patchSize: [number, number, number] = [1, 2, 2]
): number {
  const t = Math.floor((frames - 1) / 4) + 1;
  const h = Math.floor(height / 8);
  const w = Math.floor(width / 8);
  const [pt, ph, pw] = patchSize;
  return Math.floor(t / pt) * Math.floor(h / ph) * Math.floor(w / pw);
}

/**
 * Generate standard aspect ratio buckets (16:9, 9:16, 1:1, 4:3, 3:4) for both Video (T=4k+1) and Images (T=1).
 * All dimensions are multiples of 16.
 */
export function getStandardBuckets(presetKey: PresetKey = 'tiny'): BucketSpec[] {
  type ResTuple = [number, number, '16:9' | '9:16' | '1:1' | '4:3' | '3:4'];

  let resolutions: ResTuple[];
  let videoFrames: number[];

  if (presetKey === 'tiny') {
    resolutions = [
      [64, 64, '1:1'],
      [48, 80, '16:9'],
      [80, 48, '9:16'],
      [48, 64, '4:3'],
      [64, 48, '3:4'],
    ];
    videoFrames = [17];
  } else if (presetKey === 'small') {
    resolutions = [
      [256, 256, '1:1'],
      [192, 336, '16:9'],
      [336, 192, '9:16'],
      [224, 288, '4:3'],
      [288, 224, '3:4'],
    ];
    videoFrames = [33];
  } else if (presetKey === 'wan480p') {
    resolutions = [
      [480, 832, '16:9'],
      [832, 480, '9:16'],
      [640, 640, '1:1'],
      [560, 736, '4:3'],
      [736, 560, '3:4'],
    ];
    videoFrames = [81];
  } else {
    // wan14b
    resolutions = [
      [720, 1280, '16:9'],
      [1280, 720, '9:16'],
      [960, 960, '1:1'],
      [832, 1104, '4:3'],
      [1104, 832, '3:4'],
    ];
    videoFrames = [81];
  }

  const buckets: BucketSpec[] = [];

  // Video buckets
  for (const f of videoFrames) {
    for (const [h, w, arLabel] of resolutions) {
      buckets.push({
        frames: f,
        height: h,
        width: w,
        label: `Video ${f}f ${w}x${h} (${arLabel})`,
        aspectRatioLabel: arLabel,
        aspectRatio: w / h,
        tokens: calculateTokens(f, h, w, PRESETS[presetKey].patchSize),
      });
    }
  }

  // Image buckets (T=1, satisfying 1 = 4(0) + 1)
  for (const [h, w, arLabel] of resolutions) {
    buckets.push({
      frames: 1,
      height: h,
      width: w,
      label: `Image 1f ${w}x${h} (${arLabel})`,
      aspectRatioLabel: arLabel,
      aspectRatio: w / h,
      tokens: calculateTokens(1, h, w, PRESETS[presetKey].patchSize),
    });
  }

  return buckets;
}

/**
 * Assigns clip to closest bucket by temporal distance and aspect ratio.
 */
export function findNearestBucket(
  frames: number,
  height: number,
  width: number,
  buckets: BucketSpec[],
  isImage: boolean = false
): BucketSpec {
  const targetAspect = width / Math.max(1, height);
  const candidates = isImage || frames === 1
    ? buckets.filter((b) => b.frames === 1)
    : buckets.filter((b) => b.frames > 1);

  const pool = candidates.length > 0 ? candidates : buckets;
  let bestBucket = pool[0];
  let bestDist = Infinity;

  for (const b of pool) {
    const aspectDiff = Math.abs(b.aspectRatio - targetAspect);
    const frameDiff = Math.abs(b.frames - frames) / Math.max(1, b.frames);
    const dist = aspectDiff + 0.5 * frameDiff;

    if (dist < bestDist) {
      bestDist = dist;
      bestBucket = b;
    }
  }

  return bestBucket;
}

/**
 * Computes dynamic batch size per bucket to maintain a roughly constant token count per batch.
 */
export function computeBucketBatchSize(
  bucket: BucketSpec,
  targetTokensPerBatch: number,
  minBatchSize: number = 1,
  maxBatchSize: number = 16
): number {
  const tokens = Math.max(1, bucket.tokens);
  const scaled = Math.round(targetTokensPerBatch / tokens);
  return Math.max(minBatchSize, Math.min(maxBatchSize, scaled));
}

/**
 * Deterministic PRNG seeded random numbers (Mulberry32)
 */
export function createRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fisher-Yates shuffle with custom RNG
 */
export function shuffleArray<T>(arr: T[], rng: () => number): T[] {
  const copy = [...arr];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

/**
 * Simulate DistributedBucketSampler generation
 */
export function generateSamplerBatches(
  assignments: BucketSpec[],
  targetTokens: number,
  epoch: number,
  seed: number,
  rank: number = 0,
  worldSize: number = 1,
  minBatchSize: number = 1,
  maxBatchSize: number = 16
): { batches: number[][]; bucketBatchSizes: Map<string, number> } {
  const rng = createRng(seed + epoch * 1000 + 7);
  const bucketToIndices = new Map<string, number[]>();
  const bucketMap = new Map<string, BucketSpec>();

  assignments.forEach((bucket, idx) => {
    const key = `${bucket.frames}_${bucket.height}_${bucket.width}`;
    if (!bucketToIndices.has(key)) {
      bucketToIndices.set(key, []);
      bucketMap.set(key, bucket);
    }
    bucketToIndices.get(key)!.push(idx);
  });

  const bucketBatchSizes = new Map<string, number>();
  const allBatches: number[][] = [];

  for (const [key, indices] of bucketToIndices.entries()) {
    const bucket = bucketMap.get(key)!;
    const bs = computeBucketBatchSize(bucket, targetTokens, minBatchSize, maxBatchSize);
    bucketBatchSizes.set(key, bs);

    const shuffledIndices = shuffleArray(indices, rng);
    for (let i = 0; i < shuffledIndices.length; i += bs) {
      allBatches.push(shuffledIndices.slice(i, i + bs));
    }
  }

  const shuffledBatches = shuffleArray(allBatches, rng);

  // Distributed rank sharding
  if (worldSize > 1) {
    const remainder = shuffledBatches.length % worldSize;
    const trimmed = remainder !== 0 ? shuffledBatches.slice(0, shuffledBatches.length - remainder) : shuffledBatches;
    const rankBatches = trimmed.filter((_, idx) => idx % worldSize === rank);
    return { batches: rankBatches, bucketBatchSizes };
  }

  return { batches: shuffledBatches, bucketBatchSizes };
}

/**
 * Mock 16-channel running statistics
 */
export function calculateChannelStats(channels: number = 16): LatentStats {
  // Typical VAE latent statistics before normalization:
  // mean ranges around -0.1 to +0.1, std around 0.85 to 1.35
  const mean: number[] = [];
  const std: number[] = [];
  const rng = createRng(42);

  for (let c = 0; c < channels; c++) {
    const m = (rng() - 0.5) * 0.28;
    const s = 0.82 + rng() * 0.45;
    mean.push(Number(m.toFixed(4)));
    std.push(Number(s.toFixed(4)));
  }

  return {
    numChannels: channels,
    mean,
    std,
    totalElementsSampled: 20480,
  };
}
