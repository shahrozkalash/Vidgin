/**
 * Types and interfaces for VidGen Video Latent Diffusion Studio
 */

export interface BucketSpec {
  frames: number; // Must satisfy (frames - 1) % 4 === 0
  height: number; // Multiple of 16
  width: number;  // Multiple of 16
  label: string;
  aspectRatioLabel: '16:9' | '9:16' | '1:1' | '4:3' | '3:4';
  aspectRatio: number;
  tokens: number; // After 4x temporal, 8x spatial, (1, 2, 2) patchify
}

export type PresetKey = 'tiny' | 'small' | 'vidgen480p' | 'vidgen14b';

export interface PresetConfig {
  key: PresetKey;
  name: string;
  tagline: string;
  resolution: string;
  height: number;
  width: number;
  baseFrames: number;
  tokens: number;
  ditConfig: string;
  params: string;
  targetTokensPerBatch: number;
  patchSize: [number, number, number];
}

export interface VideoClip {
  id: string;
  mediaType: 'video' | 'image';
  source: string;
  rawDuration: number;
  rawFps: number;
  rawFrames: number;
  rawHeight: number;
  rawWidth: number;
  captionLong: string;
  captionShort: string;
  captionStructured: {
    subject: string;
    action: string;
    setting: string;
    lighting: string;
    cameraMotion: string;
  };
  assignedBucket?: BucketSpec;
}

export interface BatchItem {
  id: string;
  mediaType: 'video' | 'image';
  prompt: string;
  isCaptionDropped: boolean;
  bucket: [number, number, number];
  pixels: Float32Array; // Flattened [3, T, H, W] in [-1.0, 1.0]
  frames: number;
  height: number;
  width: number;
  variance: number;
  minVal: number;
  maxVal: number;
  framesCanvasData?: ImageData[];
}

export interface SamplerState {
  epoch: number;
  batchIdx: number;
  seed: number;
  totalBatches: number;
  batches: number[][]; // Indices per batch
}

export interface LatentStats {
  numChannels: number;
  mean: number[];
  std: number[];
  totalElementsSampled: number;
}

export interface PrecomputeShard {
  id: string;
  latentShape: [number, number, number, number]; // [16, t, h, w]
  dtypeLatent: 'float16' | 'float32';
  dtypeText: 'bfloat16' | 'float32';
  variants: ('latents_orig' | 'latents_hflip' | 'latents_crop2')[];
  shardSizeKb: number;
}

export interface VAERoundTripResult {
  clipId: string;
  mediaType: 'video' | 'image';
  inputFrames: number;
  latentFrames: number; // k + 1
  latentShape: [number, number, number, number]; // [16, t, h, w]
  psnrDb: number;
  ssim: number;
  lpips: number;
  chunkedToleranceL1: number;
  matchesTolerance: boolean;
  unnormalizedFailure: boolean;
}

export interface VAEEvaluationReport {
  numClips: number;
  meanPsnrDb: number;
  meanSsim: number;
  meanLpips: number;
  meanChunkedVsUnchunkedL1: number;
  allShapesCorrect: boolean;
  toleranceSatisfied: boolean;
  clips: VAERoundTripResult[];
}

export interface PatchifyCalcResult {
  frames: number;
  height: number;
  width: number;
  patchSize: [number, number, number];
  latentT: number;
  latentH: number;
  latentW: number;
  gridT: number;
  gridH: number;
  gridW: number;
  totalTokens: number;
  isVidgen480pTarget: boolean;
}

export interface RoPEPartitionSpec {
  headDim: number;
  dt: number;
  dh: number;
  dw: number;
  theta: number;
}

export interface ExpandedPromptResult {
  original: string;
  denseCaption: string;
  style: string;
  subject: string;
  action: string;
  setting: string;
  lighting: string;
  cameraMotion: string;
  tokenCount: number;
}

export interface LongVideoSpecs {
  durationSeconds: number; // 30
  fps: number; // 16 or 24
  totalFrames: number; // 481 for 16fps, 721 for 24fps (strictly 4k + 1)
  k: number; // 120 (16fps) or 180 (24fps)
  latentFrames: number; // 121 (16fps) or 181 (24fps)
  chunkDurationSec: number; // ~5s (81 frames, 21 latents)
  framesPerChunk: number; // 81
  latentFramesPerChunk: number; // 21
  numChunks: number; // 6
  overlapLatents: number; // 1 latent frame overlap
  tokensPerChunk: number;
  totalTokensFullAttn: number;
  attentionComplexityReduction: number;
  fullAttnVramGb: number;
  chunkedKvVramGb: number;
}

export interface TimelineSegment {
  index: number;
  timeRangeSec: [number, number]; // [0, 5], [5, 10], ...
  segmentCaption: string;
  cameraMotion: string;
  action: string;
  isCaptionDropped: boolean; // 10% CFG dropout
  status: 'pending' | 'denoising' | 'cached' | 'decoded';
  identitySim: number; // 0.0 to 1.0 (DINO similarity)
  colorDrift: number; // percent shift
  borderPsnr: number; // dB continuity across chunk seam
}

export interface LongVideoRolloutState {
  currentChunk: number;
  activeWindow: number[];
  anchorFrames: number;
  contextNoiseSigma: number;
  memoryKvMb: number;
  memoryFullAttnMb: number;
  streamedChunks: number[];
  isComplete: boolean;
}

export interface LongVideoAcceptanceTest {
  id: string;
  title: string;
  category: 'CPU_NOW' | 'GPU_LATER';
  spec: string;
  status: 'passed' | 'pending' | 'running';
  metricValue: string;
  tolerance: string;
}




