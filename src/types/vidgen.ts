/**
 * Types and interfaces for VidGen Wan Video Latent Diffusion Studio
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

export type PresetKey = 'tiny' | 'small' | 'wan480p' | 'wan14b';

export interface PresetConfig {
  key: PresetKey;
  name: string;
  tagline: string;
  resolution: string;
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
  textEmbDim: number;
  variants: ('latents_orig' | 'latents_hflip' | 'latents_crop2')[];
  shardSizeKb: number;
}
