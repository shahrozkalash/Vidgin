import { VideoClip, BucketSpec, PresetKey } from '../types/vidgen';
import { getStandardBuckets, findNearestBucket } from './bucketEngine';

export const SAMPLE_CAPTIONS = [
  {
    long: 'A glowing cyan neon digit three floats smoothly across an indigo spatial coordinate grid while bouncing off reflective bounding walls with soft ambient shadows.',
    short: 'Digit 3 floating on spatial grid.',
    structured: {
      subject: 'Glowing neon digit 3',
      action: 'Translating diagonally and bouncing',
      setting: 'Coordinate grid dark space',
      lighting: 'Cyan luminescence with ambient glow',
      cameraMotion: 'Static frontal orthogonal viewport',
    },
  },
  {
    long: 'Cinematic trajectory of an energetic orange digit seven crossing paths with a teal numeral, exhibiting linear momentum and elastic boundary reflection.',
    short: 'Orange 7 bouncing in frame.',
    structured: {
      subject: 'Orange numeral 7',
      action: 'Elastic collision and diagonal drift',
      setting: 'Deep charcoal background',
      lighting: 'Warm specular highlight',
      cameraMotion: 'Fixed perspective',
    },
  },
  {
    long: 'High-speed synthetic moving MNIST pair demonstrating spatio-temporal coherence with continuous time trajectory and zero inter-frame jitter.',
    short: 'Two moving digits interacting.',
    structured: {
      subject: 'Interacting digits 4 and 8',
      action: 'Continuous linear trajectory',
      setting: 'Synthetic latent test canvas',
      lighting: 'Dual-tone chromatic lighting',
      cameraMotion: 'Locked frame',
    },
  },
  {
    long: 'Static high-resolution generative image tile showcasing pristine geometric boundary sharpness and uniform channel noise distribution.',
    short: 'Still texture canvas.',
    structured: {
      subject: 'Static texture and numerals',
      action: 'None (T=1 static frame)',
      setting: 'Spatial laboratory test bench',
      lighting: 'Flat neutral illumination',
      cameraMotion: 'None',
    },
  },
  {
    long: 'Vertical video clip of animated glowing vector glyphs drifting with smooth sinusoidal velocity modulation across dark spacetime.',
    short: 'Vertical drifting vectors.',
    structured: {
      subject: 'Vector numerals 2 and 9',
      action: 'Sinusoidal vertical drift',
      setting: 'Portrait 9:16 aspect arena',
      lighting: 'Vibrant neon edges',
      cameraMotion: 'Fixed vertical aspect',
    },
  },
];

export function generateSampleClips(count: number = 40, preset: PresetKey = 'tiny', imageRatio: number = 0.2): VideoClip[] {
  const buckets = getStandardBuckets(preset);
  const clips: VideoClip[] = [];

  const baseW = preset === 'tiny' ? 64 : preset === 'small' ? 256 : 480;
  const baseH = preset === 'tiny' ? 64 : preset === 'small' ? 256 : 480;
  const baseFrames = preset === 'tiny' ? 17 : preset === 'small' ? 33 : 81;

  for (let i = 0; i < count; i++) {
    const isImage = Math.random() < imageRatio;
    const captionObj = SAMPLE_CAPTIONS[i % SAMPLE_CAPTIONS.length];

    // Vary raw dimensions around aspect ratios
    let rawH = baseH;
    let rawW = baseW;
    const arType = i % 5;
    if (arType === 0) {
      // 16:9 approx
      rawH = Math.round((baseH * 0.75) / 16) * 16;
      rawW = Math.round((baseW * 1.25) / 16) * 16;
    } else if (arType === 1) {
      // 9:16 approx
      rawH = Math.round((baseH * 1.25) / 16) * 16;
      rawW = Math.round((baseW * 0.75) / 16) * 16;
    } else if (arType === 2) {
      // 1:1
      rawH = baseH;
      rawW = baseW;
    } else if (arType === 3) {
      // 4:3
      rawH = Math.round((baseH * 0.85) / 16) * 16;
      rawW = Math.round((baseW * 1.1) / 16) * 16;
    } else {
      // 3:4
      rawH = Math.round((baseH * 1.1) / 16) * 16;
      rawW = Math.round((baseW * 0.85) / 16) * 16;
    }

    const rawFrames = isImage ? 1 : Math.round(baseFrames * (0.8 + 0.4 * Math.random()));
    const rawFps = isImage ? 0 : [15, 24, 30, 60][i % 4];
    const duration = isImage ? 0 : rawFrames / rawFps;

    const assignedBucket = findNearestBucket(rawFrames, rawH, rawW, buckets, isImage);

    clips.push({
      id: `clip_${String(i + 1).padStart(4, '0')}`,
      mediaType: isImage ? 'image' : 'video',
      source: isImage ? 'Synthetic Image / Pexels' : 'Moving MNIST / OpenVid-1M',
      rawDuration: Number(duration.toFixed(2)),
      rawFps,
      rawFrames,
      rawHeight: rawH,
      rawWidth: rawW,
      captionLong: captionObj.long,
      captionShort: captionObj.short,
      captionStructured: captionObj.structured,
      assignedBucket,
    });
  }

  return clips;
}
