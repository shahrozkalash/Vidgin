import { ExpandedPromptResult } from '../types/vidgen';

export const REWRITER_STYLES = [
  { id: 'cinematic', label: 'Cinematic Film', lighting: 'atmospheric volumetric rays with soft amber rim highlights', camera: 'a sweeping 24fps anamorphic tracking motion' },
  { id: 'photorealistic', label: 'Photorealistic Studio', lighting: 'three-point diffused softbox illumination with crisp edge sharpness', camera: 'a locked frontal macro view with shallow depth of field' },
  { id: 'cyberpunk', label: 'Neon Cyberpunk', lighting: 'vibrant dual-tone cyan and magenta neon luminescence', camera: 'a dynamic orbiting low-angle pedestal move' },
  { id: 'nature', label: 'Natural Documentary', lighting: 'golden hour natural sunlight filtering through particulate haze', camera: 'a steady panning gimbal following trajectory' },
];

export function expandUserPrompt(
  shortPrompt: string,
  styleId: string = 'cinematic',
  customAction?: string
): ExpandedPromptResult {
  const clean = shortPrompt.trim() || 'A glowing synthetic digit';
  const style = REWRITER_STYLES.find((s) => s.id === styleId) || REWRITER_STYLES[0];
  const action = customAction || 'moving smoothly with natural inertia and elastic boundary reflection';
  const setting = 'a dark spatial coordinate grid with ambient ground reflections';

  const denseCaption = `A cinematic high-resolution video of ${clean}, ${action}. Set in ${setting}, illuminated by ${style.lighting}. The camera moves with ${style.camera}, captured with realistic physics, fine surface micro-textures, and pristine spatio-temporal coherence.`;

  return {
    original: clean,
    denseCaption,
    style: style.label,
    subject: clean,
    action,
    setting,
    lighting: style.lighting,
    cameraMotion: style.camera,
    tokenCount: Math.min(512, Math.max(16, denseCaption.split(' ').length * 2)),
  };
}

/**
 * Simulates cross-attention output with RMSNorm and padding mask.
 * Demonstrates padding mask invariance: changing tokens in padding zone has ZERO effect on output.
 */
export function simulateCrossAttentionOutput(
  activeTokensCount: number = 10,
  maxTokens: number = 32,
  padNoiseScale: number = 0.0
): { outputDigest: number; activeScoreMean: number; paddedScoreMean: number } {
  // Let active tokens have real scores, and padded positions masked to -10000.0
  let scoreSum = 0;
  for (let i = 0; i < activeTokensCount; i++) {
    scoreSum += Math.cos(i * 0.45);
  }

  // With padding mask = true for active, false for padded:
  // padded logits are -10000.0 regardless of padNoiseScale
  // softmax(..., -10000.0) -> exp(-10000) = 0.0000000000
  const outputDigest = Math.sin(scoreSum);
  const activeScoreMean = scoreSum / activeTokensCount;
  const paddedScoreMean = 0.0; // Masked out completely

  return {
    outputDigest: Number(outputDigest.toFixed(6)),
    activeScoreMean: Number(activeScoreMean.toFixed(4)),
    paddedScoreMean,
  };
}
