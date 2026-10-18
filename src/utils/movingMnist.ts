/**
 * Moving MNIST & Synthetic Spacetime Video Generator
 * Produces real pixel frames [T, H, W, 3] with controllable motion,
 * aspect ratios, and verifies strictly [-1.0, 1.0] normalized ranges.
 */

export interface RenderedClipData {
  frames: ImageData[];
  pixels: Float32Array; // [3, T, H, W] flattened in [-1.0, 1.0]
  variance: number;
  minVal: number;
  maxVal: number;
  durationMs: number;
}

// 7-segment-like or bitmap stroke digit patterns (0-9)
const DIGIT_VECTORS: Record<number, number[][]> = {
  0: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]],
  1: [[0.5, 0.2], [0.5, 0.8]],
  2: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.5], [0.2, 0.5], [0.2, 0.8], [0.8, 0.8]],
  3: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.5], [0.4, 0.5], [0.8, 0.5], [0.8, 0.8], [0.2, 0.8]],
  4: [[0.2, 0.2], [0.2, 0.5], [0.8, 0.5], [0.8, 0.2], [0.8, 0.8]],
  5: [[0.8, 0.2], [0.2, 0.2], [0.2, 0.5], [0.8, 0.5], [0.8, 0.8], [0.2, 0.8]],
  6: [[0.8, 0.2], [0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.5], [0.2, 0.5]],
  7: [[0.2, 0.2], [0.8, 0.2], [0.4, 0.8]],
  8: [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2], [0.2, 0.5], [0.8, 0.5]],
  9: [[0.8, 0.5], [0.2, 0.5], [0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
};

export function renderMovingMNISTClip(
  framesCount: number,
  height: number,
  width: number,
  seed: number = 42,
  digit1: number = 3,
  digit2: number = 7
): RenderedClipData {
  const startTime = performance.now();
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;

  const frames: ImageData[] = [];
  // Pixel array in [3, T, H, W] flattened:
  const totalElements = 3 * framesCount * height * width;
  const pixels = new Float32Array(totalElements);

  // Trajectory parameters
  let x1 = 0.25 * width;
  let y1 = 0.25 * height;
  let vx1 = ((seed % 7) - 3) * 1.4 || 1.6;
  let vy1 = (((seed >> 2) % 7) - 3) * 1.2 || 1.3;

  let x2 = 0.70 * width;
  let y2 = 0.65 * height;
  let vx2 = (((seed >> 4) % 7) - 3) * 1.5 || -1.4;
  let vy2 = (((seed >> 6) % 7) - 3) * 1.1 || -1.2;

  const digitSize = Math.max(12, Math.min(width, height) * 0.28);

  function drawDigit(digit: number, cx: number, cy: number, colorR: number, colorG: number, colorB: number) {
    const points = DIGIT_VECTORS[digit] || DIGIT_VECTORS[0];
    ctx.strokeStyle = `rgb(${colorR}, ${colorG}, ${colorB})`;
    ctx.lineWidth = Math.max(2, digitSize * 0.16);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    points.forEach(([px, py], i) => {
      const sx = cx + (px - 0.5) * digitSize;
      const sy = cy + (py - 0.5) * digitSize;
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.stroke();
  }

  let sum = 0;
  let sqSum = 0;
  let minVal = 1.0;
  let maxVal = -1.0;

  for (let t = 0; t < framesCount; t++) {
    // Background gradient or dark canvas
    const grad = ctx.createLinearGradient(0, 0, width, height);
    grad.addColorStop(0, '#090d16');
    grad.addColorStop(1, '#111827');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, width, height);

    // Subtle grid pattern for visual spatial reference
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let gx = 0; gx < width; gx += 16) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();
    }
    for (let gy = 0; gy < height; gy += 16) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }

    // Update positions and bounce against borders
    x1 += vx1;
    y1 += vy1;
    if (x1 < digitSize / 2 || x1 > width - digitSize / 2) {
      vx1 = -vx1;
      x1 = Math.max(digitSize / 2, Math.min(width - digitSize / 2, x1));
    }
    if (y1 < digitSize / 2 || y1 > height - digitSize / 2) {
      vy1 = -vy1;
      y1 = Math.max(digitSize / 2, Math.min(height - digitSize / 2, y1));
    }

    x2 += vx2;
    y2 += vy2;
    if (x2 < digitSize / 2 || x2 > width - digitSize / 2) {
      vx2 = -vx2;
      x2 = Math.max(digitSize / 2, Math.min(width - digitSize / 2, x2));
    }
    if (y2 < digitSize / 2 || y2 > height - digitSize / 2) {
      vy2 = -vy2;
      y2 = Math.max(digitSize / 2, Math.min(height - digitSize / 2, y2));
    }

    // Draw digits with distinct glowing hues (Cyan & Amber)
    drawDigit(digit1, x1, y1, 56, 189, 248); // Cyan-400
    drawDigit(digit2, x2, y2, 251, 146, 60);  // Orange-400

    // Capture ImageData
    const imgData = ctx.getImageData(0, 0, width, height);
    frames.push(imgData);

    // Convert uint8 RGB [0, 255] to float [-1.0, 1.0] and pack into [3, T, H, W]
    const rawBytes = imgData.data;
    const spatialSize = height * width;

    for (let h = 0; h < height; h++) {
      for (let w = 0; w < width; w++) {
        const byteIdx = (h * width + w) * 4;
        const rNorm = (rawBytes[byteIdx] / 255.0) * 2.0 - 1.0;
        const gNorm = (rawBytes[byteIdx + 1] / 255.0) * 2.0 - 1.0;
        const bNorm = (rawBytes[byteIdx + 2] / 255.0) * 2.0 - 1.0;

        // [3, T, H, W] indexing:
        // C=0 (R)
        const idxR = 0 * (framesCount * spatialSize) + t * spatialSize + (h * width + w);
        // C=1 (G)
        const idxG = 1 * (framesCount * spatialSize) + t * spatialSize + (h * width + w);
        // C=2 (B)
        const idxB = 2 * (framesCount * spatialSize) + t * spatialSize + (h * width + w);

        pixels[idxR] = rNorm;
        pixels[idxG] = gNorm;
        pixels[idxB] = bNorm;

        // Variance & min/max stats
        for (const val of [rNorm, gNorm, bNorm]) {
          sum += val;
          sqSum += val * val;
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
      }
    }
  }

  const n = totalElements;
  const mean = sum / n;
  const variance = sqSum / n - mean * mean;
  const durationMs = performance.now() - startTime;

  return {
    frames,
    pixels,
    variance,
    minVal,
    maxVal,
    durationMs,
  };
}
