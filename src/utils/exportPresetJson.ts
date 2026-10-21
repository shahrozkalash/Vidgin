/**
 * Utility to export a preset's complete architecture configuration
 * and performance metrics as a downloadable JSON file or formatted object.
 */

import { PresetKey } from '../types/vidgen';
import { PRESETS } from './bucketEngine';
import { calculateDiTParameters, DIT_PRESET_CONFIGS } from './ditEngine';
import { calculateLongVideoSpecs, estimateAttentionAnalytical } from './longVideoEngine';

export interface PresetArchitectureMetricsExport {
  schemaVersion: string;
  exportedAt: string;
  generator: string;
  preset: {
    key: PresetKey;
    name: string;
    paramsLabel: string;
    targetHardware: string;
    resolution: {
      width: number;
      height: number;
      formattedResolution: string;
      aspectRatio: string;
      totalPixels: number;
    };
    latentSpace: {
      latentChannels: number;
      spatialDownsampleFactor: number;
      temporalDownsampleFactor: number;
      spatialLatentShape: [number, number];
    };
    runtimeDefaults: {
      durationSeconds: number;
      fps: number;
      totalFrames: number;
      latentFrames: number;
    };
  };
  architecture: {
    modelFamily: string;
    patchSize: [number, number, number];
    patchDescription: string;
    numLayers: number;
    hiddenDimension: number;
    numAttentionHeads: number;
    headDimension: number;
    feedForwardDimension: number;
    crossAttentionTextDimension: number;
    inputChannels: number;
    outputChannels: number;
    modulation: {
      type: string;
      sharedModulationProjection: boolean;
      estimatedParameterSavingsPercent: number;
    };
    parameterBreakdown: {
      totalParameters: number;
      totalParametersFormatted: string;
      patchEmbedding: number;
      timestepEmbedder: number;
      textProjector: number;
      selfAttentionPerBlock: number;
      crossAttentionPerBlock: number;
      feedForwardPerBlock: number;
      blockModulationOffsetPerBlock: number;
      totalPerBlock: number;
      allTransformerBlocks: number;
      outputHead: number;
    };
  };
  performanceMetrics: {
    benchmarkSetup: {
      benchmarkDurationSeconds: number;
      frameRateFps: number;
      totalRawFrames: number;
      totalLatentFrames: number;
      spatialTokensPerFrame: number;
      totalVisualTokens: number;
      conditioningTextTokens: number;
      totalSequenceLength: number;
    };
    memoryFootprint: {
      units: string;
      tensorPrecision: string;
      fullAttentionMatrix: {
        megabytes: number;
        gigabytes: number;
        exceedsColabT4Vram16GB: boolean;
        colabT4Status: string;
        explanation: string;
      };
      decoupledAttentionMatrix: {
        megabytes: number;
        gigabytes: number;
      };
      chunkedStreamingAttentionMatrix: {
        megabytes: number;
        gigabytes: number;
        numChunks: number;
        chunkSequenceLength: number;
      };
      kvCacheVramEstimate: {
        fullAttentionVramGB: number;
        chunkedStreamingKvVramGB: number;
        vramSavingsGB: number;
      };
    };
    computeComplexity: {
      units: string;
      analyticalFlopsFormula: string;
      fullAttentionTFlops: number;
      decoupledSpatialTemporalTFlops: number;
      chunkedAttentionTFlops: number;
      complexityReductionFactor: string;
      flopsSavingsPercentVsFullAttention: string;
    };
  };
}

/**
 * Builds the complete structured export object for a given preset key and duration.
 */
export function generatePresetExportData(
  presetKey: PresetKey,
  durationSec = 30,
  fps = 16
): PresetArchitectureMetricsExport {
  const presetCfg = PRESETS[presetKey] || PRESETS.tiny;
  const ditCfg = DIT_PRESET_CONFIGS[presetKey] || DIT_PRESET_CONFIGS.tiny;
  const ditParams = calculateDiTParameters(ditCfg, presetKey);
  const analytical = estimateAttentionAnalytical(durationSec, fps, presetKey);
  const longVideoSpecs = calculateLongVideoSpecs(durationSec, fps, presetKey);

  const hwTargets: Record<PresetKey, string> = {
    tiny: 'Consumer CPU / Edge Device (Apple Silicon, Laptop iGPU)',
    small: 'Single Consumer GPU (RTX 3060 / 4060 8GB-12GB)',
    vidgen480p: 'Prosumer GPU / Free Colab T4 (RTX 3090 / 4090 / T4 16GB)',
    vidgen9b: 'High-End Datacenter GPU (A100 80GB, H100 SXM, Dual A100)',
    vidgen14b: 'High-End Datacenter GPU (A100 80GB, H100 SXM, Multi-GPU Cluster)',
  };

  const latentW = Math.round(presetCfg.width / 8);
  const latentH = Math.round(presetCfg.height / 8);

  const vramSavings = Math.max(0, longVideoSpecs.fullAttnVramGb - longVideoSpecs.chunkedKvVramGb);
  const flopsSavings = (
    (1 - analytical.chunkedAttnFlopsTFlops / Math.max(0.001, analytical.fullAttnFlopsTFlops)) *
    100
  ).toFixed(1);

  return {
    schemaVersion: '1.0.0',
    exportedAt: new Date().toISOString(),
    generator: 'Open-VidGen DiT Architecture & Inference Suite',
    preset: {
      key: presetKey,
      name: presetCfg.name,
      paramsLabel: presetCfg.params,
      targetHardware: hwTargets[presetKey],
      resolution: {
        width: presetCfg.width,
        height: presetCfg.height,
        formattedResolution: presetCfg.resolution,
        aspectRatio:
          presetCfg.width === presetCfg.height
            ? '1:1'
            : presetCfg.width > presetCfg.height
            ? `${(presetCfg.width / presetCfg.height).toFixed(2)}:1 (Landscape)`
            : `1:${(presetCfg.height / presetCfg.width).toFixed(2)} (Portrait)`,
        totalPixels: presetCfg.width * presetCfg.height,
      },
      latentSpace: {
        latentChannels: 16,
        spatialDownsampleFactor: 8,
        temporalDownsampleFactor: 4,
        spatialLatentShape: [latentH, latentW],
      },
      runtimeDefaults: {
        durationSeconds: durationSec,
        fps,
        totalFrames: durationSec * fps,
        latentFrames: Math.floor((durationSec * fps - 1) / 4) + 1,
      },
    },
    architecture: {
      modelFamily: 'Diffusion Transformer (DiT) with 3D Spacetime Patchify',
      patchSize: ditCfg.patchSize,
      patchDescription: `${ditCfg.patchSize[0]} temporal x ${ditCfg.patchSize[1]} vertical x ${ditCfg.patchSize[2]} horizontal`,
      numLayers: ditCfg.numLayers,
      hiddenDimension: ditCfg.hiddenDim,
      numAttentionHeads: ditCfg.numHeads,
      headDimension: ditCfg.headDim,
      feedForwardDimension: ditCfg.ffnDim,
      crossAttentionTextDimension: ditCfg.textDim,
      inputChannels: ditCfg.inChannels,
      outputChannels: ditCfg.outChannels,
      modulation: {
        type: 'adaLN-Zero with Single Shared Linear Projection',
        sharedModulationProjection: true,
        estimatedParameterSavingsPercent: Math.round(ditParams.sharedModSavingsPercent),
      },
      parameterBreakdown: {
        totalParameters: ditParams.totalParams,
        totalParametersFormatted: `${(ditParams.totalParams / 1_000_000).toFixed(2)}M`,
        patchEmbedding: ditParams.patchEmbed,
        timestepEmbedder: ditParams.timestepEmbedder,
        textProjector: ditParams.textProjector,
        selfAttentionPerBlock: ditParams.selfAttnPerBlock,
        crossAttentionPerBlock: ditParams.crossAttnPerBlock,
        feedForwardPerBlock: ditParams.ffnPerBlock,
        blockModulationOffsetPerBlock: ditParams.blockModPerBlock,
        totalPerBlock: ditParams.totalPerBlock,
        allTransformerBlocks: ditParams.allBlocks,
        outputHead: ditParams.outputHead,
      },
    },
    performanceMetrics: {
      benchmarkSetup: {
        benchmarkDurationSeconds: durationSec,
        frameRateFps: fps,
        totalRawFrames: analytical.totalRawFrames,
        totalLatentFrames: analytical.latentFrames,
        spatialTokensPerFrame: analytical.spatialTokensPerFrame,
        totalVisualTokens: analytical.totalVisualTokens,
        conditioningTextTokens: analytical.textTokens,
        totalSequenceLength: analytical.totalSeqLen,
      },
      memoryFootprint: {
        units: 'Megabytes (MB) and Gigabytes (GB) for intermediate attention tensors',
        tensorPrecision: 'FP16 (2 bytes per activation/element)',
        fullAttentionMatrix: {
          megabytes: analytical.fullAttnMatrixMb,
          gigabytes: analytical.fullAttnMatrixGb,
          exceedsColabT4Vram16GB: analytical.exceedsT4Memory,
          colabT4Status: analytical.exceedsT4Memory
            ? 'Exceeds 16GB VRAM limit (Out of Memory without streaming)'
            : 'Fits comfortably on free Colab T4 (<= 16GB)',
          explanation:
            'Quadratic growth makes long clips expensive in time and, without memory-efficient attention, in memory.',
        },
        decoupledAttentionMatrix: {
          megabytes: analytical.decoupledAttnMatrixMb,
          gigabytes: analytical.decoupledAttnMatrixGb,
        },
        chunkedStreamingAttentionMatrix: {
          megabytes: analytical.chunkedAttnMatrixMb,
          gigabytes: analytical.chunkedAttnMatrixGb,
          numChunks: analytical.numChunks,
          chunkSequenceLength: analytical.chunkSeqLen,
        },
        kvCacheVramEstimate: {
          fullAttentionVramGB: longVideoSpecs.fullAttnVramGb,
          chunkedStreamingKvVramGB: longVideoSpecs.chunkedKvVramGb,
          vramSavingsGB: parseFloat(vramSavings.toFixed(2)),
        },
      },
      computeComplexity: {
        units: 'TeraFLOPs (TFLOPs) for forward attention across entire clip',
        analyticalFlopsFormula: '2 * depth * heads * seq_len^2 * head_dim',
        fullAttentionTFlops: analytical.fullAttnFlopsTFlops,
        decoupledSpatialTemporalTFlops: analytical.decoupledAttnFlopsTFlops,
        chunkedAttentionTFlops: analytical.chunkedAttnFlopsTFlops,
        complexityReductionFactor: `${longVideoSpecs.attentionComplexityReduction.toFixed(1)}x`,
        flopsSavingsPercentVsFullAttention: `${flopsSavings}%`,
      },
    },
  };
}

/**
 * Exports the preset's architecture and performance configuration as a downloadable JSON file.
 * Returns the exported object and the filename.
 */
export function exportPresetAsJsonFile(
  presetKey: PresetKey,
  durationSec = 30,
  fps = 16
): { data: PresetArchitectureMetricsExport; filename: string } {
  const data = generatePresetExportData(presetKey, durationSec, fps);
  const filename = `vidgen-${presetKey}-architecture-metrics.json`;
  const jsonString = JSON.stringify(data, null, 2);

  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    const blob = new Blob([jsonString], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  return { data, filename };
}
