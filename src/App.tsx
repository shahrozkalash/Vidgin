import React, { useState } from 'react';
import { PresetKey } from './types/vidgen';
import { Navbar } from './components/Navbar';
import { VisualBatchInspector } from './components/VisualBatchInspector';
import { BucketMatrixView } from './components/BucketMatrixView';
import { DistributedSamplerSim } from './components/DistributedSamplerSim';
import { PrecomputeStudio } from './components/PrecomputeStudio';
import { ThroughputBenchmark } from './components/ThroughputBenchmark';
import { Part2SpecsAndTests } from './components/Part2SpecsAndTests';
import { VAEStudio } from './components/VAEStudio';
import { PatchifyRoPEStudio } from './components/PatchifyRoPEStudio';
import { TextConditionerStudio } from './components/TextConditionerStudio';
import { DiTStudio } from './components/DiTStudio';
import { SamplerStudio } from './components/SamplerStudio';
import { ScalingStudio } from './components/ScalingStudio';
import { EvaluationStudio } from './components/EvaluationStudio';
import { ExtensionStudio } from './components/ExtensionStudio';
import { ImprovementStudio } from './components/ImprovementStudio';
import { LongVideoStudio } from './components/LongVideoStudio';

export default function App() {
  const [currentPreset, setCurrentPreset] = useState<PresetKey>('tiny');
  const [activeTab, setActiveTab] = useState<string>('long_video_30s');

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      {/* Studio Header & Navigation */}
      <Navbar
        currentPreset={currentPreset}
        onPresetChange={setCurrentPreset}
        activeTab={activeTab}
        onTabChange={setActiveTab}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {activeTab === 'long_video_30s' && (
          <LongVideoStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'targeted_improvements' && (
          <ImprovementStudio />
        )}

        {activeTab === 'extensions_studio' && (
          <ExtensionStudio />
        )}

        {activeTab === 'evaluation_vbench' && (
          <EvaluationStudio />
        )}

        {activeTab === 'scaling_efficiency' && (
          <ScalingStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'sampler_pipeline' && (
          <SamplerStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'dit_backbone' && (
          <DiTStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'text_conditioner' && (
          <TextConditionerStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'patchify_rope' && (
          <PatchifyRoPEStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'vae' && (
          <VAEStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'batch_inspector' && (
          <VisualBatchInspector currentPreset={currentPreset} />
        )}

        {activeTab === 'bucketing' && (
          <BucketMatrixView currentPreset={currentPreset} />
        )}

        {activeTab === 'sampler' && (
          <DistributedSamplerSim currentPreset={currentPreset} />
        )}

        {activeTab === 'precompute' && (
          <PrecomputeStudio currentPreset={currentPreset} />
        )}

        {activeTab === 'throughput' && (
          <ThroughputBenchmark currentPreset={currentPreset} />
        )}

        {activeTab === 'specs_tests' && (
          <Part2SpecsAndTests currentPreset={currentPreset} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
            <span>VidGen Engine • Part 13: 30-Second Video (481f, 16 FPS, Flat O(1) Memory)</span>
          </div>
          <div className="flex items-center gap-3 text-slate-400">
            <span>T = 4k + 1 (481f / 30s)</span>
            <span>•</span>
            <span>121 Latents</span>
            <span>•</span>
            <span>Block-Causal KV Cache</span>
            <span>•</span>
            <span>16 FPS Wan Target</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
