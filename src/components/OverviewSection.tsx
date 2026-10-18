import React, { useState } from 'react';
import { Layers, HardDrive, Scissors, Sparkles, ChevronDown, CheckCircle2, ShieldCheck } from 'lucide-react';

export const OverviewSection: React.FC = () => {
  const [openAccordion, setOpenAccordion] = useState<string | null>(null);

  const toggleAccordion = (id: string) => {
    setOpenAccordion((prev) => (prev === id ? null : id));
  };

  const coreConcepts = [
    {
      icon: Layers,
      title: '5-Second Causal Chunks',
      badge: '81 frames @ 16 FPS',
      desc: 'Instead of trying to generate 30 seconds at once—which explodes memory—videos are synthesized forward in 5-second increments.',
      color: 'text-indigo-400 bg-indigo-950/60 border-indigo-800/60',
    },
    {
      icon: HardDrive,
      title: 'Flat O(1) Memory',
      badge: 'Constant ~18.4 GB',
      desc: 'Generation stays strictly within memory budget. Whether producing 5 seconds or 5 minutes, VRAM never climbs or crashes with OOM.',
      color: 'text-cyan-400 bg-cyan-950/60 border-cyan-800/60',
    },
    {
      icon: Scissors,
      title: 'Seamless Seam Blending',
      badge: '1-Latent Cosine',
      desc: 'A 1-frame boundary overlap smoothly fades adjacent chunks together using cosine weighting, eliminating visible jump cuts.',
      color: 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60',
    },
    {
      icon: Sparkles,
      title: 'Decoupled Attention',
      badge: 'Spatial + Temporal',
      desc: 'Separates fine spatial details from time-based motion. Slashes attention compute by over 80% while retaining character consistency.',
      color: 'text-sky-400 bg-sky-950/60 border-sky-800/60',
    },
  ];

  return (
    <section id="overview" className="scroll-mt-20 space-y-6">
      {/* Section Header */}
      <div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
          Overview
        </h2>
        <p className="mt-1 text-base text-slate-300">
          How Vidgen delivers long-video synthesis without hardware bottlenecks.
        </p>
      </div>

      {/* 4 Core Concept Cards (Mobile: 1 col, Tablet: 2 col, Desktop: 4 col) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {coreConcepts.map((item, idx) => {
          const Icon = item.icon;
          return (
            <div
              key={idx}
              className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800/90 flex flex-col justify-between space-y-3 hover:border-slate-700 transition-colors"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center border ${item.color}`}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300">
                    {item.badge}
                  </span>
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {item.title}
                </h3>
                <p className="text-base text-slate-300 leading-relaxed">
                  {item.desc}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Accordion Container for Deep Technical Details */}
      <div className="space-y-3 pt-2">
        {/* Accordion 1: Mathematical Foundation */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('math')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'math'}
          >
            <div className="flex items-center gap-2.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span>Deep Technical Specs: 3D Causal Latents &amp; RoPE Partition</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'math' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'math' && (
            <div className="px-5 pb-5 pt-2 border-t border-slate-800/60 text-sm text-slate-300 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs font-mono text-cyan-400 block mb-1">
                    Frame Rule: T = 4k + 1
                  </span>
                  <p className="text-xs text-slate-300">
                    4x temporal VAE compression preserves single-frame image encoding ($k=0 \to 1$) and maps 81 frames to 21 latent frames cleanly.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs font-mono text-cyan-400 block mb-1">
                    RoPE 44 / 42 / 42 Split
                  </span>
                  <p className="text-xs text-slate-300">
                    128-dim attention heads partitioned evenly into time ($dt=44$), height ($dh=42$), and width ($dw=42$) for continuous coordinate geometry.
                  </p>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-950/70 border border-slate-800">
                  <span className="text-xs font-mono text-cyan-400 block mb-1">
                    16-Channel Latents
                  </span>
                  <p className="text-xs text-slate-300">
                    Causal 3D convolutions with zero mean and unit variance normalization prevent drift across long sliding horizons.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Accordion 2: System Audit Status */}
        <div className="rounded-2xl bg-slate-900/60 border border-slate-800 overflow-hidden">
          <button
            onClick={() => toggleAccordion('audit')}
            className="w-full min-h-[48px] px-5 py-3.5 flex items-center justify-between text-left text-sm font-semibold text-slate-200 hover:text-white hover:bg-slate-800/40 transition-colors"
            aria-expanded={openAccordion === 'audit'}
          >
            <div className="flex items-center gap-2.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Full System Audit: Parts 0 through 13 Passing</span>
            </div>
            <ChevronDown
              className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                openAccordion === 'audit' ? 'rotate-180' : ''
              }`}
            />
          </button>

          {openAccordion === 'audit' && (
            <div className="px-5 pb-5 pt-2 border-t border-slate-800/60 text-sm text-slate-300 space-y-3">
              <p className="text-xs text-slate-400">
                All algorithmic foundations are verified in Python CPU unit tests with zero mock shortcuts.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Part 0-4: Bucketing, 3D VAE, 3D RoPE (Passing)</span>
                </div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Part 5-8: umT5, DiT Backbone, Rectified Flow (Passing)</span>
                </div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Part 9-11: Ulysses Parallelism, VBench, LoRA (Passing)</span>
                </div>
                <div className="flex items-center gap-2 p-2 rounded-lg bg-slate-950/60 border border-slate-800">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span>Part 12-13: 30-Second Chunked KV-Cache (Passing)</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
