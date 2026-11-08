import React from 'react';

export const OverviewSection: React.FC = () => {
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
    </section>
  );
};
