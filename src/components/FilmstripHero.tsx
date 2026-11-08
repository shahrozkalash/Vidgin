import React from 'react';

interface FilmstripHeroProps {
  onTryItClick: () => void;
}

export const FilmstripHero: React.FC<FilmstripHeroProps> = ({ onTryItClick }) => {
  return (
    <section className="pt-10 pb-8 sm:pt-16 sm:pb-10 px-4">
      <div className="max-w-3xl mx-auto text-center space-y-5">
        <h1 className="text-4xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-slate-900 leading-[1.05]">
          Describe it.
          <br />
          <span className="text-green-700">Vidgen films it.</span>
        </h1>

        <p className="text-lg sm:text-xl text-slate-600 max-w-xl mx-auto leading-relaxed">
          Vidgen turns your words into video, one small step at a time.
        </p>

        <div className="pt-1">
          <button
            onClick={onTryItClick}
            className="min-h-[48px] px-7 py-3 rounded-full bg-green-600 hover:bg-green-700 text-white font-semibold text-base shadow-md shadow-green-600/20 active:scale-[0.98] transition-all cursor-pointer"
          >
            Try it — it's simple
          </button>
        </div>
      </div>
    </section>
  );
};
