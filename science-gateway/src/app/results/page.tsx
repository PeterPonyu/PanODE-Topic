import PageShell from '@/components/PageShell';
import FigurePanel from '@/components/FigurePanel';

const SHIPPED_FIGURES = ['F01', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10'] as const;

export default function ResultsPage() {
  return (
    <PageShell title="Results" kicker="Outcome figures">
      <p>
        Primary interpretability results are β gene programs (Fig. 6). Fig. 2 documents the
        four-scatter trade-off layout in caption only — the PNG is not wired on this Site.
      </p>

      <section className="rounded-2xl border border-dashed border-slate-300 bg-slate-50/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Fig. 2 · caption only</h2>
        <p className="mt-2 text-slate-700">
          Four-scatter trade-off panel (NMI vs ASW across variants). Live manuscript uses TikZ
          boxes; the four-scatter PNG from benchmarks is intentionally not shipped here to avoid
          contradicting the caption-first layout lock.
        </p>
      </section>

      <div className="grid gap-6">
        {SHIPPED_FIGURES.map((file) => (
          <FigurePanel
            key={file}
            src={`/figures/${file}.png`}
            alt={`${file} results panel`}
            kicker={`${file}`}
            caption={`Results panel ${file}. See manuscript FIGURE-PROVENANCE for source paths.`}
          />
        ))}
      </div>
    </PageShell>
  );
}
