import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function MethodsPage() {
  return (
    <PageShell title="Methods" kicker="Protocol and definitions">
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Model</h2>
        <p>
          Simplex-constrained neural topic VAE with optional flow-matching refinement. Topic
          proportions live on the simplex; decoder β matrices define interpretable gene programs.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Benchmark protocol</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>11 named Wilcoxon externals, 37 metrics each, 12 matched N_pairs</li>
          <li>Pure-VAE Gaussian baseline for concordance comparison</li>
          <li>Topic-FM variants (Base, Transformer, Contrastive) for geometry comparison</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Exclusions</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>Fig. 2 four-scatter PNG not deployed (caption-only lock)</li>
          <li>No claim that Topic-FM wins NMI or ARI</li>
          <li>No GPU benchmark campaigns from this Site leaf</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Reproducibility</h2>
        <p>
          Public code:{' '}
          <a href={SITE.github} className="text-brand hover:underline" target="_blank" rel="noopener noreferrer">
            github.com/PeterPonyu/PanODE-Topic
          </a>
          . Sister DPMM method:{' '}
          <a href={SITE.sisterSite} className="text-brand hover:underline">
            PanODE-DPMM Site
          </a>
          .
        </p>
      </section>
    </PageShell>
  );
}
