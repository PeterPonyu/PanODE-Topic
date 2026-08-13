import { ClaimBlock } from '@/components/PageShell';
import RouteCards from '@/components/RouteCards';
import FigurePanel from '@/components/FigurePanel';
import StatTile from '@/components/StatTile';
import { METRIC_LOCKS, SITE } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
        {SITE.kicker}
      </p>
      <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
        {SITE.title}
      </h1>
      <p className="mt-4 max-w-3xl text-lg text-slate-700">{SITE.lead}</p>

      <section className="mt-10 rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Physical object
        </h2>
        <p className="mt-2 text-slate-800">{SITE.physicalObject}</p>
      </section>

      <div className="mt-8">
        <ClaimBlock />
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <div className="rounded-2xl border border-blue-200 bg-blue-50/50 p-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-blue-700">
            {METRIC_LOCKS.pureVae.label}
          </p>
          <div className="mt-3 flex gap-6 font-mono">
            <div>
              <p className="text-2xl font-bold text-brand">{METRIC_LOCKS.pureVae.nmi}</p>
              <p className="text-xs text-slate-600">NMI</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-brand">{METRIC_LOCKS.pureVae.ari}</p>
              <p className="text-xs text-slate-600">ARI</p>
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-600">
            Gaussian-prior Pure-VAE baseline. Concordance lock — Topic-FM does not win NMI or ARI.
          </p>
        </div>
        <div className="rounded-2xl border border-teal-200 bg-teal-50/50 p-6">
          <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">
            {METRIC_LOCKS.topicFm.label}
          </p>
          <div className="mt-3 flex gap-6 font-mono">
            <div>
              <p className="text-2xl font-bold text-teal-700">{METRIC_LOCKS.topicFm.asw}</p>
              <p className="text-xs text-slate-600">ASW</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-teal-700">{METRIC_LOCKS.topicFm.dav}</p>
              <p className="text-xs text-slate-600">DAV</p>
            </div>
          </div>
          <p className="mt-3 text-sm text-slate-600">
            Geometry lock only. Not an NMI win — flow matching sharpens contours without flipping
            concordance.
          </p>
        </div>
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile value="11" label="Wilcoxon externals" />
        <StatTile value="12" label="N_pairs matched" />
        <StatTile value="10" label="Topics / simplex dim" />
        <StatTile value="407" label="Wilcoxon rows" note="11 × 37" />
      </div>

      <section className="mt-10">
        <FigurePanel
          src="/figures/F06.png"
          alt="Setty perturbation-importance heatmaps and decoder-beta readouts"
          kicker="Fig. 6 · β gene programs"
          caption="Biological validation panel: perturbation-importance heatmaps and decoder β readouts. Primary interpretability object for this direction."
        />
      </section>

      <section className="mt-10">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Explore
        </h2>
        <RouteCards />
      </section>
    </div>
  );
}
