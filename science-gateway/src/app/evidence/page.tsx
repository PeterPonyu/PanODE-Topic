import PageShell from '@/components/PageShell';
import StatTile from '@/components/StatTile';
import { METRIC_LOCKS } from '@/lib/site';

export default function EvidencePage() {
  return (
    <PageShell title="Evidence" kicker="Metrics and controls">
      <p>
        Verifier-gated metric locks from matched Wilcoxon externals. Concordance and geometry are
        reported separately — not collapsed into a single win narrative.
      </p>

      <div className="grid gap-4 sm:grid-cols-2">
        <StatTile
          value={`NMI ${METRIC_LOCKS.pureVae.nmi}`}
          label="Pure-VAE concordance"
          note={`ARI ${METRIC_LOCKS.pureVae.ari} — leads both`}
        />
        <StatTile
          value={`ASW ${METRIC_LOCKS.topicFm.asw}`}
          label="Topic-FM geometry"
          note={`DAV ${METRIC_LOCKS.topicFm.dav} — not NMI/ARI win`}
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatTile value="11" label="Externals" />
        <StatTile value="37" label="Metrics each" />
        <StatTile value="407" label="Wilcoxon rows" note="11 × 37" />
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Split-outcome interpretation</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-700">
          <li>Pure-VAE leads NMI (0.564) and ARI (0.363) — concordance lock</li>
          <li>Topic-FM-Transformer leads ASW (0.501) and DAV (0.763) — geometry lock</li>
          <li>Flow refinement does not rewrite β or flip 16-core NMI/ARI rankings</li>
        </ul>
      </section>
    </PageShell>
  );
}
