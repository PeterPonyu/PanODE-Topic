import PageShell from '@/components/PageShell';
import { METRIC_LOCKS, SITE } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Claims" kicker="Falsifiable statements">
      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 1 — split outcomes</h2>
        <p className="mt-3 text-slate-700">{SITE.primaryClaim}</p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Would refute
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Topic-FM beats Pure-VAE on NMI or ARI under reproduced 16-core protocol</li>
          <li>Flow refinement changes decoder β programs without explicit retraining claim</li>
        </ul>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 2 — interpretability object</h2>
        <p className="mt-3 text-slate-700">
          Decoder β gene programs (Fig. 6) are the primary biological readout — not clustering
          leaderboard dominance.
        </p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Out of scope
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Single-metric “best method” narrative across NMI and ASW simultaneously</li>
          <li>Fig. 2 four-scatter PNG on this Site (caption-only)</li>
          <li>Publication-forward packaging or invented article DOI</li>
        </ul>
      </section>

      <p className="text-xs text-slate-500">
        Locked values: Pure-VAE NMI {METRIC_LOCKS.pureVae.nmi} · ARI {METRIC_LOCKS.pureVae.ari}; Topic-FM
        ASW {METRIC_LOCKS.topicFm.asw} · DAV {METRIC_LOCKS.topicFm.dav}.
      </p>
    </PageShell>
  );
}
