import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Scope" kicker="Code page">
      <section className="rounded-xl border border-indigo-100 bg-indigo-50/40 p-5">
        <h2 className="text-lg font-semibold text-slate-900">What this site is</h2>
        <p className="mt-3 text-slate-700">
          A companion page for the public PanODE-Topic repository. It describes model names,
          runners, and directory layout.
        </p>
      </section>
      <section className="rounded-xl border border-slate-200 bg-white p-5">
        <h2 className="text-lg font-semibold text-slate-900">What this site is not</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-700">
          <li>Not a journal article and not a preprint landing page</li>
          <li>No article DOI</li>
          <li>No manuscript figures or evaluation-score tables</li>
        </ul>
      </section>
      <p>
        Code:{' '}
        <a href={SITE.github} className="text-indigo-800 underline-offset-2 hover:underline">
          {SITE.github.replace('https://', '')}
        </a>
      </p>
    </PageShell>
  );
}
