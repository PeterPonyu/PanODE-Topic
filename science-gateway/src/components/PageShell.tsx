import { SITE } from '@/lib/site';

export default function PageShell({
  title,
  kicker,
  children,
}: {
  title: string;
  kicker?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      {kicker ? (
        <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
          {kicker}
        </p>
      ) : null}
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">{title}</h1>
      <div className="mt-8 space-y-6 text-slate-700">{children}</div>
    </div>
  );
}

export function ClaimBlock() {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
      <h2 className="text-lg font-semibold text-slate-900">Primary claim</h2>
      <p className="mt-3 text-slate-700">{SITE.primaryClaim}</p>
      <p className="mt-4 text-sm text-slate-500">
        Scope: unpublished results site. Refutation hooks belong on the Claims route.
      </p>
    </section>
  );
}
