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
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      {kicker ? (
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-800">
          {kicker}
        </p>
      ) : null}
      <h1 className="mt-3 font-mono text-3xl font-bold tracking-tight text-slate-900">{title}</h1>
      <div className="mt-8 space-y-6 text-slate-700">{children}</div>
    </div>
  );
}

export function ClaimBlock() {
  return (
    <section className="rounded-xl border border-indigo-100 bg-indigo-50/40 p-5">
      <h2 className="font-mono text-sm font-semibold text-indigo-900">Public code page</h2>
      <p className="mt-3 text-sm leading-6 text-slate-700">{SITE.primaryClaim}</p>
    </section>
  );
}
