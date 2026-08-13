import Link from 'next/link';
import { ROUTES } from '@/lib/site';

export default function RouteCards() {
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      {ROUTES.map((route) => (
        <Link
          key={route.href}
          href={route.href}
          className="card-hover flex gap-4 rounded-2xl border border-slate-200 bg-white/80 p-5"
        >
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-brand text-sm font-bold text-white">
            {route.number}
          </span>
          <div>
            <h3 className="text-lg font-semibold text-slate-900">{route.label}</h3>
            <p className="mt-1 text-sm text-slate-600">{route.blurb}</p>
          </div>
        </Link>
      ))}
    </div>
  );
}
