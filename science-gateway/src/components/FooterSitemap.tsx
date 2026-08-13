import { badgeEntries } from '@/lib/badges';
import { ROUTES, SITE } from '@/lib/site';

export default function FooterSitemap() {
  return (
    <footer className="mt-12 border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 text-[13px] text-slate-500 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <span className="font-medium text-slate-700">{SITE.title}</span>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          {ROUTES.map((route) => (
            <a key={route.href} href={route.href} className="transition-colors hover:text-teal-600">
              {route.label}
            </a>
          ))}
          <a href={SITE.homepage} className="transition-colors hover:text-teal-600">
            Homepage
          </a>
          <a href={SITE.scportal} className="transition-colors hover:text-teal-600">
            SCPortal
          </a>
          {badgeEntries()
            .filter(({ badge }) => badge.enabled && badge.href)
            .map(({ key, badge }) => (
              <a
                key={key}
                href={badge.href}
                className="transition-colors hover:text-teal-600"
                target="_blank"
                rel="noopener noreferrer"
              >
                {badge.label}
              </a>
            ))}
        </div>
      </div>
    </footer>
  );
}
