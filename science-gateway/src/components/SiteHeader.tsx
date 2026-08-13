'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { badgeEntries, isExternalHref } from '@/lib/badges';
import { ROUTES, SITE } from '@/lib/site';

function navClass(active: boolean): string {
  return [
    'shrink-0 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-semibold transition-colors',
    active
      ? 'bg-teal-50 text-teal-700 ring-1 ring-teal-100'
      : 'text-slate-600 hover:bg-teal-50 hover:text-teal-700',
  ].join(' ');
}

function chipClass(): string {
  return 'shrink-0 whitespace-nowrap rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-500 transition-colors hover:border-teal-300 hover:text-teal-700';
}

function disabledBadgeClass(): string {
  return 'shrink-0 cursor-not-allowed whitespace-nowrap rounded-lg border border-dashed border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-400';
}

function isActive(pathname: string, href: string): boolean {
  if (href === '/') {
    return pathname === '/';
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function SiteHeader() {
  const pathname = usePathname() || '/';
  const [menuOpen, setMenuOpen] = useState(false);

  const seriesLeaves = [
    { href: SITE.homepage, label: 'Homepage' },
    { href: SITE.scportal, label: 'SCPortal' },
  ];

  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 shadow-sm backdrop-blur">
      <div className="mx-auto flex min-h-14 max-w-7xl flex-wrap items-center justify-between gap-2 px-4 py-2 sm:px-6 lg:flex-nowrap">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2 text-base font-bold tracking-tight text-brand"
          onClick={() => setMenuOpen(false)}
        >
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-brand text-xs font-bold text-white">
            sg
          </span>
          <span className="max-w-[12rem] truncate text-slate-800 sm:max-w-xs">{SITE.navTitle}</span>
        </Link>

        <nav
          className="hidden min-w-0 flex-1 items-center justify-center gap-1 lg:flex"
          aria-label="Primary"
        >
          {ROUTES.map((item) => (
            <Link key={item.href} href={item.href} className={navClass(isActive(pathname, item.href))}>
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="hidden shrink-0 flex-wrap items-center justify-end gap-2 lg:flex">
          {seriesLeaves.map((item) => (
            <a key={item.href} href={item.href} className={chipClass()}>
              {item.label}
            </a>
          ))}
          {badgeEntries().map(({ key, badge }) =>
            badge.enabled && badge.href ? (
              <a
                key={key}
                href={badge.href}
                className={chipClass()}
                {...(isExternalHref(badge.href)
                  ? { target: '_blank', rel: 'noopener noreferrer' }
                  : {})}
              >
                {badge.label}
              </a>
            ) : (
              <span
                key={key}
                className={disabledBadgeClass()}
                aria-disabled="true"
                title={badge.disabledReason}
              >
                {badge.label}
              </span>
            ),
          )}
        </div>

        <button
          type="button"
          className="inline-flex min-h-11 min-w-11 items-center justify-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 transition-colors hover:border-teal-300 hover:text-teal-700 lg:hidden"
          aria-expanded={menuOpen}
          aria-controls="sg-mobile-nav"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span>{menuOpen ? 'Close' : 'Menu'}</span>
          <span aria-hidden="true">{menuOpen ? '×' : '☰'}</span>
        </button>
      </div>

      <div
        id="sg-mobile-nav"
        className={`${menuOpen ? 'block' : 'hidden'} border-t border-slate-200 bg-white px-4 py-3 lg:hidden`}
      >
        <nav className="mx-auto grid max-w-7xl gap-2" aria-label="Mobile">
          {ROUTES.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={navClass(isActive(pathname, item.href))}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </Link>
          ))}
          <div className="mt-2 grid gap-2 border-t border-slate-100 pt-3 text-sm font-semibold">
            {seriesLeaves.map((item) => (
              <a key={item.href} href={item.href} className="text-slate-600 hover:text-teal-700">
                {item.label}
              </a>
            ))}
            {badgeEntries().map(({ key, badge }) =>
              badge.enabled && badge.href ? (
                <a
                  key={key}
                  href={badge.href}
                  className="text-slate-600 hover:text-teal-700"
                  {...(isExternalHref(badge.href)
                    ? { target: '_blank', rel: 'noopener noreferrer' }
                    : {})}
                >
                  {badge.label}
                </a>
              ) : (
                <span key={key} className="text-slate-400" aria-disabled="true">
                  {badge.label} ({badge.disabledReason})
                </span>
              ),
            )}
          </div>
        </nav>
      </div>
    </header>
  );
}
