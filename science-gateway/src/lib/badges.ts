import { BADGES, type BadgeConfig } from './site';

export function badgeEntries(): Array<{ key: string; badge: BadgeConfig }> {
  return [
    { key: 'archive', badge: BADGES.archive },
    { key: 'code', badge: BADGES.code },
    { key: 'site', badge: BADGES.site },
    { key: 'articleDoi', badge: BADGES.articleDoi },
  ];
}

export function isExternalHref(href: string): boolean {
  return href.startsWith('http');
}
