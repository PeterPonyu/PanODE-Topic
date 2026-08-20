/**
 * PanODE-Topic public code page. Do not put unpublished metric locks or figures here.
 */
export const SITE = {
  slug: 'PanODE-Topic',
  navTitle: 'PanODE-Topic',
  title: 'PanODE-Topic',
  kicker: 'Topic-FM · Dirichlet topic VAE · public code',
  lead:
    'This repository implements Topic-FM and Pure-VAE model families for single-cell representation learning. The page describes the public code. It is not a journal article.',
  physicalObject:
    'A neural topic VAE with simplex topic proportions, a decoder β matrix, and optional flow-matching refinement.',
  primaryClaim:
    'The public tree contains model code, benchmark runners, and evaluation utilities. No article DOI is assigned.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  github: 'https://github.com/PeterPonyu/PanODE-Topic',
} as const;

export type BadgeConfig = {
  label: string;
  href?: string;
  enabled: boolean;
  disabledReason?: string;
};

export const BADGES = {
  code: {
    label: 'Code',
    href: SITE.github,
    enabled: true,
  } satisfies BadgeConfig,
  site: {
    label: 'Site',
    href: 'https://peterponyu.github.io/PanODE-Topic/',
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No Zenodo record',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'Not a journal article',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Layout', number: '01', blurb: 'Public directories in this repository.' },
  { href: '/methods', label: 'Run', number: '02', blurb: 'Topic-FM runners and default settings.' },
  { href: '/evidence', label: 'Models', number: '03', blurb: 'Topic-FM and Pure-VAE families.' },
  { href: '/claims', label: 'Scope', number: '04', blurb: 'Code page, not a paper.' },
] as const;
