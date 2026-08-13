/**
 * PanODE-Topic science gateway site config.
 */
export const SITE = {
  slug: 'PanODE-Topic',
  navTitle: 'PanODE-Topic',
  title:
    'Simplex-constrained topic VAEs with flow refinement recover interpretable gene programs without winning NMI',
  kicker: 'ZF Lab · simplex topics · β gene programs',
  lead:
    'Topic proportions on the simplex and decoder β gene programs. Flow matching tightens geometry; it does not rewrite β and does not flip concordance locks.',
  physicalObject:
    'Decoder β gene programs and setty perturbation-importance heatmaps (Fig. 6). Fig. 2 is caption-only (four-scatter not wired).',
  primaryClaim:
    'Pure-VAE leads NMI/ARI concordance; Topic-FM-Transformer leads ASW/DAV geometry — a split outcome, not a single clustering win.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  github: 'https://github.com/PeterPonyu/PanODE-Topic',
  sisterSite: 'https://peterponyu.github.io/PanODE-DPMM/',
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
    disabledReason: 'No Zenodo record yet',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'On acceptance',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Results', number: '01', blurb: 'β programs (Fig. 6); Fig. 2 caption-only.' },
  { href: '/methods', label: 'Methods', number: '02', blurb: 'Simplex topic VAE, flow refinement, benchmarks.' },
  { href: '/evidence', label: 'Evidence', number: '03', blurb: 'Concordance vs geometry metric locks.' },
  { href: '/claims', label: 'Claims', number: '04', blurb: 'Split-outcome claims and scope limits.' },
] as const;

/** Science locks — verifier-gated numbers */
export const METRIC_LOCKS = {
  pureVae: { nmi: '0.564', ari: '0.363', label: 'Pure-VAE · label concordance' },
  topicFm: { asw: '0.501', dav: '0.763', label: 'Topic-FM-Transformer · geometry' },
} as const;
