#!/usr/bin/env node
/**
 * Scaffold a science-gateway Next.js portal into a Pages repo.
 * Usage: node scaffold-portal.mjs <configKey> <targetDir>
 */
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execSync } from 'node:child_process';
import { SITE_CONFIGS } from './site-configs.mjs';

const __dirname = dirname(fileURLToPath(import.meta.url));
const prototypeRoot = join(__dirname, '..');
const labsRoot = resolve(prototypeRoot, '../../..');

const configKey = process.argv[2];
const targetDir = resolve(process.argv[3] ?? '');
const cfg = SITE_CONFIGS[configKey];

if (!cfg || !targetDir) {
  console.error('Usage: node scaffold-portal.mjs <b1|a3|b28|spgd> <targetDir>');
  process.exit(1);
}

const skip = new Set(['node_modules', '.next', 'out', '.git']);
function copyPrototype(src, dest) {
  mkdirSync(dest, { recursive: true });
  for (const entry of readdirSync(src, { withFileTypes: true })) {
    if (skip.has(entry.name)) continue;
    const from = join(src, entry.name);
    const to = join(dest, entry.name);
    cpSync(from, to, { recursive: true });
  }
}

console.log(`Scaffolding ${cfg.slug} → ${targetDir}`);
copyPrototype(prototypeRoot, targetDir);

const basePath = `/${cfg.slug}`;
writeFileSync(
  join(targetDir, 'src/lib/site.ts'),
  `/**
 * ${cfg.slug} — science gateway config (generated; edit in site-configs.mjs + re-scaffold).
 */
export const SITE = {
  slug: '${cfg.slug}',
  shortName: ${JSON.stringify(cfg.shortName)},
  title: ${JSON.stringify(cfg.title)},
  kicker: ${JSON.stringify(cfg.kicker)},
  lead: ${JSON.stringify(cfg.lead)},
  physicalObject: ${JSON.stringify(cfg.physicalObject)},
  primaryClaim: ${JSON.stringify(cfg.primaryClaim)},
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
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
    enabled: false,
    disabledReason: 'No anonymous public HTTPS 200 yet',
  } satisfies BadgeConfig,
  site: {
    label: 'Site',
    href: ${JSON.stringify(cfg.siteUrl)},
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    href: 'https://doi.org/${cfg.archiveDoi}',
    enabled: true,
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'On acceptance',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Results', number: '01' },
  { href: '/methods', label: 'Methods', number: '02' },
  { href: '/evidence', label: 'Evidence', number: '03' },
  { href: '/claims', label: 'Claims', number: '04' },
] as const;

export const STATS = ${JSON.stringify(cfg.stats, null, 2)} as const;

export const RESULTS_FIGURES = ${JSON.stringify(cfg.resultsFigures, null, 2)} as const;

export const EVIDENCE_TILES = ${JSON.stringify(cfg.evidenceTiles, null, 2)} as const;

export const METHODS_SUMMARY = ${JSON.stringify(cfg.methodsSummary)} as const;

export const CLAIMS = ${JSON.stringify(cfg.claims, null, 2)} as const;
`,
);

writeFileSync(
  join(targetDir, 'src/app/page.tsx'),
  `import { ClaimBlock } from '@/components/PageShell';
import RouteCards from '@/components/RouteCards';
import StatTile from '@/components/StatTile';
import { SITE, STATS } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
        {SITE.kicker}
      </p>
      <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-900">{SITE.title}</h1>
      <p className="mt-4 max-w-3xl text-lg text-slate-700">{SITE.lead}</p>

      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        {STATS.map((tile) => (
          <StatTile key={tile.label} value={tile.value} label={tile.label} />
        ))}
      </div>

      <section className="mt-10 rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Physical object
        </h2>
        <p className="mt-2 text-slate-800">{SITE.physicalObject}</p>
      </section>

      <div className="mt-8">
        <ClaimBlock />
      </div>

      <section className="mt-10">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Explore
        </h2>
        <RouteCards />
      </section>
    </div>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/app/results/page.tsx'),
  `import FigurePanel from '@/components/FigurePanel';
import PageShell from '@/components/PageShell';
import { RESULTS_FIGURES } from '@/lib/site';

export default function ResultsPage() {
  return (
    <PageShell title="Results" kicker="Primary outcomes">
      <p>Primary outcomes and figure panels — labeled as results, not a generic figures dump.</p>
      <div className="mt-8 grid gap-8">
        {RESULTS_FIGURES.map((fig) => (
          <FigurePanel key={fig.src} src={fig.src} alt={fig.alt} caption={fig.caption} />
        ))}
      </div>
    </PageShell>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/app/methods/page.tsx'),
  `import PageShell from '@/components/PageShell';
import { METHODS_SUMMARY } from '@/lib/site';

export default function MethodsPage() {
  return (
    <PageShell title="Methods" kicker="Protocol and scope">
      <p>{METHODS_SUMMARY}</p>
      <p className="text-sm text-slate-500">
        Reproducibility: public code is not published yet. Archive DOI in the header when enabled.
      </p>
    </PageShell>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/app/evidence/page.tsx'),
  `import PageShell from '@/components/PageShell';
import StatTile from '@/components/StatTile';
import { EVIDENCE_TILES } from '@/lib/site';

export default function EvidencePage() {
  return (
    <PageShell title="Evidence" kicker="Metrics and controls">
      <p>Verifier-gated numerals and negative controls supporting the primary claim.</p>
      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        {EVIDENCE_TILES.map((tile) => (
          <StatTile key={tile.label} value={tile.value} label={tile.label} />
        ))}
      </div>
    </PageShell>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/app/claims/page.tsx'),
  `import PageShell from '@/components/PageShell';
import { CLAIMS } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Claims" kicker="Falsifiable statements">
      <div className="space-y-6">
        {CLAIMS.map((item) => (
          <article key={item.claim} className="rounded-2xl border border-slate-200 bg-white/80 p-6">
            <h2 className="text-lg font-semibold text-slate-900">{item.claim}</h2>
            <p className="mt-2 text-sm text-slate-600">
              <strong>Scope:</strong> {item.scope}
            </p>
            <p className="mt-2 text-sm text-slate-500">
              <strong>Would refute:</strong> {item.refutation}
            </p>
          </article>
        ))}
      </div>
    </PageShell>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/components/StatTile.tsx'),
  `export default function StatTile({ value, label }: { value: string; label: string }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-5">
      <strong className="block font-mono text-2xl text-brand">{value}</strong>
      <span className="mt-1 block text-sm text-slate-600">{label}</span>
    </div>
  );
}
`,
);

writeFileSync(
  join(targetDir, 'src/components/FigurePanel.tsx'),
  `import Image from 'next/image';

export default function FigurePanel({
  src,
  alt,
  caption,
}: {
  src: string;
  alt: string;
  caption: string;
}) {
  return (
    <figure className="overflow-hidden rounded-2xl border border-slate-200 bg-white/80">
      <Image src={src} alt={alt} width={1600} height={900} className="h-auto w-full" unoptimized />
      <figcaption className="border-t border-slate-100 px-5 py-4 text-sm text-slate-600">
        {caption}
      </figcaption>
    </figure>
  );
}
`,
);

const headerPath = join(targetDir, 'src/components/SiteHeader.tsx');
let header = readFileSync(headerPath, 'utf8');
header = header.replace(
  '<span className="text-slate-800">{SITE.title}</span>',
  '<span className="text-slate-800">{SITE.shortName}</span>',
);
writeFileSync(headerPath, header);

writeFileSync(
  join(targetDir, 'next.config.mjs'),
  readFileSync(join(prototypeRoot, 'next.config.mjs'), 'utf8'),
);

writeFileSync(
  join(targetDir, '.github/workflows/pages.yml'),
  `name: Deploy GitHub Pages

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

env:
  NEXT_PUBLIC_BASE_PATH: ${basePath}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
      - uses: actions/setup-node@49933ea5288caeca8642d29e6ab316508ebc106b
        with:
          node-version: '20'
          cache: npm
      - run: npm ci
      - run: npm run build
      - run: npm run verify
      - run: rm -rf docs && cp -r out docs
      - name: Upload Pages artifact
        if: github.event_name != 'pull_request'
        uses: actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa
        with:
          path: docs

  deploy:
    if: github.event_name != 'pull_request'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: \${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e
`,
);

const mediaDest = join(targetDir, 'public/media');
rmSync(mediaDest, { recursive: true, force: true });
mkdirSync(mediaDest, { recursive: true });
const mediaSrc = join(labsRoot, cfg.mediaSource);
for (const entry of readdirSync(mediaSrc)) {
  if (entry.endsWith('.png')) {
    cpSync(join(mediaSrc, entry), join(mediaDest, entry));
  }
}

const gatewaySrc = join(labsRoot, 'pages-gateway-tests');
const gatewayDest = join(targetDir, 'pages-gateway-tests');
if (!existsSync(gatewayDest)) {
  cpSync(gatewaySrc, gatewayDest, {
    recursive: true,
    filter: (src) => !src.includes('node_modules') && !src.includes('test-results') && !src.includes('playwright-report'),
  });
}

const lapsWorkflow = join(labsRoot, 'pages-gateway-tests/.github/workflows/laps-gateway-tests.yml');
mkdirSync(join(targetDir, '.github/workflows'), { recursive: true });
cpSync(lapsWorkflow, join(targetDir, '.github/workflows/laps-gateway-tests.yml'));

writeFileSync(
  join(targetDir, 'README.md'),
  `# ${cfg.slug}

Science gateway (Next.js static export) for unpublished results.

- Live: ${cfg.siteUrl}
- IA: Home / Results / Methods / Evidence / Claims
- Archive: https://doi.org/${cfg.archiveDoi}

\`\`\`bash
NEXT_PUBLIC_BASE_PATH=${basePath} npm ci && npm run build && npm run verify
\`\`\`
`,
);

console.log('Installing dependencies and building…');
execSync('npm ci', { cwd: targetDir, stdio: 'inherit' });
execSync(`NEXT_PUBLIC_BASE_PATH=${basePath} npm run build`, { cwd: targetDir, stdio: 'inherit' });
execSync('npm run verify', { cwd: targetDir, stdio: 'inherit' });
rmSync(join(targetDir, 'docs'), { recursive: true, force: true });
cpSync(join(targetDir, 'out'), join(targetDir, 'docs'), { recursive: true });
console.log(`Done: ${targetDir} (docs/ synced from out/)`);
