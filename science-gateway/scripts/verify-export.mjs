#!/usr/bin/env node
/**
 * Minimal G1/G3/G6/G9 checks for science-gateway static export.
 * Usage: node scripts/verify-export.mjs
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const out = join(process.cwd(), 'out');
const required = [
  'index.html',
  'results/index.html',
  'methods/index.html',
  'evidence/index.html',
  'claims/index.html',
  '.nojekyll',
];
const forbidden = ['abstract', 'cite', 'team'];
const denylist = ['PEERJ_REVIEWER_FAQ.md', 'PEERJ_PORTAL_INPUTS.txt', 'superpowers'];

let failed = 0;

for (const rel of required) {
  const p = join(out, rel);
  if (!existsSync(p)) {
    console.error(`FAIL G1: missing ${rel}`);
    failed += 1;
  }
}

for (const dir of forbidden) {
  if (existsSync(join(out, dir))) {
    console.error(`FAIL G3: forbidden route directory out/${dir}/`);
    failed += 1;
  }
}

function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (denylist.includes(entry.name)) {
        console.error(`FAIL G9: denylist dir ${p}`);
        failed += 1;
      }
      walk(p);
    } else if (denylist.some((d) => entry.name.includes(d))) {
      console.error(`FAIL G9: denylist file ${p}`);
      failed += 1;
    }
  }
}

if (existsSync(out)) {
  walk(out);
  const html = readFileSync(join(out, 'index.html'), 'utf8');
  if (/github\.com\/PeterPonyu\/HetCLOP/i.test(html)) {
    console.error('FAIL G6: private HetCLOP Code href in index.html');
    failed += 1;
  }
  for (const label of ['Abstract', 'Cite', 'Team']) {
    if (new RegExp(`>${label}<`, 'i').test(html)) {
      console.error(`FAIL G3: journal nav label "${label}" in index.html`);
      failed += 1;
    }
  }
  if (/Get started|Try now|Launch/i.test(html)) {
    console.error('FAIL G7: product headline pattern in index.html');
    failed += 1;
  }
}

if (failed) {
  process.exit(1);
}

console.log(`verify-export: ok (${required.length} required paths)`);
