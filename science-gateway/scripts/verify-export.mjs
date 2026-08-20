#!/usr/bin/env node
/**
 * Static-export checks for the PanODE-Topic public code page.
 * Usage: node scripts/verify-export.mjs
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';

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
const leakPatterns = [
  /unpublished results/i,
  /\bNMI\b/,
  /\bARI\b/,
  /\bASW\b/,
  /\bDAV\b/,
  /0\.564/,
  /0\.363/,
  /0\.501/,
  /0\.763/,
  /Get started|Try now|Launch/i,
];
const rasterExt = new Set(['.png', '.pdf', '.jpg', '.jpeg', '.webp']);

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
    } else if (rasterExt.has(extname(entry.name).toLowerCase())) {
      console.error(`FAIL leak: unpublished raster ${p}`);
      failed += 1;
    }
  }
}

if (existsSync(out)) {
  walk(out);
  const htmlFiles = [];
  function collectHtml(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, entry.name);
      if (entry.isDirectory()) collectHtml(p);
      else if (entry.name.endsWith('.html') || entry.name.endsWith('.txt')) {
        htmlFiles.push(p);
      }
    }
  }
  collectHtml(out);
  for (const file of htmlFiles) {
    if (!statSync(file).isFile()) continue;
    const text = readFileSync(file, 'utf8');
    if (/github\.com\/PeterPonyu\/HetCLOP/i.test(text)) {
      console.error(`FAIL G6: private HetCLOP Code href in ${file}`);
      failed += 1;
    }
    for (const label of ['Abstract', 'Cite', 'Team']) {
      if (new RegExp(`>${label}<`, 'i').test(text) && file.endsWith('index.html')) {
        console.error(`FAIL G3: journal nav label "${label}" in ${file}`);
        failed += 1;
      }
    }
    for (const pat of leakPatterns) {
      if (pat.test(text)) {
        console.error(`FAIL leak: ${pat} matched ${file}`);
        failed += 1;
      }
    }
  }
}

if (failed) {
  process.exit(1);
}

console.log(`verify-export: ok (${required.length} required paths)`);
