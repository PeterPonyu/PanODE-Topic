#!/usr/bin/env node
import { cpSync, existsSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const outDir = join(process.cwd(), 'out');
const docsDir = join(process.cwd(), '..', 'docs');

if (!existsSync(outDir)) {
  console.error('sync-docs: missing out/ — run npm run build first');
  process.exit(1);
}

rmSync(docsDir, { recursive: true, force: true });
cpSync(outDir, docsDir, { recursive: true });
console.log(`sync-docs: copied out/ → ${docsDir}`);
