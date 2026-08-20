import { existsSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join, extname } from 'node:path';

const outDir = join(process.cwd(), 'out');
writeFileSync(join(outDir, '.nojekyll'), '');

const figuresDir = join(outDir, 'figures');
if (existsSync(figuresDir)) {
  for (const name of readdirSync(figuresDir)) {
    const ext = extname(name).toLowerCase();
    if (['.png', '.pdf', '.jpg', '.jpeg', '.webp'].includes(ext)) {
      rmSync(join(figuresDir, name), { force: true });
    }
  }
}

console.log('post-export: wrote out/.nojekyll and stripped unpublished rasters');
