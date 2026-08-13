import { writeFileSync } from 'node:fs';
import { join } from 'node:path';

const outDir = join(process.cwd(), 'out');
writeFileSync(join(outDir, '.nojekyll'), '');
console.log('post-export: wrote out/.nojekyll');
