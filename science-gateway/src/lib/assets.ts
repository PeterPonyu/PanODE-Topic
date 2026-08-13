/** Static asset path with basePath prefix for GitHub Pages project sites. */
export function assetPath(relativePath: string): string {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
  const normalized = relativePath.replace(/^\//, '');
  return base ? `${base}/${normalized}` : `/${normalized}`;
}
