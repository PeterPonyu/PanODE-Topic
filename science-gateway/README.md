# Science Gateway — Next.js reference prototype

Minimal **Next.js App Router static export** pattern for unpublished LAPS leaves.

**Contract:** `labs/DESIGN-science-gateway.md`  
**PRD / tests:** `.omx/plans/prd-science-gateway-next.md`

## Quick start

```bash
cd .omx/prototypes/science-gateway-next
npm ci
NEXT_PUBLIC_BASE_PATH=/science-gateway-reference npm run build
npm run verify
```

Output: `out/` with Home + Results / Methods / Evidence / Claims.

## Per-site migration checklist

1. Copy this tree into the portal repo (or submodule the shared components later).
2. Set `src/lib/site.ts` — title, object, claim, badges (fail-closed).
3. Set `NEXT_PUBLIC_BASE_PATH=/<pages-repo>/` to match GitHub Pages URL math.
4. Replace placeholder copy on each route.
5. Wire `.github/workflows/pages.yml` with the correct `basePath`.
6. Run G1–G11 from the PRD before enabling Site badge.

## Do not

- Mass-edit `active/*/docs/` from this reference alone.
- Add Abstract / Cite / Team routes for unpublished sites.
- Enable Code badge until anonymous public HTTPS 200.
- Invent article DOI.

## scCCVGBen

Published benchmark — **exempt** from this IA. Align-only per parent PRD.
