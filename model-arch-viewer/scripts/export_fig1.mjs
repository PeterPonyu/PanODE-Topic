/**
 * Export per-series Fig.1 architecture diagrams as PNGs via Playwright.
 *
 * Usage:
 *   1. Start the Next.js dev server: npm run dev
 *   2. Run: PLAYWRIGHT_BROWSERS_PATH=$(pwd)/pw-browsers node scripts/export_fig1.mjs
 */
import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..", "..");
const PORT = process.env.PORT || 3000;
const BASE = `http://localhost:${PORT}`;

const EXPORTS = [
  { route: "/figure1/dpmm", out: path.join(ROOT, "benchmarks/paper_figures/dpmm/Fig1_arch_dpmm.png") },
  { route: "/figure1/topic", out: path.join(ROOT, "benchmarks/paper_figures/topic/Fig1_arch_topic.png") },
];

async function main() {
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({
    viewport: { width: 960, height: 1200 },
    deviceScaleFactor: 1.5,
  });

  for (const { route, out } of EXPORTS) {
    const page = await ctx.newPage();
    console.log(`  Loading ${BASE}${route} ...`);
    await page.goto(`${BASE}${route}`, { waitUntil: "networkidle" });
    // Wait for diagrams to render
    await page.waitForTimeout(2000);

    // Screenshot the figure root container only (avoids viewport blank area)
    const el = await page.$("#figure1-root");
    if (el) {
      fs.mkdirSync(path.dirname(out), { recursive: true });
      await el.screenshot({ path: out, type: "png" });
      console.log(`  Saved: ${out}`);
    } else {
      console.log(`  WARNING: content wrapper not found for ${route}`);
      await page.screenshot({ path: out, fullPage: true });
      console.log(`  Saved (fullpage fallback): ${out}`);
    }
    await page.close();
  }

  await browser.close();
  console.log("Done.");
}

main().catch((e) => { console.error(e); process.exit(1); });
