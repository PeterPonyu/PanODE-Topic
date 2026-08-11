import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = "http://localhost:3099/figure1";
const OUT = path.resolve(__dirname, "..", "benchmarks", "paper_figures");

for (const series of ["dpmm", "topic"]) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 700, height: 700 }, deviceScaleFactor: 3 });
  await page.goto(`${BASE}/${series}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  const el = await page.$("#figure1-root");
  if (el) {
    const outPath = path.join(OUT, series, `Fig1_arch_${series}.png`);
    await el.screenshot({ path: outPath, type: "png" });
    console.log(`Saved: ${outPath}`);
  } else {
    console.error(`#figure1-root not found for ${series}`);
  }
  await browser.close();
}
