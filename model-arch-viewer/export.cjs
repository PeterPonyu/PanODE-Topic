const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch();
  for (const s of ['dpmm', 'topic']) {
    const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
    await page.goto('http://localhost:3099/figure1/' + s, { waitUntil: 'networkidle' });
    const el = await page.locator('#figure1-root');
    const outPath = '../benchmarks/paper_figures/' + s + '/Fig1_arch_' + s + '.png';
    await el.screenshot({ path: outPath, scale: 'device' });
    console.log(s + ' -> ' + outPath);
    await page.close();
  }
  await browser.close();
  process.exit(0);
})();
