import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function MethodsPage() {
  return (
    <PageShell title="How to run" kicker="Runners and defaults">
      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-slate-900">Runners</h2>
        <p>
          Base and single-dataset jobs go through{' '}
          <code className="font-mono text-[13px]">benchmarks/runners/benchmark_base.py</code> with{' '}
          <code className="font-mono text-[13px]">--series topic</code>. Cross-dataset jobs use{' '}
          <code className="font-mono text-[13px]">benchmark_crossdata.py</code> and configured
          registry keys.
        </p>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-slate-900">Default training settings</h2>
        <ul className="list-disc space-y-1 pl-5">
          <li>Learning rate <code className="font-mono text-[13px]">1e-3</code></li>
          <li>Batch size <code className="font-mono text-[13px]">128</code></li>
          <li>Topics / latent dim <code className="font-mono text-[13px]">10</code></li>
          <li>Epochs <code className="font-mono text-[13px]">1000</code></li>
          <li>KL weight <code className="font-mono text-[13px]">0.01</code> for Topic-FM</li>
          <li>Flow warmup <code className="font-mono text-[13px]">50</code> epochs</li>
          <li>Flow weight <code className="font-mono text-[13px]">0.1</code></li>
          <li>HVGs <code className="font-mono text-[13px]">3000</code>, max cells <code className="font-mono text-[13px]">3000</code></li>
        </ul>
      </section>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold text-slate-900">Code</h2>
        <p>
          <a href={SITE.github} className="text-indigo-800 underline-offset-2 hover:underline">
            github.com/PeterPonyu/PanODE-Topic
          </a>
        </p>
      </section>
    </PageShell>
  );
}
