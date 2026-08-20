import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function ResultsPage() {
  return (
    <PageShell title="Repository layout" kicker="Public directories">
      <p>
        This site does not host manuscript figures or evaluation tables. The public repository
        layout is:
      </p>
      <pre className="overflow-x-auto rounded-xl border border-slate-200 bg-slate-50 p-4 font-mono text-[13px] leading-6 text-slate-800">
        <code>{`PanODE-Topic/
├── models/        Topic-FM, Topic, and Pure-VAE implementations
├── benchmarks/    Training, evaluation, and validation runners
├── eval_lib/      Baseline wrappers and evaluation utilities
├── experiments/   Comparison and analysis workflows
├── refined_figures/
├── scripts/       Maintenance helpers
├── utils/         Shared training and data helpers
├── src/           Visualization helpers
└── vcd/           Visual consistency diagnostics`}</code>
      </pre>
      <p>
        Source:{' '}
        <a href={SITE.github} className="text-indigo-800 underline-offset-2 hover:underline">
          github.com/PeterPonyu/PanODE-Topic
        </a>
        .
      </p>
    </PageShell>
  );
}
