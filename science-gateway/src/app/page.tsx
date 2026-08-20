import RouteCards from '@/components/RouteCards';
import { SITE } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-12 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-indigo-800">
        {SITE.kicker}
      </p>
      <h1 className="mt-3 font-mono text-3xl font-bold tracking-tight text-slate-900 sm:text-4xl">
        {SITE.title}
      </h1>
      <p className="mt-5 max-w-2xl text-base leading-7 text-slate-700">{SITE.lead}</p>

      <section className="mt-10 rounded-xl border border-indigo-100 bg-indigo-50/40 p-5">
        <h2 className="font-mono text-sm font-semibold text-indigo-900">What this code implements</h2>
        <p className="mt-2 text-sm leading-6 text-slate-700">{SITE.physicalObject}</p>
        <p className="mt-3 text-sm leading-6 text-slate-700">{SITE.primaryClaim}</p>
      </section>

      <section className="mt-10">
        <h2 className="font-mono text-sm font-semibold text-slate-900">Model families</h2>
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm text-slate-700">
            <thead>
              <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 pr-4 font-medium">Encoder</th>
                <th className="py-2 pr-4 font-medium">Prior</th>
                <th className="py-2 font-medium">Flow matching</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[13px]">
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Topic-FM-Base</td>
                <td className="py-2 pr-4">MLP</td>
                <td className="py-2 pr-4">Dirichlet / logistic-normal</td>
                <td className="py-2">Yes</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Topic-FM-Transformer</td>
                <td className="py-2 pr-4">Self-attention</td>
                <td className="py-2 pr-4">Dirichlet / logistic-normal</td>
                <td className="py-2">Yes</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Topic-FM-Contrastive</td>
                <td className="py-2 pr-4">MLP + MoCo</td>
                <td className="py-2 pr-4">Dirichlet / logistic-normal</td>
                <td className="py-2">Yes</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Topic-FM-GAT</td>
                <td className="py-2 pr-4">GAT over kNN</td>
                <td className="py-2 pr-4">Dirichlet / logistic-normal</td>
                <td className="py-2">Yes (optional)</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Pure-VAE</td>
                <td className="py-2 pr-4">MLP</td>
                <td className="py-2 pr-4">Gaussian</td>
                <td className="py-2">No</td>
              </tr>
              <tr className="border-b border-slate-100">
                <td className="py-2 pr-4">Pure-Transformer-VAE</td>
                <td className="py-2 pr-4">Self-attention</td>
                <td className="py-2 pr-4">Gaussian</td>
                <td className="py-2">No</td>
              </tr>
              <tr>
                <td className="py-2 pr-4">Pure-Contrastive-VAE</td>
                <td className="py-2 pr-4">MLP + MoCo</td>
                <td className="py-2 pr-4">Gaussian</td>
                <td className="py-2">No</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="font-mono text-sm font-semibold text-slate-900">Run</h2>
        <pre className="mt-3 overflow-x-auto rounded-xl border border-slate-200 bg-slate-50 p-4 text-[13px] leading-6 text-slate-800">
          <code>{`python benchmarks/runners/benchmark_base.py --series topic --data-path <dataset-path>
python benchmarks/runners/benchmark_base.py --models Topic-FM-Transformer Pure-VAE --data-path <dataset-path>
python benchmarks/runners/benchmark_crossdata.py --datasets <dataset-key> <dataset-key>`}</code>
        </pre>
        <p className="mt-3 text-sm leading-6 text-slate-600">
          Pass a local dataset with <code className="font-mono text-[13px]">--data-path</code> for
          single-dataset runs. Cross-dataset runs need registry keys aligned on the machine that
          executes them.
        </p>
      </section>

      <section className="mt-10">
        <h2 className="mb-3 font-mono text-sm font-semibold text-slate-900">Pages</h2>
        <RouteCards />
      </section>
    </div>
  );
}
