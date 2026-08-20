import PageShell from '@/components/PageShell';

export default function EvidencePage() {
  return (
    <PageShell title="Model families" kicker="What is implemented">
      <p>
        Topic-FM models use a Dirichlet or logistic-normal topic prior and optional flow-matching
        refinement. Pure-VAE models are Gaussian-prior counterparts without flow matching.
        Topic-FM-GAT is optional and needs graph dependencies.
      </p>
      <ul className="list-disc space-y-2 pl-5">
        <li>Topic-FM-Base, Topic-FM-Transformer, Topic-FM-Contrastive, Topic-FM-GAT</li>
        <li>Pure-VAE, Pure-Transformer-VAE, Pure-Contrastive-VAE</li>
      </ul>
      <p>
        This page does not publish evaluation tables or manuscript figures. Use the repository if
        you want the implementations themselves.
      </p>
    </PageShell>
  );
}
