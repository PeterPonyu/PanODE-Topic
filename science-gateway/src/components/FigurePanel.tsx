import Image from 'next/image';
import { assetPath } from '@/lib/assets';

type FigurePanelProps = {
  src: string;
  alt: string;
  caption: string;
  kicker?: string;
};

export default function FigurePanel({ src, alt, caption, kicker }: FigurePanelProps) {
  const resolvedSrc = assetPath(src);
  return (
    <figure className="overflow-hidden rounded-2xl border border-slate-200 bg-white/80">
      <div className="relative w-full">
        <Image
          src={resolvedSrc}
          alt={alt}
          width={1200}
          height={900}
          className="h-auto w-full max-w-full"
          unoptimized
        />
      </div>
      <figcaption className="space-y-2 p-5 text-sm text-slate-600">
        {kicker ? (
          <p className="text-xs font-semibold uppercase tracking-wide text-teal-700">{kicker}</p>
        ) : null}
        <p>{caption}</p>
      </figcaption>
    </figure>
  );
}
