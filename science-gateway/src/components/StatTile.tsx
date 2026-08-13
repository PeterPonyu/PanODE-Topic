type StatTileProps = {
  value: string;
  label: string;
  note?: string;
};

export default function StatTile({ value, label, note }: StatTileProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white/80 p-5 text-center">
      <p className="font-mono text-2xl font-bold text-brand">{value}</p>
      <p className="mt-1 text-sm font-semibold text-slate-800">{label}</p>
      {note ? <p className="mt-1 text-xs text-slate-500">{note}</p> : null}
    </div>
  );
}
