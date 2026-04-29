function SkeletonBlock() {
  return (
    <div className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/50 p-4">
      <div className="h-3 w-2/3 animate-pulse rounded bg-slate-700" />
      <div className="h-3 w-full animate-pulse rounded bg-slate-700" />
      <div className="h-3 w-5/6 animate-pulse rounded bg-slate-700" />
      <div className="h-3 w-4/5 animate-pulse rounded bg-slate-700" />
      <div className="h-3 w-full animate-pulse rounded bg-slate-700" />
    </div>
  );
}

export default function ResponseCard({ loading, response, confidence, inferenceSeconds, themeColor }) {
  if (loading) {
    return <SkeletonBlock />;
  }
  if (!response) {
    return (
      <div className="rounded-lg border border-dashed border-slate-700 bg-slate-900/30 p-6 text-center text-sm text-slate-500">
        Submit news to see this persona&apos;s analysis.
      </div>
    );
  }

  const pct = Math.round(Math.min(1, Math.max(0, confidence)) * 100);

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-4 shadow-inner">
      <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-100">{response}</p>
      <div className="mt-4 border-t border-slate-800 pt-3">
        <div className="mb-1 flex items-center justify-between text-xs text-slate-400">
          <span>Confidence</span>
          <span>{pct}%</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, backgroundColor: themeColor }}
          />
        </div>
        {inferenceSeconds != null && (
          <div className="mt-2 inline-flex rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
            {Number(inferenceSeconds).toFixed(1)}s
          </div>
        )}
      </div>
    </div>
  );
}
