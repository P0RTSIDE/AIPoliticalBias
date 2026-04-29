import { useState } from "react";

const MIN_LEN = 20;

export default function NewsInput({ onSubmit, disabled }) {
  const [text, setText] = useState("");

  const submit = () => {
    const t = text.trim();
    if (t.length < MIN_LEN) {
      return;
    }
    onSubmit(t);
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4 shadow-lg">
      <label htmlFor="news" className="mb-2 block text-sm font-medium text-slate-300">
        News headline or article
      </label>
      <textarea
        id="news"
        rows={8}
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
        placeholder={`Paste at least ${MIN_LEN} characters…`}
        className="w-full resize-y rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 placeholder:text-slate-600 focus:border-slate-500 focus:outline-none focus:ring-1 focus:ring-slate-500 disabled:opacity-60"
      />
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={submit}
          disabled={disabled || text.trim().length < MIN_LEN}
          className="rounded-lg bg-democrat px-5 py-2.5 text-sm font-semibold text-white shadow hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Analyze"}
        </button>
        <span className="text-xs text-slate-500">
          {text.trim().length} / {MIN_LEN}+ characters
        </span>
      </div>
    </div>
  );
}
