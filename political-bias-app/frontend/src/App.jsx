import { useState } from "react";
import axios from "axios";
import NewsInput from "./components/NewsInput.jsx";
import PersonaColumn from "./components/PersonaColumn.jsx";

const API_BASE = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const INITIAL = {
  democrat: null,
  republican: null,
  centrist: null,
};

export default function App() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(INITIAL);

  const handleAnalyze = async (text) => {
    setError(null);
    setResults(INITIAL);
    setLoading(true);
    const started = performance.now();
    try {
      const res = await axios.post(
        `${API_BASE}/analyze`,
        { news: text },
        { headers: { "Content-Type": "application/json" }, validateStatus: () => true }
      );
      if (res.status !== 200) {
        const detail =
          res.data?.detail ||
          (typeof res.data === "string" ? res.data : JSON.stringify(res.data));
        setError(`Request failed (${res.status}): ${detail || "Unknown error"}`);
        return;
      }
      setResults({
        democrat: res.data.democrat,
        republican: res.data.republican,
        centrist: res.data.centrist,
      });
    } catch (e) {
      setError(e?.message || "Network error — is the API running?");
    } finally {
      setLoading(false);
      void started;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/80 px-4 py-4 backdrop-blur">
        <h1 className="text-xl font-semibold tracking-tight md:text-2xl">
          AI political bias — three personas
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Compare how Democrat, Republican, and centrist-tuned models interpret the same news
          (CPU inference, local adapters only).
        </p>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-4">
        <div
          className="mb-4 rounded-lg border border-amber-700/50 bg-amber-950/40 px-4 py-3 text-sm text-amber-100"
          role="status"
        >
          Running on CPU — each analysis takes roughly 30–90 seconds per persona. Please be
          patient.
        </div>

        {error && (
          <div
            className="mb-4 rounded-lg border border-red-600/60 bg-red-950/50 px-4 py-3 text-sm text-red-100"
            role="alert"
          >
            {error}
          </div>
        )}

        <NewsInput onSubmit={handleAnalyze} disabled={loading} />

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <PersonaColumn
            id="democrat"
            title="Democrat"
            accentClass="border-democrat text-democrat"
            themeColor="#1a56db"
            icon="donkey"
            loading={loading}
            data={results.democrat}
          />
          <PersonaColumn
            id="republican"
            title="Republican"
            accentClass="border-republican text-republican"
            themeColor="#e02424"
            icon="elephant"
            loading={loading}
            data={results.republican}
          />
          <PersonaColumn
            id="centrist"
            title="Centrist"
            accentClass="border-centrist text-centrist"
            themeColor="#7e3af2"
            icon="scales"
            loading={loading}
            data={results.centrist}
          />
        </div>
      </div>
    </div>
  );
}
