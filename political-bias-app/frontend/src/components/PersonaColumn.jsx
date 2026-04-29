import ResponseCard from "./ResponseCard.jsx";

function IconDonkey({ color }) {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden>
      <path
        d="M6 18c0-4 3-8 8-9V6h2v3c5 1 8 5 8 9v2H6v-2z"
        stroke={color}
        strokeWidth="1.5"
        fill="none"
      />
      <circle cx="11" cy="17" r="1.2" fill={color} />
      <circle cx="19" cy="17" r="1.2" fill={color} />
      <path d="M10 22h10" stroke={color} strokeWidth="1.2" strokeLinecap="round" />
    </svg>
  );
}

function IconElephant({ color }) {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden>
      <path
        d="M8 14c0-4 3.5-7 8-7s8 3 8 7v6c0 2-1 3-3 3h-2l-1 4h-2l-1-3h-2l-1 3h-2l-1-4H9c-2 0-3-1-3-3v-6z"
        stroke={color}
        strokeWidth="1.5"
        fill="none"
      />
      <circle cx="12" cy="15" r="1" fill={color} />
    </svg>
  );
}

function IconScales({ color }) {
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden>
      <path d="M16 6v20" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
      <path d="M8 10h16" stroke={color} strokeWidth="1.5" />
      <path d="M10 10l-3 8h6l-3-8zM22 10l-3 8h6l-3-8z" stroke={color} strokeWidth="1.2" fill="none" />
    </svg>
  );
}

function Icon({ name, color }) {
  if (name === "elephant") return <IconElephant color={color} />;
  if (name === "scales") return <IconScales color={color} />;
  return <IconDonkey color={color} />;
}

export default function PersonaColumn({
  id,
  title,
  accentClass,
  themeColor,
  icon,
  loading,
  data,
}) {
  return (
    <section
      className={`flex flex-col rounded-xl border-2 bg-slate-900/40 p-4 shadow-lg ${accentClass}`}
      aria-labelledby={`${id}-heading`}
    >
      <div className="mb-3 flex items-center gap-2">
        <Icon name={icon} color={themeColor} />
        <h2 id={`${id}-heading`} className="text-lg font-semibold">
          {title}
        </h2>
      </div>
      <ResponseCard
        loading={loading}
        response={data?.response}
        confidence={data?.confidence ?? 0}
        inferenceSeconds={data?.inference_seconds}
        themeColor={themeColor}
      />
    </section>
  );
}
