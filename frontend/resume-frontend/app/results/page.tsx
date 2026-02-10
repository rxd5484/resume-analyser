"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

type EvidenceMatch = {
  requirement: string;
  status: "covered" | "not_found";
  quote?: string | null;
  chunk_id?: number | null;
  confidence?: number;
  reason?: string;
};

type AnalyzeResponse = {
  fit_score: number;
  must_have: EvidenceMatch[];
  nice_to_have: EvidenceMatch[];
  missing_must_have: string[];
  missing_nice_to_have: string[];
};

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function LabelMono({ children }: { children: React.ReactNode }) {
  return (
    <div className="font-[var(--font-mono)] text-xs tracking-widest uppercase text-[var(--muted-foreground)]">
      {children}
    </div>
  );
}

function Rule({ thick }: { thick?: boolean }) {
  return <div className={cn("w-full border-t", thick ? "border-t-4 border-black" : "border-t border-black")} />;
}

function ScoreBlock({ score }: { score: number }) {
  const label = score >= 75 ? "Strong" : score >= 50 ? "Mixed" : "Weak";
  return (
    <div className="border border-black p-6 md:p-8">
      <LabelMono>Match score</LabelMono>
      <div className="mt-2 flex items-end gap-4">
        <div className="font-[var(--font-display)] text-6xl md:text-7xl leading-none tracking-tighter">
          {Math.round(score)}
        </div>
        <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)] pb-2">
          / 100 — {label}
        </div>
      </div>
      <div className="mt-6 border-t border-black pt-4 text-[var(--muted-foreground)]">
        “Covered” requires an exact quote from your resume.
      </div>
    </div>
  );
}

function ReqList({ title, items }: { title: string; items: EvidenceMatch[] }) {
  return (
    <div className="border border-black">
      <div className="border-b border-black p-4 md:p-5 flex items-center justify-between">
        <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
          {title}
        </div>
        <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest">
          Covered {items.filter(i => i.status === "covered").length}/{items.length}
        </div>
      </div>

      <ul className="divide-y divide-black">
        {items.map((m, idx) => (
          <li
            key={`${m.requirement}-${idx}`}
            className={cn("p-4 md:p-5 transition-colors duration-100", "hover:bg-black hover:text-white")}
          >
            <div className="flex items-start justify-between gap-6">
              <div>
                <div className="font-[var(--font-display)] text-lg md:text-xl leading-snug tracking-tight">
                  {m.requirement}
                </div>
                <div className="mt-2 font-[var(--font-mono)] text-xs uppercase tracking-widest opacity-70">
                  {m.status === "covered" ? "Covered" : "Not found"}
                  {typeof m.chunk_id === "number" ? ` · chunk ${m.chunk_id}` : ""}
                  {typeof m.confidence === "number" ? ` · ${(m.confidence * 100).toFixed(0)}%` : ""}
                </div>

                {m.status === "covered" && m.quote ? (
                  <blockquote className="mt-4 border-l-4 border-current pl-4 italic opacity-90">
                    “{m.quote}”
                  </blockquote>
                ) : (
                  m.reason ? <div className="mt-3 text-sm opacity-80">{m.reason}</div> : null
                )}
              </div>

              <div className="shrink-0 border-2 border-current px-3 py-2 font-[var(--font-mono)] text-xs uppercase tracking-widest">
                {m.status === "covered" ? "✓" : "×"}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default function ResultsPage() {
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [jd, setJd] = useState<string>("");
  const [resumeName, setResumeName] = useState<string>("");

  useEffect(() => {
    const raw = sessionStorage.getItem("last_result");
    const rawJd = sessionStorage.getItem("last_jd") || "";
    const rn = sessionStorage.getItem("last_resume_name") || "";

    setJd(rawJd);
    setResumeName(rn);

    if (raw) {
      try {
        setData(JSON.parse(raw));
      } catch {
        setData(null);
      }
    }
  }, []);

  const missingMust = useMemo(() => data?.missing_must_have || [], [data]);

  if (!data) {
    return (
      <div className="mm-texture min-h-screen">
        <div className="mx-auto max-w-6xl px-6 md:px-8 lg:px-12 py-10 md:py-14">
          <LabelMono>Results</LabelMono>
          <h1 className="mt-3 font-[var(--font-display)] tracking-tighter leading-none text-4xl md:text-6xl">
            No report found.
          </h1>
          <p className="mt-6 text-[var(--muted-foreground)] text-lg">
            Go back and run an analysis first.
          </p>

          <div className="mt-10">
            <Link
              href="/"
              className="inline-flex items-center justify-center px-8 py-4 text-sm font-medium uppercase tracking-widest border-2 border-black bg-black text-white hover:bg-white hover:text-black transition-colors duration-100"
            >
              Back to upload <span className="ml-3">←</span>
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mm-texture min-h-screen">
      <div className="mx-auto max-w-6xl px-6 md:px-8 lg:px-12 py-10 md:py-14">
        <header className="pb-10 md:pb-14">
          <div className="flex items-start justify-between gap-6">
            <div>
              <LabelMono>Report</LabelMono>
              <h1 className="mt-3 font-[var(--font-display)] tracking-tighter leading-none text-5xl md:text-7xl">
                MATCH
                <span className="italic">.</span>
                REPORT
              </h1>

              <div className="mt-6 space-y-2">
                <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
                  Resume · <span className="text-black">{resumeName || "—"}</span>
                </div>
                <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
                  JD length · <span className="text-black">{jd.trim().length}</span> chars
                </div>
              </div>
            </div>

            <div className="hidden md:block w-64">
              <Rule thick />
              <div className="mt-4 border-4 border-black w-14 h-14" />
            </div>
          </div>
        </header>

        <Rule thick />

        {/* Inverted “statement” section */}
        <section className="mt-10 bg-black text-white p-6 md:p-10 relative overflow-hidden">
          <div
            className="absolute inset-0 pointer-events-none"
            style={{
              backgroundImage:
                "repeating-linear-gradient(90deg, transparent, transparent 1px, #fff 1px, #fff 2px)",
              backgroundSize: "4px 100%",
              opacity: 0.03,
            }}
          />
          <div className="relative grid grid-cols-1 md:grid-cols-2 gap-8 items-end">
            <div>
              <LabelMono>Thesis</LabelMono>
              <div className="mt-3 font-[var(--font-display)] text-4xl md:text-5xl tracking-tighter leading-none">
                Evidence
                <span className="italic">.</span>
                Not vibes.
              </div>
              <p className="mt-5 text-white/80 leading-relaxed max-w-xl">
                We only mark a requirement “covered” if we can quote it exactly from your resume.
              </p>
            </div>
            <div className="border border-white p-6 md:p-8">
              <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-white/70">
                Fit score
              </div>
              <div className="mt-2 font-[var(--font-display)] text-7xl md:text-8xl tracking-tighter leading-none">
                {Math.round(data.fit_score)}
              </div>
            </div>
          </div>
        </section>

        <div className="mt-10 grid grid-cols-1 lg:grid-cols-3 gap-6">
          <ScoreBlock score={data.fit_score} />

          <div className="lg:col-span-2 border border-black p-6 md:p-8">
            <LabelMono>What to fix first</LabelMono>
            <div className="mt-3 font-[var(--font-display)] text-2xl md:text-3xl tracking-tight">
              Missing must-haves
            </div>
            <ul className="mt-6 space-y-3">
              {missingMust.length ? (
                missingMust.map((x, i) => (
                  <li key={i} className="border-b border-black pb-3">
                    <span className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)] mr-3">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="text-lg">{x}</span>
                  </li>
                ))
              ) : (
                <li className="text-[var(--muted-foreground)]">
                  None — your resume covers all extracted must-haves.
                </li>
              )}
            </ul>
          </div>
        </div>

        <div className="mt-10 grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ReqList title="Must have" items={data.must_have} />
          <ReqList title="Nice to have" items={data.nice_to_have} />
        </div>

        <footer className="pt-14 mt-14 border-t-4 border-black flex items-center justify-between gap-6">
          <Link
            href="/"
            className="font-[var(--font-mono)] text-xs uppercase tracking-widest border-b border-black hover:border-b-4 transition-all duration-100"
          >
            ← New analysis
          </Link>

          <button
            className="font-[var(--font-mono)] text-xs uppercase tracking-widest border-b border-black hover:border-b-4 transition-all duration-100"
            onClick={() => window.print()}
          >
            Print / Save PDF
          </button>
        </footer>
      </div>
    </div>
  );
}
