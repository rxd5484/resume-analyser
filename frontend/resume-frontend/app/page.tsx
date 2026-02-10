"use client";

import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

// tiny clsx
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
  return (
    <div
      className={cn(
        "w-full border-t",
        thick ? "border-t-4 border-black" : "border-t border-black"
      )}
    />
  );
}

function Button({
  children,
  variant = "primary",
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "outline";
}) {
  const base =
    "inline-flex items-center justify-center px-8 py-4 text-sm font-medium uppercase tracking-widest border-2 transition-colors duration-100 focus-visible:outline focus-visible:outline-3 focus-visible:outline-black focus-visible:outline-offset-3 disabled:opacity-40 disabled:cursor-not-allowed";
  const styles =
    variant === "primary"
      ? "bg-black text-white border-black hover:bg-white hover:text-black"
      : "bg-white text-black border-black hover:bg-black hover:text-white";

  return (
    <button {...props} className={cn(base, styles, props.className)}>
      {children}
      <span className="ml-3">→</span>
    </button>
  );
}

function Card({
  title,
  kicker,
  children,
}: {
  title: string;
  kicker: string;
  children: React.ReactNode;
}) {
  return (
    <section className="border border-black p-6 md:p-8">
      <LabelMono>{kicker}</LabelMono>
      <h2 className="mt-2 font-[var(--font-display)] text-2xl md:text-3xl tracking-tight">
        {title}
      </h2>
      <div className="mt-6">{children}</div>
    </section>
  );
}

type Phase = "idle" | "uploading" | "analyzing";

function formatErr(text: string) {
  const t = (text || "").trim();
  if (!t) return "Request failed.";
  // try to pull FastAPI {"detail": "..."} out of plain text
  try {
    const j = JSON.parse(t);
    if (j?.detail) return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
  } catch {}
  // otherwise keep it short
  return t.length > 400 ? t.slice(0, 400) + "…" : t;
}

export default function UploadPage() {
  const router = useRouter();
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jdText, setJdText] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);

  const base = useMemo(() => API_BASE.replace(/\/$/, ""), []);

  const canSubmit = useMemo(() => {
    return (
      !!resumeFile &&
      jdText.trim().length >= 80 &&
      !!API_BASE &&
      phase === "idle"
    );
  }, [resumeFile, jdText, phase]);

  async function onAnalyze() {
    setError(null);

    // Helpful log to verify env is loaded on Vercel
    // eslint-disable-next-line no-console
    console.log("API_BASE", process.env.NEXT_PUBLIC_API_BASE_URL);

    if (!API_BASE) {
      setError(
        "Missing NEXT_PUBLIC_API_BASE_URL. Add it in Vercel → Project → Settings → Environment Variables."
      );
      return;
    }
    if (!resumeFile) {
      setError("Upload a PDF or DOCX resume.");
      return;
    }
    if (jdText.trim().length < 80) {
      setError("Paste more of the job description (at least ~80 chars).");
      return;
    }

    try {
      // 1) Upload resume -> get resume_id
      setPhase("uploading");
      const up = new FormData();
      up.append("resume", resumeFile);

      const upRes = await fetch(`${base}/upload`, {
        method: "POST",
        body: up,
      });

      if (!upRes.ok) throw new Error(formatErr(await upRes.text()));

      const upJson = await upRes.json();
      const resume_id: string | undefined = upJson?.resume_id;

      if (!resume_id) {
        throw new Error("Upload succeeded but resume_id missing in response.");
      }

      // 2) Analyze using resume_id + jd_text
      setPhase("analyzing");
      const fd = new FormData();
      fd.append("resume_id", resume_id); // MUST match backend Form(...)
      fd.append("jd_text", jdText); // MUST match backend Form(...)

      const res = await fetch(`${base}/analyze`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) throw new Error(formatErr(await res.text()));

      const json = await res.json();

      // demo: store result locally, then navigate
      sessionStorage.setItem("last_result", JSON.stringify(json));
      sessionStorage.setItem("last_jd", jdText);
      sessionStorage.setItem("last_resume_name", resumeFile.name);

      router.push("/results");
    } catch (e: any) {
      setError(e?.message || "Something broke.");
    } finally {
      setPhase("idle");
    }
  }

  const buttonLabel =
    phase === "uploading" ? "Uploading" : phase === "analyzing" ? "Analyzing" : "Analyze";

  return (
    <div className="mm-texture min-h-screen">
      <div className="mx-auto max-w-6xl px-6 md:px-8 lg:px-12 py-10 md:py-14">
        <header className="pb-10 md:pb-14">
          <LabelMono>Evidence-based matching</LabelMono>
          <h1 className="mt-3 font-[var(--font-display)] tracking-tighter leading-none text-5xl md:text-7xl lg:text-8xl">
            THE <span className="italic">MATCH</span>
          </h1>
          <p className="mt-6 max-w-2xl text-lg md:text-xl leading-relaxed text-[var(--muted-foreground)]">
            Upload a resume + paste a job description. We generate a match report with exact quotes as proof.
          </p>

          <div className="mt-8 flex items-center gap-6">
            <div className="w-12 h-12 border-4 border-black" />
            <div className="flex-1">
              <Rule thick />
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card title="Upload resume" kicker="Step 01">
            <div className="space-y-4">
              <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
                PDF or DOCX
              </div>

              <label className="block border-2 border-black p-5 cursor-pointer hover:bg-black hover:text-white transition-colors duration-100">
                <input
                  type="file"
                  accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  className="hidden"
                  onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
                />
                <div className="font-[var(--font-display)] text-xl tracking-tight">
                  {resumeFile ? resumeFile.name : "Choose file"}
                </div>
                <div className="mt-2 font-[var(--font-mono)] text-xs uppercase tracking-widest opacity-70">
                  {resumeFile ? `${Math.round(resumeFile.size / 1024)} KB` : "No file selected"}
                </div>
              </label>

              <div className="text-sm text-[var(--muted-foreground)] leading-relaxed">
                Tip: text-based PDFs extract best. If your PDF is scanned, results may be weaker.
              </div>
            </div>
          </Card>

          <Card title="Paste job description" kicker="Step 02">
            <div className="space-y-4">
              <textarea
                value={jdText}
                onChange={(e) => setJdText(e.target.value)}
                placeholder="Paste the full job description here…"
                className={cn(
                  "w-full min-h-[220px] md:min-h-[260px] resize-y",
                  "bg-white text-black placeholder:text-[var(--muted-foreground)] placeholder:italic",
                  "border-2 border-black p-4",
                  "focus:outline-none focus-visible:outline-none focus:border-b-4"
                )}
              />

              <div className="flex items-center justify-between gap-4">
                <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
                  {jdText.trim().length} chars
                </div>

                <Button onClick={onAnalyze} disabled={!canSubmit}>
                  {buttonLabel}
                </Button>
              </div>

              {!API_BASE ? (
                <div className="border border-black p-4 text-sm">
                  <LabelMono>Config</LabelMono>
                  <div className="mt-2 text-[var(--muted-foreground)]">
                    Add{" "}
                    <code className="font-[var(--font-mono)]">NEXT_PUBLIC_API_BASE_URL</code>{" "}
                    in Vercel → <span className="font-[var(--font-mono)]">Settings → Environment Variables</span>
                    <div className="mt-2 font-[var(--font-mono)] text-xs">
                      Example: https://resume-analyser-production.up.railway.app
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </Card>
        </div>

        {error ? (
          <div className="mt-8 border-4 border-black p-6 bg-black text-white">
            <LabelMono>Error</LabelMono>
            <div className="mt-3 font-[var(--font-display)] text-2xl md:text-3xl tracking-tight">
              {error}
            </div>
          </div>
        ) : null}

        <footer className="pt-14 mt-14 border-t-4 border-black flex items-center justify-between gap-6">
          <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
            Minimalist Monochrome · Demo
          </div>
          <div className="font-[var(--font-mono)] text-xs uppercase tracking-widest text-[var(--muted-foreground)]">
            Two-page report
          </div>
        </footer>
      </div>
    </div>
  );
}
