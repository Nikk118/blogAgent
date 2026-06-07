import {
  Braces,
  FileText,
  Globe2,
  Image as ImageIcon,
  Network,
  Sparkles,
} from "lucide-react"

import { getResultTitle } from "@/lib/blog-normalizers"
import type { BlogResult, ExecutionStatus, WorkspaceSession } from "@/types/blog"

export function WorkspaceHero({
  activeSession,
  draftTopic,
  evidenceCount,
  imageCount,
  result,
  sectionCount,
  status,
  wordCount,
}: {
  activeSession?: WorkspaceSession
  draftTopic: string
  evidenceCount: number
  imageCount: number
  result?: BlogResult
  sectionCount: number
  status: ExecutionStatus
  wordCount: number
}) {
  const title =
    activeSession?.result || result
      ? getResultTitle(result, activeSession?.title)
      : draftTopic || "Autonomous blog generation workspace"

  const statusLabel =
    status === "running"
      ? "Running"
      : status === "completed"
        ? "Completed"
        : status === "failed"
          ? "Failed"
          : "Idle"

  const statusDesc =
    status === "running"
      ? "PLANNING · RESEARCHING · WRITING · REDUCING"
      : status === "completed"
        ? "GRAPH EXECUTION SUCCESSFUL /// ARTIFACT READY"
        : status === "failed"
          ? "GRAPH RUN FAILED — REVIEW TERMINAL TRACE"
          : "ENTER TOPIC TO LAUNCH AUTONOMOUS GRAPH"

  const dotColor =
    status === "running"
      ? "bg-yellow-400"
      : status === "completed"
        ? "bg-[#00e676]"
        : status === "failed"
          ? "bg-red-500"
          : "bg-gray-400"

  return (
    <section className="border-[3px] border-black shadow-[8px_8px_0px_#000] bg-[#f5f0e8] overflow-hidden">
      {/* Ticker */}
      <div className="bg-black text-[#c8f135] font-mono text-[11px] font-black uppercase tracking-widest py-1.5 overflow-hidden whitespace-nowrap">
        <span className="inline-block animate-[ticker_18s_linear_infinite]">
          {`${"BLOG AGENT OS \u00a0///\u00a0 LANGGRAPH RUNTIME \u00a0///\u00a0 MULTI-AGENT PIPELINE \u00a0///\u00a0 ROUTER → RESEARCH → ORCHESTRATOR → FANOUT → REDUCER \u00a0///\u00a0 AI OPERATING SYSTEM \u00a0///\u00a0 ".repeat(3)}`}
        </span>
      </div>

      <div className="grid lg:grid-cols-[1fr_340px]">
        {/* Left */}
        <div className="relative p-7 border-r-[3px] border-black">
          {/* Hatch accent top */}
          <div
            className="absolute top-0 left-0 right-0 h-[5px]"
            style={{
              background:
                "repeating-linear-gradient(90deg,#c8f135 0,#c8f135 16px,#000 16px,#000 20px)",
            }}
          />

          {/* Badges */}
          <div className="flex flex-wrap gap-2 mt-3 mb-5">
            <span className="inline-flex items-center gap-1.5 border-[2px] border-black bg-[#ff2d78] px-2.5 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-white shadow-[3px_3px_0px_#000]">
              <Sparkles className="size-3.5 stroke-[3px]" />
              AI Operating System
            </span>
            <span className="inline-flex items-center gap-1.5 border-[2px] border-black bg-[#fce135] px-2.5 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-black shadow-[3px_3px_0px_#000]">
              <Network className="size-3.5 stroke-[2.5px]" />
              LangGraph Runtime
            </span>
          </div>

          <h1
            className="text-black uppercase leading-[0.92] tracking-[0.01em] mb-5"
            style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "clamp(36px,5vw,56px)" }}
          >
            {title}
          </h1>

          {/* Description - yellow slab */}
          <p className="border-[2px] border-black bg-[#fce135] shadow-[4px_4px_0px_#000] px-4 py-2.5 text-[13px] font-bold leading-relaxed text-black max-w-lg">
            {status === "running"
              ? "Planning, researching, writing, reducing, and preparing the final markdown artifact."
              : status === "completed"
                ? "Final artifact is ready. Inspect the plan, sources, generated visuals, markdown, and graph execution trail."
                : status === "failed"
                  ? "The latest graph run failed. Review the terminal trace and retry with adjusted instructions."
                  : "Enter a target topic to launch the autonomous blog graph."}
          </p>

          {/* Corner hatch texture */}
          <div
            className="pointer-events-none absolute bottom-0 right-0 w-28 h-28"
            style={{
              background:
                "repeating-linear-gradient(45deg,transparent 0,transparent 4px,rgba(0,0,0,0.06) 4px,rgba(0,0,0,0.06) 6px)",
            }}
          />
        </div>

        {/* Right - Metrics 2x2 */}
        <div className="grid grid-cols-2">
          <HeroMetric icon={FileText} label="Words" value={wordCount} />
          <HeroMetric icon={Braces} label="Sections" value={sectionCount} />
          <HeroMetric icon={Globe2} label="Sources" value={evidenceCount} last />
          <HeroMetric icon={ImageIcon} label="Images" value={imageCount} last />
        </div>
      </div>

      {/* Status Bar */}
      <div className="border-t-[3px] border-black flex items-center">
        <div className="flex items-center gap-2 px-4 py-2 border-r-[2px] border-black font-mono text-[10px] font-black uppercase tracking-wider">
          <span className={`w-2 h-2 border-[2px] border-black ${dotColor}`} />
          {statusLabel}
        </div>
        <span className="flex-1 text-right pr-4 font-mono text-[10px] font-black uppercase tracking-widest text-gray-500">
          {statusDesc}
        </span>
      </div>
    </section>
  )
}

function HeroMetric({
  icon: Icon,
  label,
  value,
  last = false,
}: {
  icon: typeof FileText
  label: string
  value: number
  last?: boolean
}) {
  return (
    <div
      className={`group relative flex flex-col gap-3 p-4 border-black transition-colors hover:bg-[#c8f135]
        border-b-[3px] border-r-[3px]
        [&:nth-child(even)]:border-r-0
        ${last ? "border-b-0" : ""}
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex size-8 items-center justify-center border-[2px] border-black bg-black text-[#fce135]">
          <Icon className="size-4 stroke-[2.5px]" />
        </div>
        <span className="font-mono text-[9px] font-black uppercase tracking-[0.18em] text-gray-400">
          {label}
        </span>
      </div>
      <p className="font-mono text-[44px] font-black leading-none text-black" style={{ fontFamily: "var(--font-bebas), sans-serif" }}>
        {value}
      </p>
      {/* corner accent dot */}
      <span className="absolute bottom-2 right-2 w-2.5 h-2.5 border-[2px] border-black bg-[#ff2d78]" />
    </div>
  )
}