import {
  Braces,
  FileText,
  Globe2,
  Image as ImageIcon,
  Network,
  Sparkles,
} from "lucide-react"

import { getResultTitle } from "@/lib/blog-normalizers"
import { cn } from "@/lib/utils"
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

  return (
    <section className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/[0.035] p-5 shadow-[0_30px_120px_rgba(0,0,0,0.38)] sm:p-7">
      <div className="absolute inset-0 runtime-grid opacity-[0.22]" />
      <div className="absolute -right-24 -top-24 size-72 rounded-full bg-teal-300/12 blur-3xl animate-orbit-pulse" />
      <div className="absolute -bottom-28 left-1/4 size-72 rounded-full bg-fuchsia-300/10 blur-3xl animate-orbit-pulse [animation-delay:1.8s]" />
      <div className="relative grid gap-7 lg:grid-cols-[minmax(0,1fr)_420px]">
        <div className="min-w-0">
          <div className="mb-5 flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-full border border-teal-300/20 bg-teal-300/10 px-3 py-1 text-xs font-medium text-teal-100">
              <Sparkles className="size-3.5" />
              AI operating system
            </span>
            <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-black/20 px-3 py-1 font-mono text-[11px] text-zinc-400">
              <Network className="size-3.5" />
              LangGraph runtime
            </span>
          </div>

          <h1 className="max-w-4xl text-balance text-3xl font-semibold leading-tight text-zinc-50 sm:text-4xl lg:text-5xl">
            {title}
          </h1>

          <p className="mt-4 max-w-2xl text-sm leading-6 text-zinc-400 sm:text-base">
            {status === "running"
              ? "Planning, researching, writing, reducing, and preparing the final markdown artifact."
              : status === "completed"
                ? "Final artifact is ready. Inspect the plan, sources, generated visuals, markdown, and graph execution trail."
                : status === "failed"
                  ? "The latest graph run failed. Review the terminal trace and retry with adjusted instructions."
                  : "Enter a target topic to launch the autonomous blog graph."}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <HeroMetric
            accent="text-cyan-200"
            icon={FileText}
            label="Words"
            value={wordCount}
          />
          <HeroMetric
            accent="text-emerald-200"
            icon={Braces}
            label="Sections"
            value={sectionCount}
          />
          <HeroMetric
            accent="text-amber-200"
            icon={Globe2}
            label="Sources"
            value={evidenceCount}
          />
          <HeroMetric
            accent="text-fuchsia-200"
            icon={ImageIcon}
            label="Images"
            value={imageCount}
          />
        </div>
      </div>
    </section>
  )
}

function HeroMetric({
  accent,
  icon: Icon,
  label,
  value,
}: {
  accent: string
  icon: typeof FileText
  label: string
  value: number
}) {
  return (
    <div className="group rounded-2xl border border-white/10 bg-black/24 p-4 transition duration-300 hover:-translate-y-0.5 hover:border-white/15 hover:bg-white/[0.06]">
      <div className="flex items-center justify-between">
        <span
          className={cn(
            "flex size-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04]",
            accent
          )}
        >
          <Icon className="size-4" />
        </span>
        <span className="font-mono text-[11px] uppercase text-zinc-600">
          {label}
        </span>
      </div>
      <p className="mt-5 font-mono text-2xl font-semibold text-zinc-50">
        {value}
      </p>
    </div>
  )
}
