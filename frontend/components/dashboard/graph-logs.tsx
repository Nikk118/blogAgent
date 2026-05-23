import { Circle, Terminal } from "lucide-react"

import { buildGraphLogLines } from "@/lib/blog-normalizers"
import { cn } from "@/lib/utils"
import type { BlogResult, ExecutionStatus, LogTone } from "@/types/blog"

export function GraphLogs({
  error,
  phaseIndex,
  result,
  status,
  topic,
}: {
  error?: string
  phaseIndex: number
  result?: BlogResult
  status: ExecutionStatus
  topic: string
}) {
  const lines = buildGraphLogLines(status, result, topic, error, phaseIndex)

  return (
    <section className="overflow-hidden rounded-2xl border border-white/10 bg-[#050609] shadow-[0_30px_120px_rgba(0,0,0,0.36)]">
      <div className="flex items-center justify-between border-b border-white/10 bg-[#1b2a41]/[0.035] px-4 py-3">
        <div className="flex items-center gap-2 text-sm font-medium text-[#ffffff]">
          <Terminal className="size-4 text-[#f5a31a]" />
          Runtime Terminal
        </div>
        <div className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-full bg-rose-400/70" />
          <span className="size-2.5 rounded-full bg-amber-300/70" />
          <span className="size-2.5 rounded-full bg-emerald-300/70" />
        </div>
      </div>
      <div className="dashboard-scrollbar relative max-h-[620px] overflow-auto p-4 font-mono text-[12px]">
        <div className="pointer-events-none absolute inset-x-0 top-0 h-16 bg-gradient-to-b from-teal-300/5 to-transparent" />
        <div className="space-y-2">
          {lines.map((line, index) => (
            <div
              className="grid grid-cols-[72px_92px_minmax(0,1fr)] gap-3 rounded-lg px-2 py-1.5 transition hover:bg-[#1b2a41]"
              key={`${line.stamp}-${line.node}-${index}`}
            >
              <span className="text-[#e4e4e4]/60">{line.stamp}</span>
              <span className={cn("inline-flex items-center gap-2", toneClass(line.tone))}>
                <Circle className="size-2 fill-current" />
                {line.node}
              </span>
              <span className="min-w-0 break-words text-[#e4e4e4]/80">
                {line.message}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function toneClass(tone: LogTone) {
  if (tone === "active") {
    return "text-[#f5a31a]"
  }

  if (tone === "success") {
    return "text-emerald-700"
  }

  if (tone === "warning") {
    return "text-amber-700"
  }

  if (tone === "error") {
    return "text-rose-200"
  }

  if (tone === "info") {
    return "text-cyan-700"
  }

  return "text-[#e4e4e4]/60"
}
