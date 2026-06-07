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
<section className="overflow-hidden border-[3px] border-black bg-[#f5f0e8] shadow-[6px_6px_0px_#000]">

  {/* Header */}
  <div className="flex items-center justify-between border-b-[3px] border-black bg-[#ff2d78] px-4 py-3 text-white">
    <div className="flex items-center gap-2 font-mono text-xs font-black uppercase tracking-widest">
      <Terminal className="size-4 stroke-[3px]" />
      Runtime Terminal
    </div>

    {/* Window dots */}
    <div className="flex items-center gap-1.5">
      <span className="size-3 border-[2px] border-black bg-white" />
      <span className="size-3 border-[2px] border-black bg-[#fce135]" />
      <span className="size-3 border-[2px] border-black bg-black" />
    </div>
  </div>

  {/* Terminal body */}
  <div className="dashboard-scrollbar relative max-h-[620px] overflow-auto bg-[#f5f0e8] p-4 font-mono text-xs text-black">
    <div className="space-y-0.5">
      {lines.map((line, index) => (
        <div
          className="grid grid-cols-[72px_100px_minmax(0,1fr)] gap-3 border-b border-black/10 px-2 py-2 font-bold transition-colors hover:bg-[#fce135]/30"
          key={`${line.stamp}-${line.node}-${index}`}
        >
          <span className="font-medium text-gray-400">{line.stamp}</span>

          <span className={cn("inline-flex items-center gap-1.5 font-black uppercase tracking-wider text-[11px]", toneClass(line.tone))}>
            <Circle className="size-1.5 fill-current stroke-current" />
            {line.node}
          </span>

          <span className="min-w-0 break-words font-medium text-black">
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
    return "text-amber-600"
  }

  if (tone === "success") {
    return "text-emerald-600"
  }

  if (tone === "warning") {
    return "text-orange-600"
  }

  if (tone === "error") {
    return "text-[#ff007f]" // Neo Pink acts as our loud, high-contrast error indicator on light sheets
  }

  if (tone === "info") {
    return "text-sky-600"
  }

  return "text-gray-500"
}