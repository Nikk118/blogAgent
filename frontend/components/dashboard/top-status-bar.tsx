import {
  Activity,
  Command,
  Cpu,
  Layers3,
  Radio,
  Search,
} from "lucide-react"

import { StatusPill } from "@/components/dashboard/status-pill"
import type { ExecutionStatus, WorkspaceSession } from "@/types/blog"

export function TopStatusBar({
  activeSession,
  evidenceCount,
  sectionCount,
  status,
  wordCount,
}: {
  activeSession?: WorkspaceSession
  evidenceCount: number
  sectionCount: number
  status: ExecutionStatus
  wordCount: number
}) {
  return (
    <header className="sticky top-0 z-30 flex min-h-16 items-center justify-between border-b border-white/10 bg-[#030407]/78 px-4 backdrop-blur-2xl sm:px-6 lg:px-8">
      <div className="flex min-w-0 items-center gap-3">
        <StatusPill status={status} />
        <div className="hidden min-w-0 items-center gap-2 text-sm text-zinc-500 sm:flex">
          <Radio className="size-4 text-teal-200/80" />
          <span className="truncate font-mono text-[12px]">
            {activeSession?.id ? `session:${activeSession.id.slice(0, 8)}` : "session:standby"}
          </span>
        </div>
      </div>

      <div className="hidden items-center gap-2 lg:flex">
        <Metric icon={Layers3} label="sections" value={sectionCount} />
        <Metric icon={Search} label="sources" value={evidenceCount} />
        <Metric icon={Activity} label="words" value={wordCount} />
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden items-center gap-2 rounded-full border border-white/10 bg-white/[0.035] px-3 py-1.5 text-[11px] text-zinc-500 sm:flex">
          <Command className="size-3.5" />
          <span>Command surface</span>
        </div>
        <div className="flex size-8 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-zinc-400">
          <Cpu className="size-4" />
        </div>
      </div>
    </header>
  )
}

function Metric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Layers3
  label: string
  value: number
}) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.035] px-3 py-1.5">
      <Icon className="size-3.5 text-zinc-400" />
      <span className="font-mono text-[11px] text-zinc-300">{value}</span>
      <span className="text-[11px] text-zinc-600">{label}</span>
    </div>
  )
}
