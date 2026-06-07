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
    <header className="sticky top-0 z-30 flex min-h-16 items-center justify-between border-b-[3px] border-black bg-white px-4 sm:px-6 lg:px-8 rounded-none">
      <div className="flex min-w-0 items-center gap-4">
        <StatusPill status={status} />
        <div className="hidden min-w-0 items-center gap-2 text-black sm:flex">
          <Radio className="size-4 text-[#ff007f] stroke-[2.5px]" />
          <span className="truncate font-mono text-xs font-black uppercase tracking-wider bg-gray-100 border border-black/20 px-2 py-0.5 rounded-none">
            {activeSession?.id ? `session:${activeSession.id.slice(0, 8)}` : "session:standby"}
          </span>
        </div>
      </div>

      <div className="hidden items-center gap-3 lg:flex">
        <Metric icon={Layers3} label="sections" value={sectionCount} />
        <Metric icon={Search} label="sources" value={evidenceCount} />
        <Metric icon={Activity} label="words" value={wordCount} />
      </div>

      <div className="flex items-center gap-3">
        {/* Command Surface Tag */}
        <div className="hidden items-center gap-2 border-[2px] border-black bg-white px-3 py-1 font-mono text-[11px] font-black uppercase tracking-wider text-black shadow-[2px_2px_0px_#000000] sm:flex rounded-none">
          <Command className="size-3.5 stroke-[2.5px]" />
          <span>Command surface</span>
        </div>
        
        {/* CPU Engine Block Badge */}
        <div className="flex size-8 items-center justify-center border-[2px] border-black bg-[#fce166] text-black shadow-[2px_2px_0px_#000000] rounded-none">
          <Cpu className="size-4 stroke-[2.5px]" />
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
    <div className="flex items-center gap-2.5 border-[2px] border-black bg-white px-3 py-1.5 shadow-[3px_3px_0px_#000000] rounded-none transition-all">
      {/* Icon - Given heavier line weight to match the border system */}
      <Icon className="size-4 shrink-0 text-black stroke-[2.5px]" />
      
      {/* Value - Bold monospace text */}
      <span className="font-mono text-xs font-black text-black">
        {value}
      </span>
      
      {/* Label - Transformed into a clean, uppercase industrial sub-label */}
      <span className="font-mono text-[10px] font-black uppercase tracking-widest text-gray-400">
        {label}
      </span>
    </div>
  )
}