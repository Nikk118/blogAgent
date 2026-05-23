import {
  CheckCircle2,
  Circle,
  Loader2,
  PauseCircle,
  XCircle,
} from "lucide-react"

import { cn } from "@/lib/utils"
import type { ExecutionStatus, TimelineStatus } from "@/types/blog"

const statusCopy: Record<ExecutionStatus, string> = {
  idle: "Idle",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
}

const statusClasses: Record<ExecutionStatus, string> = {
  idle: "border-white/10 bg-[#1b2a41] text-[#e4e4e4]/80",
  running:
    "border-teal-200 bg-teal-400/10 text-[#f5a31a] shadow-[0_0_28px_rgba(45,212,191,0.13)]",
  completed:
    "border-emerald-300/25 bg-emerald-400/10 text-emerald-700 shadow-[0_0_28px_rgba(52,211,153,0.12)]",
  failed:
    "border-rose-300/25 bg-rose-400/10 text-rose-200 shadow-[0_0_28px_rgba(251,113,133,0.12)]",
}

export function StatusPill({
  className,
  label,
  status,
}: {
  className?: string
  label?: string
  status: ExecutionStatus
}) {
  const Icon =
    status === "running"
      ? Loader2
      : status === "completed"
        ? CheckCircle2
        : status === "failed"
          ? XCircle
          : PauseCircle

  return (
    <span
      className={cn(
        "inline-flex h-7 items-center gap-2 rounded-full border px-3 text-[11px] font-medium",
        statusClasses[status],
        className
      )}
    >
      <Icon
        className={cn(
          "size-3.5",
          status === "running" && "animate-spin"
        )}
      />
      {label || statusCopy[status]}
    </span>
  )
}

export function TimelineStatusIcon({ status }: { status: TimelineStatus }) {
  if (status === "running") {
    return (
      <span className="relative flex size-7 items-center justify-center rounded-full border border-teal-300/30 bg-teal-300/15 text-[#f5a31a]">
        <span className="absolute size-2 rounded-full bg-teal-300 animate-live-dot" />
        <Loader2 className="size-3.5 animate-spin" />
      </span>
    )
  }

  if (status === "completed") {
    return (
      <span className="flex size-7 items-center justify-center rounded-full border border-emerald-300/30 bg-emerald-300/15 text-emerald-700">
        <CheckCircle2 className="size-3.5" />
      </span>
    )
  }

  if (status === "failed") {
    return (
      <span className="flex size-7 items-center justify-center rounded-full border border-rose-300/30 bg-rose-300/15 text-rose-200">
        <XCircle className="size-3.5" />
      </span>
    )
  }

  if (status === "skipped") {
    return (
      <span className="flex size-7 items-center justify-center rounded-full border border-amber-300/25 bg-amber-50 text-amber-700">
        <Circle className="size-3" />
      </span>
    )
  }

  return (
    <span className="flex size-7 items-center justify-center rounded-full border border-white/10 bg-[#1b2a41]/[0.03] text-[#e4e4e4]/60">
      <Circle className="size-3" />
    </span>
  )
}
