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

// Flat, high-contrast Neo-Brutalist color mapping
const statusClasses: Record<ExecutionStatus, string> = {
  idle: "bg-white text-black",
  running: "bg-[#fce166] text-black shadow-[2px_2px_0px_#000000]",
  completed: "bg-emerald-400 text-black shadow-[2px_2px_0px_#000000]",
  failed: "bg-[#ff007f] text-white shadow-[2px_2px_0px_#000000]",
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
        "inline-flex h-7 items-center gap-1.5 border-[2px] border-black px-2.5 font-mono text-[10px] font-black uppercase tracking-wider rounded-none transition-all",
        statusClasses[status],
        className
      )}
    >
      <Icon
        className={cn(
          "size-3.5 stroke-[3px]",
          status === "running" && "animate-spin"
        )}
      />
      <span>{label || statusCopy[status]}</span>
    </span>
  )
}

export function TimelineStatusIcon({ status }: { status: TimelineStatus }) {
  // Common brutalist square container for all timeline items
  const baseBoxClass = "flex size-7 items-center justify-center border-[2px] border-black rounded-none shadow-[2px_2px_0px_#000000] transition-all"

  if (status === "running") {
    return (
      <span className={cn(baseBoxClass, "bg-[#fce166] text-black")}>
        <Loader2 className="size-3.5 animate-spin stroke-[3px]" />
      </span>
    )
  }

  if (status === "completed") {
    return (
      <span className={cn(baseBoxClass, "bg-emerald-400 text-black")}>
        <CheckCircle2 className="size-3.5 stroke-[3px]" />
      </span>
    )
  }

  if (status === "failed") {
    return (
      <span className={cn(baseBoxClass, "bg-[#ff007f] text-white")}>
        <XCircle className="size-3.5 stroke-[3px]" />
      </span>
    )
  }

  if (status === "skipped") {
    return (
      <span className={cn(baseBoxClass, "bg-orange-400 text-black")}>
        <Circle className="size-3 stroke-[3px] fill-current" />
      </span>
    )
  }

  // Pending / Idle State
  return (
    <span className={cn(baseBoxClass, "bg-white text-gray-400 border-gray-300 shadow-none")}>
      <Circle className="size-2.5 stroke-[2.5px]" />
    </span>
  )
}