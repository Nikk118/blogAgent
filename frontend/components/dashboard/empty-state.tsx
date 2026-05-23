import type { LucideIcon } from "lucide-react"

import { cn } from "@/lib/utils"

export function EmptyState({
  className,
  description,
  icon: Icon,
  title,
}: {
  className?: string
  description: string
  icon: LucideIcon
  title: string
}) {
  return (
    <div
      className={cn(
        "flex min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 bg-[#1b2a41]/[0.025] p-8 text-center",
        className
      )}
    >
      <div className="mb-5 flex size-12 items-center justify-center rounded-2xl border border-white/10 bg-[#1b2a41] text-[#e4e4e4]/90 shadow-2xl">
        <Icon className="size-5" />
      </div>
      <h3 className="text-sm font-medium text-[#ffffff]">{title}</h3>
      <p className="mt-2 max-w-md text-sm leading-6 text-[#e4e4e4]/60">
        {description}
      </p>
    </div>
  )
}

export function SkeletonLine({
  className,
}: {
  className?: string
}) {
  return (
    <div
      className={cn(
        "h-3 rounded-full bg-[linear-gradient(90deg,rgba(255,255,255,0.04),rgba(255,255,255,0.12),rgba(255,255,255,0.04))] animate-soft-shimmer",
        className
      )}
    />
  )
}

export function PanelSkeleton() {
  return (
    <div className="space-y-5 rounded-2xl border border-white/10 bg-[#1b2a41]/[0.025] p-6">
      <SkeletonLine className="h-5 w-1/3" />
      <div className="grid gap-4 md:grid-cols-3">
        <SkeletonLine className="h-28" />
        <SkeletonLine className="h-28" />
        <SkeletonLine className="h-28" />
      </div>
      <div className="space-y-3">
        <SkeletonLine className="w-full" />
        <SkeletonLine className="w-10/12" />
        <SkeletonLine className="w-8/12" />
      </div>
    </div>
  )
}
