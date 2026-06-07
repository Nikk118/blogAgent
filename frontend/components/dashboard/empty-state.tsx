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
        "flex min-h-[360px] flex-col items-center justify-center border-[3px] border-dashed border-black bg-white p-8 text-center rounded-none shadow-[4px_4px_0px_#000000]",
        className
      )}
    >
      {/* Icon Frame - High Contrast Neo Pink Stamp */}
      <div className="mb-5 flex size-12 items-center justify-center border-[2px] border-black bg-[#ff007f] text-white shadow-[3px_3px_0px_#000000] rounded-none">
        <Icon className="size-5 stroke-[2.5px]" />
      </div>

      {/* Typography Headers */}
      <h3 className="font-mono text-sm font-black uppercase tracking-wider text-black">
        {title}
      </h3>
      
      <p className="mt-2 max-w-md text-xs font-bold leading-6 text-gray-700">
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
        "bg-gray-200 animate-pulse rounded-none",
        className
      )}
    />
  )
}

export function PanelSkeleton() {
  return (
    <div className="space-y-5 border-[3px] border-black bg-white p-6 shadow-[6px_6px_0px_#000000] rounded-none">
      
      {/* Title Skeleton: Stark gray block with an aggressive border */}
      <SkeletonLine className="h-6 w-1/3 border-[2px] border-black bg-gray-200" />
      
      {/* Cards Skeletons: Adding a faint neo-pink wash to the loading cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <SkeletonLine className="h-28 border-[2px] border-black bg-[#ff007f]/10" />
        <SkeletonLine className="h-28 border-[2px] border-black bg-[#ff007f]/10" />
        <SkeletonLine className="h-28 border-[2px] border-black bg-[#ff007f]/10" />
      </div>
      
      {/* Text Block Skeletons */}
      <div className="space-y-3">
        <SkeletonLine className="h-4 w-full border-[2px] border-black bg-gray-200" />
        <SkeletonLine className="h-4 w-10/12 border-[2px] border-black bg-gray-200" />
        <SkeletonLine className="h-4 w-8/12 border-[2px] border-black bg-gray-200" />
      </div>
    </div>
  )
}