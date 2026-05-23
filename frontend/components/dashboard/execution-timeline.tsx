import { Workflow } from "lucide-react"

import { TimelineStatusIcon } from "@/components/dashboard/status-pill"
import { cn } from "@/lib/utils"
import type { TimelineNode } from "@/types/blog"

export function ExecutionTimeline({ nodes }: { nodes: TimelineNode[] }) {
  return (
    <section className="glass-panel rounded-2xl p-4">
      <div className="mb-5 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-zinc-100">Execution Graph</p>
          <p className="mt-1 text-xs text-zinc-500">LangGraph node timeline</p>
        </div>
        <div className="flex size-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] text-teal-200">
          <Workflow className="size-4" />
        </div>
      </div>

      <div className="relative space-y-1">
        <div className="absolute bottom-7 left-3.5 top-7 w-px bg-gradient-to-b from-teal-300/40 via-white/10 to-transparent" />
        {nodes.map((node) => (
          <div className="relative flex gap-3 py-2" key={node.id}>
            <TimelineStatusIcon status={node.status} />
            <div
              className={cn(
                "min-w-0 flex-1 rounded-xl border p-3 transition duration-300",
                node.status === "running" &&
                  "border-teal-300/20 bg-teal-300/10",
                node.status === "completed" &&
                  "border-emerald-300/15 bg-emerald-300/[0.06]",
                node.status === "failed" &&
                  "border-rose-300/20 bg-rose-300/10",
                node.status === "queued" &&
                  "border-white/8 bg-white/[0.025]",
                node.status === "skipped" &&
                  "border-amber-300/15 bg-amber-300/[0.05]"
              )}
            >
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-medium text-zinc-100">
                  {node.label}
                </h3>
                <span className="rounded-full border border-white/10 px-2 py-0.5 font-mono text-[10px] uppercase text-zinc-500">
                  {node.status}
                </span>
              </div>
              <p className="mt-1 text-xs leading-5 text-zinc-500">
                {node.description}
              </p>
              {node.meta ? (
                <p className="mt-2 truncate font-mono text-[11px] text-zinc-400">
                  {node.meta}
                </p>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
