import { Workflow } from "lucide-react"

import { TimelineStatusIcon } from "@/components/dashboard/status-pill"
import { cn } from "@/lib/utils"
import type { TimelineNode } from "@/types/blog"

export function ExecutionTimeline({ nodes }: { nodes: TimelineNode[] }) {
return (
    <section className="border-[3px] border-black bg-white p-5 shadow-[6px_6px_0px_#000000]">
      
      {/* Header section with a solid bottom border splitting it */}
      <div className="mb-6 flex items-center justify-between border-b-[3px] border-black pb-4">
        <div>
          <p className="text-lg font-black uppercase tracking-tight text-black">Execution Graph</p>
          <p className="mt-0.5 font-mono text-xs font-bold text-gray-600">LangGraph node timeline</p>
        </div>
        {/* Highlighter Yellow Icon Box */}
        <div className="flex size-10 items-center justify-center border-[2px] border-black bg-[#fce166] text-black shadow-[2px_2px_0px_#000000]">
          <Workflow className="size-5" />
        </div>
      </div>

      <div className="relative space-y-3">
        {/* Thick, solid structural timeline track (Replaced the soft gradient line) */}
        <div className="absolute bottom-8 left-[15px] top-8 w-[3px] bg-black" />
        
        {nodes.map((node) => (
          <div className="relative flex gap-4 py-1" key={node.id}>
            
            {/* Wrapper to make sure the external status icon layers correctly over the line */}
            <div className="relative z-10 flex shrink-0 items-center justify-center">
              <TimelineStatusIcon status={node.status} />
            </div>
            
            {/* Node Card Box */}
            <div
              className={cn(
                "min-w-0 flex-1 border-[2px] border-black p-4 transition-all duration-200",
                
                // Running Node: FLASHING NEO PINK with stark white text
                node.status === "running" &&
                  "bg-[#ff007f] text-white shadow-[4px_4px_0px_#000000] font-bold animate-[pulse_1.5s_infinite]",
                
                // Completed Node: Flat Mint Green
                node.status === "completed" &&
                  "bg-[#cff0e0] text-black shadow-[4px_4px_0px_#000000]",
                
                // Failed Node: Raw High-Contrast Red
                node.status === "failed" &&
                  "bg-[#ff4d4d] text-white shadow-[4px_4px_0px_#000000]",
                
                // Queued Node: Clean Base White
                node.status === "queued" &&
                  "bg-white text-black shadow-[2px_2px_0px_#000000]",
                
                // Skipped Node: Receded structural Gray
                node.status === "skipped" &&
                  "bg-gray-100 text-gray-400 border-gray-400 shadow-none"
              )}
            >
              <div className="flex items-center justify-between gap-3">
                <h3 className={cn(
                  "text-sm font-black uppercase tracking-tight",
                  node.status === "running" || node.status === "failed" ? "text-white" : "text-black"
                )}>
                  {node.label}
                </h3>
                
                {/* Node Status Badge */}
                <span className={cn(
                  "border-[2px] border-black px-2 py-0.5 font-mono text-[10px] font-black uppercase shadow-[1px_1px_0px_#000000]",
                  node.status === "running" || node.status === "failed" ? "bg-white text-black" : "bg-black text-white"
                )}>
                  {node.status}
                </span>
              </div>
              
              <p className={cn(
                "mt-2 text-xs font-medium leading-5",
                node.status === "running" || node.status === "failed" ? "text-white/90" : "text-gray-700"
              )}>
                {node.description}
              </p>
              
              {node.meta ? (
                <p className={cn(
                  "mt-3 truncate font-mono text-[11px] border-t border-black/10 pt-2",
                  node.status === "running" || node.status === "failed" ? "text-white/80" : "text-gray-500"
                )}>
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
