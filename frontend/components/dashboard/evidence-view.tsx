import { Database, ExternalLink, Globe2, Search } from "lucide-react"

import { EmptyState, PanelSkeleton } from "@/components/dashboard/empty-state"
import { flattenEvidence } from "@/lib/blog-normalizers"
import type { BlogResult, ExecutionStatus } from "@/types/blog"

export function EvidenceView({
  result,
  status,
}: {
  result?: BlogResult
  status: ExecutionStatus
}) {
  const evidence = flattenEvidence(result)

  if (evidence.length === 0 && status === "running") {
    return <PanelSkeleton />
  }

  if (evidence.length === 0) {
    return (
      <EmptyState
        description="Evidence cards appear when the router selects hybrid or open-book mode and research returns source fragments."
        icon={Database}
        title="No evidence attached"
      />
    )
  }

return (
   <div className="grid gap-4 lg:grid-cols-2">
  {evidence.map((item, index) => (
    
      <a className="group relative overflow-hidden border-[3px] border-black bg-[#f5f0e8] p-5 transition-all shadow-[6px_6px_0px_#000] hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000]"
      href={item.url}
      key={item.url}
      rel="noreferrer"
      target="_blank"
    >
      <div className="flex items-start gap-4">

        {/* Index */}
        <div className="flex size-10 shrink-0 items-center justify-center border-[2px] border-black bg-[#ff2d78] text-lg font-black text-white shadow-[2px_2px_0px_#000]">
          {String(index + 1).padStart(2, "0")}
        </div>

        <div className="min-w-0 flex-1">
          <div className="mb-3 flex flex-wrap items-center gap-2">

            {/* Source badge */}
            <span className="inline-flex items-center gap-1 border-[2px] border-black bg-[#fce135] px-2 py-0.5 text-[12px] font-bold text-black shadow-[2px_2px_0px_#000]">
              <Globe2 className="size-3" />
              {item.source || safeHost(item.url)}
            </span>

            {/* Date badge */}
            {item.published_at && (
              <span className="border-[2px] border-black bg-black px-2 py-0.5 font-mono text-[12px] font-bold text-[#c8f135] shadow-[2px_2px_0px_#000]">
                {item.published_at}
              </span>
            )}
          </div>

          {/* Title */}
          <h3 className="line-clamp-2 text-lg font-black uppercase leading-6 tracking-tight text-black">
            {item.title}
          </h3>

          {/* Snippet */}
          {item.snippet && (
            <p className="mt-3 line-clamp-4 border-l-[4px] border-black pl-3 text-sm font-medium leading-6 text-gray-700">
              {item.snippet}
            </p>
          )}

          {/* Footer */}
          <div className="mt-4 flex items-center justify-between gap-3 text-xs font-bold text-black">
            <span className="inline-flex items-center gap-1.5 uppercase">
              <Search className="size-4" />
              normalized source
            </span>
            <ExternalLink className="size-5 text-black transition group-hover:text-[#ff2d78]" />
          </div>
        </div>
      </div>
    </a>
  ))}
</div>
  )
}

function safeHost(url: string) {
  try {
    return new URL(url).host.replace(/^www\./, "")
  } catch {
    return "source"
  }
}
