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
        <a
          className="group relative overflow-hidden rounded-2xl border border-white/10 bg-[#1b2a41]/[0.035] p-4 transition duration-300 hover:-translate-y-1 hover:border-teal-200 hover:bg-[#1b2a41]/[0.06] hover:shadow-[0_24px_90px_rgba(20,184,166,0.09)]"
          href={item.url}
          key={item.url}
          rel="noreferrer"
          target="_blank"
        >
          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-teal-200/60 to-transparent opacity-0 transition group-hover:opacity-100" />
          <div className="flex items-start gap-3">
            <div className="flex size-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-[#1b2a41]/60 text-sm font-semibold text-teal-800">
              {String(index + 1).padStart(2, "0")}
            </div>
            <div className="min-w-0 flex-1">
              <div className="mb-2 flex flex-wrap items-center gap-2">
                <span className="inline-flex items-center gap-1 rounded-full border border-white/10 bg-[#1b2a41]/60 px-2 py-0.5 text-[11px] text-[#e4e4e4]/80">
                  <Globe2 className="size-3" />
                  {item.source || safeHost(item.url)}
                </span>
                {item.published_at ? (
                  <span className="rounded-full border border-white/10 bg-[#1b2a41]/60 px-2 py-0.5 font-mono text-[11px] text-[#e4e4e4]/60">
                    {item.published_at}
                  </span>
                ) : null}
              </div>
              <h3 className="line-clamp-2 text-base font-semibold leading-6 text-[#ffffff]">
                {item.title}
              </h3>
              {item.snippet ? (
                <p className="mt-3 line-clamp-4 text-sm leading-6 text-[#e4e4e4]/60">
                  {item.snippet}
                </p>
              ) : null}
              <div className="mt-4 flex items-center justify-between gap-3 text-xs text-[#e4e4e4]/60">
                <span className="inline-flex items-center gap-1.5">
                  <Search className="size-3.5" />
                  normalized source
                </span>
                <ExternalLink className="size-4 text-[#e4e4e4]/60 transition group-hover:text-[#f5a31a]" />
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
