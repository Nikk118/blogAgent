import type {
  BlogResult,
  EvidenceItem,
  EvidencePack,
  ExecutionStatus,
  GraphLogLine,
  ImageSpec,
  TimelineNode,
  TimelineStatus,
} from "@/types/blog"


export function extractMarkdownTitle(markdown?: string): string | null {
  if (!markdown) {
    return null
  }

  const match = markdown.match(/^#\s+(.+)$/m)
  return match?.[1]?.trim() || null
}

export function getResultTitle(
  result?: BlogResult,
  fallback = "Untitled generation"
): string {
  return (
    result?.plan?.blog_title?.trim() ||
    extractMarkdownTitle(result?.final)?.trim() ||
    fallback.trim() ||
    "Untitled generation"
  )
}

export function flattenEvidence(result?: BlogResult): EvidenceItem[] {
  const raw = result?.evidence ?? []
  const seen = new Set<string>()
  const items: EvidenceItem[] = []

  for (const entry of raw) {
    const maybePack = entry as EvidencePack

    if (Array.isArray(maybePack.evidence)) {
      for (const nested of maybePack.evidence) {
        pushEvidence(nested, items, seen)
      }
      continue
    }

    pushEvidence(entry as EvidenceItem, items, seen)
  }

  return items
}

export function getImageSpecs(result?: BlogResult): ImageSpec[] {
  const explicit = result?.image_specs ?? []

  if (explicit.length > 0) {
    return explicit
  }

  const markdown = result?.final ?? result?.md_with_placeholders ?? ""
  const matches = [...markdown.matchAll(/!\[([^\]]*)\]\(([^)]+)\)/g)]

  return matches.map((match, index) => ({
    filename: match[2] || `image-${index + 1}.png`,
    alt: match[1] || `Generated image ${index + 1}`,
    caption: match[1] || "Generated visual asset",
    prompt: "Recovered from markdown image reference.",
  }))
}

export function getSectionCount(result?: BlogResult): number {
  return result?.sections?.length ?? result?.plan?.tasks?.length ?? 0
}

export function getMarkdownWordCount(markdown?: string): number {
  if (!markdown?.trim()) {
    return 0
  }

  return markdown.trim().split(/\s+/).filter(Boolean).length
}

export function buildTimeline(
  status: ExecutionStatus,
  result?: BlogResult,
  phaseIndex = 0
): TimelineNode[] {
  const hasResearch = result?.needs_research ?? true
  const hasImages = getImageSpecs(result).length > 0
  const completed = status === "completed"

  const node = (
    id: string,
    label: string,
    description: string,
    index: number,
    meta?: string,
    skipWhenComplete = false
  ): TimelineNode => ({
    id,
    label,
    description,
    meta,
    status: resolveTimelineStatus(
      status,
      phaseIndex,
      index,
      completed && skipWhenComplete
    ),
  })

  return [
    node(
      "router",
      "Route",
      "Classify topic and choose closed, hybrid, or open-book mode.",
      0,
      result?.mode || "mode pending"
    ),
    node(
      "research",
      "Research",
      "Fetch and normalize authoritative source fragments.",
      1,
      `${result?.queries?.length ?? 0} queries`,
      !hasResearch
    ),
    node(
      "planner",
      "Plan",
      "Build the editorial outline, constraints, and section tasks.",
      2,
      `${result?.plan?.tasks?.length ?? 0} tasks`
    ),
    node(
      "workers",
      "Workers",
      "Fan out section writers and synthesize task outputs.",
      3,
      `${getSectionCount(result)} sections`
    ),
    node(
      "reducer",
      "Reducer",
      "Merge sections into a coherent markdown draft.",
      4,
      `${getMarkdownWordCount(result?.merged_md || result?.final)} words`
    ),
    node(
      "images",
      "Images",
      "Decide and place technical visual assets.",
      5,
      `${getImageSpecs(result).length} assets`,
      completed && !hasImages
    ),
    node(
      "final",
      "Final",
      "Deliver the finished blog artifact and runtime state.",
      6,
      result?.final ? "artifact ready" : "awaiting final"
    ),
  ]
}

export function buildGraphLogLines(
  status: ExecutionStatus,
  result?: BlogResult,
  topic = "Untitled topic",
  error?: string,
  phaseIndex = 0
): GraphLogLine[] {
  const lines: GraphLogLine[] = [
    {
      stamp: "T+00.0s",
      node: "INIT",
      message: `Workspace received topic payload: "${topic || "Untitled topic"}"`,
      tone: status === "idle" ? "muted" : "info",
    },
  ]

  if (status === "idle") {
    return [
      {
        stamp: "T+--.-s",
        node: "WAIT",
        message: "Graph runtime standing by for an agent instruction.",
        tone: "muted",
      },
    ]
  }

  if (status === "failed") {
    return [
      ...lines,
      {
        stamp: "T+ERR",
        node: "GRAPH",
        message: error || "Generation failed before a final artifact was produced.",
        tone: "error",
      },
    ]
  }

  const timeline = buildTimeline(status, result, phaseIndex)

  timeline.forEach((entry, index) => {
    const tone = logToneForTimelineStatus(entry.status)

    lines.push({
      stamp: `T+0${index + 1}.${(index * 7) % 10}s`,
      node: entry.id.toUpperCase(),
      message: `${entry.description} ${entry.meta ? `[${entry.meta}]` : ""}`.trim(),
      tone,
    })
  })

  if (result?.queries?.length) {
    lines.push({
      stamp: "T+08.2s",
      node: "QUERY",
      message: result.queries.join(" | "),
      tone: "info",
    })
  }

  if (result?.logs?.length) {
    result.logs.slice(-12).forEach((message, index) => {
      lines.push({
        stamp: `T+L${String(index + 1).padStart(2, "0")}`,
        node: "EVENT",
        message,
        tone: "muted",
      })
    })
  }

  if (status === "completed") {
    lines.push({
      stamp: "T+DONE",
      node: "FINAL",
      message: "Final markdown artifact committed to preview state.",
      tone: "success",
    })
  }

  return lines
}

function pushEvidence(
  item: EvidenceItem | undefined,
  items: EvidenceItem[],
  seen: Set<string>
) {
  if (!item?.url) {
    return
  }

  const key = item.url.trim()

  if (!key || seen.has(key)) {
    return
  }

  seen.add(key)
  items.push({
    title: item.title || key,
    url: key,
    published_at: item.published_at,
    snippet: item.snippet,
    source: item.source || safeHost(key),
  })
}

function safeHost(url: string): string {
  try {
    return new URL(url).host.replace(/^www\./, "")
  } catch {
    return "source"
  }
}

function resolveTimelineStatus(
  status: ExecutionStatus,
  phaseIndex: number,
  index: number,
  skipped: boolean
): TimelineStatus {
  if (skipped) {
    return "skipped"
  }

  if (status === "completed") {
    return "completed"
  }

  if (status === "failed") {
    return index < phaseIndex ? "completed" : index === phaseIndex ? "failed" : "queued"
  }

  if (status === "running") {
    return index < phaseIndex ? "completed" : index === phaseIndex ? "running" : "queued"
  }

  return "queued"
}

function logToneForTimelineStatus(status: TimelineStatus): GraphLogLine["tone"] {
  if (status === "running") {
    return "active"
  }

  if (status === "completed") {
    return "success"
  }

  if (status === "failed") {
    return "error"
  }

  if (status === "skipped") {
    return "warning"
  }

  return "muted"
}
