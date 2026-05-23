export type ExecutionStatus = "idle" | "running" | "completed" | "failed"

export type TimelineStatus =
  | ExecutionStatus
  | "queued"
  | "skipped"

export type BlogKind =
  | "explainer"
  | "tutorial"
  | "news_roundup"
  | "comparison"
  | "system_design"

export type LogTone =
  | "active"
  | "error"
  | "info"
  | "muted"
  | "success"
  | "warning"

export interface BlogTask {
  id: number
  title: string
  goal: string
  bullets: string[]
  target_words: number
  section_type: string
  tags: string[]
  requires_research: boolean
  require_citations: boolean
  require_code: boolean
}

export interface BlogPlan {
  blog_title: string
  audience: string
  tone: string
  blog_kind: BlogKind
  constraints: string[]
  tasks: BlogTask[]
}

export interface EvidenceItem {
  title: string
  url: string
  published_at?: string | null
  snippet?: string | null
  source?: string | null
}

export interface EvidencePack {
  evidence?: EvidenceItem[]
}

export interface ImageSpec {
  placeholder?: string
  filename: string
  alt: string
  caption?: string
  prompt?: string
  size?: "1024x1024" | "2048x1536" | "1536x1024" | string
  quality?: "low" | "medium" | "high" | string
}

export type BlogSection =
  | [number, string]
  | {
      id?: number
      title?: string
      content?: string
    }

export interface BlogResult {
  topic?: string
  mode?: string
  needs_research?: boolean
  queries?: string[]
  evidence?: Array<EvidencePack | EvidenceItem>
  plan?: BlogPlan | null
  sections?: BlogSection[]
  merged_md?: string
  md_with_placeholders?: string
  image_specs?: ImageSpec[]
  final?: string
  logs?: string[]
}

export interface BlogResponse {
  topic: string
  result: BlogResult
}

export interface WorkspaceSession {
  id: string
  title: string
  topic: string
  createdAt: string
  updatedAt: string
  status: ExecutionStatus
  result?: BlogResult
  error?: string
}

export interface TimelineNode {
  id: string
  label: string
  description: string
  status: TimelineStatus
  meta?: string
}

export interface GraphLogLine {
  stamp: string
  node: string
  message: string
  tone: LogTone
}
