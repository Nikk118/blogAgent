import {
  Database,
  FileText,
  Image as ImageIcon,
  LayoutDashboard,
  Terminal,
} from "lucide-react"

import { EvidenceView } from "@/components/dashboard/evidence-view"
import { GraphLogs } from "@/components/dashboard/graph-logs"
import { ImagesView } from "@/components/dashboard/images-view"
import { MarkdownPreview } from "@/components/dashboard/markdown-renderer"
import { PlanView } from "@/components/dashboard/plan-view"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { BlogResult, ExecutionStatus } from "@/types/blog"


// Build image map from result

export function WorkspaceTabs({
  error,
  phaseIndex,
  result,
  status,
  topic,
}: {
  error?: string
  phaseIndex: number
  result?: BlogResult
  status: ExecutionStatus
  topic: string
}) {

  // Build image map from result


const imageMap = (
  result?.generated_images ?? []
).reduce(
  (acc, img) => {
    if (img.image_data) {
      acc[img.filename] = img.image_data
    }
    return acc
  },
  {} as Record<string, string>
)



  return (
    <Tabs
      defaultValue="markdown"
      className="flex w-full min-w-0 flex-col gap-5"
    >
      <TabsList
        className="dashboard-scrollbar flex h-auto w-full shrink-0 justify-start gap-2 overflow-x-auto rounded-2xl border border-white/10 bg-white/[0.035] p-1.5"
        variant="line"
      >
        <TabTrigger icon={LayoutDashboard} label="Plan" value="plan" />
        <TabTrigger icon={Database} label="Evidence" value="evidence" />
        <TabTrigger icon={FileText} label="Markdown Preview" value="markdown" />
        <TabTrigger icon={ImageIcon} label="Images" value="images" />
        <TabTrigger icon={Terminal} label="Graph Logs" value="logs" />
      </TabsList>

      <TabsContent className="w-full min-w-0" value="plan">
        <PlanView plan={result?.plan} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0" value="evidence">
        <EvidenceView result={result} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0" value="markdown">
  <MarkdownPreview 
    markdown={result?.final} 
    status={status}
    images={imageMap}  // ✅ add this
  />
</TabsContent>
      <TabsContent className="w-full min-w-0" value="images">
        <ImagesView result={result} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0" value="logs">
        <GraphLogs
          error={error}
          phaseIndex={phaseIndex}
          result={result}
          status={status}
          topic={topic}
        />
      </TabsContent>
    </Tabs>
  )
}

function TabTrigger({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof FileText
  label: string
  value: string
}) {
  return (
    <TabsTrigger
      className="
h-10 flex-none rounded-xl border border-transparent px-3 text-xs
text-zinc-500 transition
hover:bg-white/[0.04]
hover:text-zinc-200

data-[state=active]:border-cyan-400/50
data-[state=active]:bg-cyan-500/15
data-[state=active]:text-white
data-[state=active]:shadow-[0_0_25px_rgba(6,182,212,0.35)]
"
      value={value}
    >
      <Icon className="size-4" />
      <span>{label}</span>
    </TabsTrigger>
  )
}
