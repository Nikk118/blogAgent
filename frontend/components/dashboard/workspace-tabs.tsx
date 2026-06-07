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
  const imageMap = (result?.generated_images ?? []).reduce((acc, img) => {
    if (img.image_data) acc[img.filename] = img.image_data
    return acc
  }, {} as Record<string, string>)

  return (
    <Tabs defaultValue="markdown" className="flex w-full min-w-0 flex-col gap-6">
      <TabsList className="dashboard-scrollbar flex h-auto w-full shrink-0 justify-start gap-2 overflow-x-auto rounded-none border-[3px] border-black bg-[#f5f0e8] p-2 shadow-[6px_6px_0px_#000]">
        <TabTrigger icon={LayoutDashboard} label="Plan"            value="plan"     />
        <TabTrigger icon={Database}        label="Evidence"        value="evidence" />
        <TabTrigger icon={FileText}        label="Markdown Preview" value="markdown" />
        <TabTrigger icon={ImageIcon}       label="Images"          value="images"   />
        <TabTrigger icon={Terminal}        label="Graph Logs"      value="logs"     />
      </TabsList>

      <TabsContent className="w-full min-w-0 focus-visible:outline-none" value="plan">
        <PlanView plan={result?.plan} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0 focus-visible:outline-none" value="evidence">
        <EvidenceView result={result} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0 focus-visible:outline-none" value="markdown">
        <MarkdownPreview markdown={result?.final} status={status} images={imageMap} />
      </TabsContent>
      <TabsContent className="w-full min-w-0 focus-visible:outline-none" value="images">
        <ImagesView result={result} status={status} />
      </TabsContent>
      <TabsContent className="w-full min-w-0 focus-visible:outline-none" value="logs">
        <GraphLogs error={error} phaseIndex={phaseIndex} result={result} status={status} topic={topic} />
      </TabsContent>
    </Tabs>
  )
}

function TabTrigger({
  icon: Icon,
  label,
  value,
}: {
  icon: any
  label: string
  value: string
}) {
  return (
    <TabsTrigger
      value={value}
      className="
        flex h-10 items-center gap-2
        border-[2px] border-black
        px-4
        font-mono text-xs font-black uppercase tracking-wider text-black
        rounded-none
        bg-white
        shadow-[2px_2px_0px_#000]
        transition-all
        hover:bg-[#fce135]
        hover:translate-x-[1px] hover:translate-y-[1px]
        hover:shadow-[1px_1px_0px_#000]
        data-[state=active]:bg-[#fce135]
        data-[state=active]:shadow-[0px_0px_0px_#000]
        data-[state=active]:translate-x-[2px]
        data-[state=active]:translate-y-[2px]
      "
    >
      <Icon className="size-4 stroke-[2.5px]" />
      <span>{label}</span>
    </TabsTrigger>
  )
}