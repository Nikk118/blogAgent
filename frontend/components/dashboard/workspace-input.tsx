"use client"

import { Send, CornerDownLeft } from "lucide-react"
import { FormEvent, useRef } from "react"

import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import type { ExecutionStatus } from "@/types/blog"

interface WorkspaceInputProps {
  draftTopic: string
  status: ExecutionStatus
  isCentered: boolean
  sidebarCollapsed: boolean
  onDraftTopicChange: (topic: string) => void
  onGenerate: (topic: string) => Promise<void> | void
}

export function WorkspaceInput({
  draftTopic,
  status,
  isCentered,
  sidebarCollapsed,
  onDraftTopicChange,
  onGenerate,
}: WorkspaceInputProps) {
  const isRunning = status === "running"

  const formRef = useRef<HTMLFormElement>(null)

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!draftTopic.trim() || isRunning) return

    const currentTopic = draftTopic

    // Clear instantly
    onDraftTopicChange("")

    // Generate
    await onGenerate(currentTopic)
  }

  async function handleKeyDown(
    event: React.KeyboardEvent<HTMLTextAreaElement>
  ) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()

      if (!draftTopic.trim() || isRunning) return

      const currentTopic = draftTopic

      // Clear instantly
      onDraftTopicChange("")

      // Generate
      await onGenerate(currentTopic)
    }
  }

return (
    <div
  className={cn(
    "fixed right-0 z-50 flex justify-center px-4 pointer-events-none transition-all duration-300 ease-in-out",
    sidebarCollapsed ? "left-[72px]" : "left-[280px]",
    isCentered
      ? "top-1/2 -translate-y-1/2 items-center"
      : "bottom-8 items-end"
  )}
>
  <form
    ref={formRef}
    onSubmit={handleSubmit}
    className={cn(
      "pointer-events-auto relative flex w-full items-center justify-between border-[3px] border-black bg-[#f5f0e8] shadow-[6px_6px_0px_#000] focus-within:shadow-[8px_8px_0px_#000] transition-all duration-200 ease-in-out",
      isCentered ? "max-w-5xl p-4" : "max-w-3xl py-3 pl-4 pr-3"
    )}
  >
    <Textarea
      autoFocus
      disabled={isRunning}
      rows={isCentered ? 3 : 1}
      value={draftTopic}
      onChange={(e) => onDraftTopicChange(e.target.value)}
      onKeyDown={handleKeyDown}
      placeholder="Describe the blog target, audience, angle, or constraints..."
      className="min-h-[44px] w-full resize-none border-none bg-transparent px-1 py-2 text-sm font-semibold leading-relaxed text-black placeholder:text-gray-400 focus-visible:border-none focus-visible:ring-0 disabled:opacity-50 pb-12"
    />

    {/* Action Controls */}
    <div className="absolute bottom-3 right-3 flex items-center gap-3">

      {/* Shortcut badge */}
      {isCentered && (
        <span className="hidden items-center gap-1 border-[2px] border-black bg-[#fce135] px-2 py-0.5 font-mono text-[10px] font-black uppercase text-black shadow-[1px_1px_0px_#000] sm:flex">
          <CornerDownLeft className="size-3 stroke-[2.5px]" />
          return
        </span>
      )}

      {/* Submit */}
      <Button
        type="submit"
        size="icon"
        disabled={!draftTopic.trim() || isRunning}
        className={cn(
          "h-9 w-9 shrink-0 rounded-none border-[2px] border-black p-0 transition-all",
          draftTopic.trim() && !isRunning
            ? "bg-[#ff2d78] text-white shadow-[2px_2px_0px_#000] hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0px_#000] hover:bg-[#ff2d78]"
            : "bg-black/5 text-gray-400 border-black/20 shadow-none cursor-not-allowed"
        )}
      >
        <Send className="size-4 stroke-[3px]" />
      </Button>
    </div>
  </form>
</div>
  )
}