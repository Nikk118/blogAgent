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
        "fixed right-0 z-50 flex justify-center px-4 pointer-events-none transition-all duration-500 ease-in-out",
        sidebarCollapsed
          ? "left-[72px]"
          : "left-[280px]",
        isCentered
          ? "top-1/2 -translate-y-1/2 items-center"
          : "bottom-8 items-end"
      )}
    >
      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className={cn(
          "pointer-events-auto relative flex w-full items-center justify-between overflow-hidden rounded-2xl border border-white/10 bg-black/40 shadow-2xl backdrop-blur-xl focus-within:border-teal-300/30 focus-within:ring-1 focus-within:ring-teal-300/20 transition-[max-width,padding] duration-300 ease-in-out",
          isCentered
            ? "max-w-5xl p-3"
            : "max-w-3xl py-2 pl-3 pr-2"
        )}
      >
        <Textarea
          autoFocus
          disabled={isRunning}
          rows={isCentered ? 3 : 1}
          value={draftTopic}
          onChange={(event) =>
            onDraftTopicChange(event.target.value)
          }
          onKeyDown={handleKeyDown}
          placeholder="Describe the blog target, audience, angle, or constraints..."
          className="min-h-[44px] w-full resize-none border-none bg-transparent px-2 py-3 text-sm leading-relaxed text-zinc-100 placeholder:text-zinc-500 focus-visible:border-none focus-visible:ring-0 disabled:opacity-50 md:text-sm"
        />

        <div className="absolute bottom-3 right-3 flex items-center gap-2">
          
          {isCentered && (
            <span className="mt-1 hidden items-center gap-1 rounded border border-white/10 bg-white/5 px-1.5 py-0.5 text-[10px] text-zinc-500 sm:flex">
              <CornerDownLeft className="size-3" />
              return
            </span>
          )}

          <Button
            type="submit"
            size="icon"
            disabled={!draftTopic.trim() || isRunning}
            className={cn(
              "h-8 w-8 shrink-0 rounded-xl p-0 transition-all",
              draftTopic.trim() && !isRunning
                ? "bg-teal-300 text-zinc-950 hover:bg-teal-200"
                : "bg-white/10 text-zinc-500 hover:bg-white/10"
            )}
          >
            <Send className="size-4" />
          </Button>
        </div>
      </form>
    </div>
  )
}