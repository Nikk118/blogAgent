"use client"

import {
  ChevronLeft,
  ChevronRight,
  MessageSquareDashed,
  Plus,
  Settings,
  Sparkles,
} from "lucide-react"

import { StatusPill } from "@/components/dashboard/status-pill"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ExecutionStatus, WorkspaceSession } from "@/types/blog"

interface WorkspaceSidebarProps {
  activeSessionId: string | null
  sessions: WorkspaceSession[]
  status: ExecutionStatus
  collapsed: boolean
  setCollapsed: React.Dispatch<React.SetStateAction<boolean>>
  onNewSession: () => void
  onSelectSession: (id: string) => void
}

export function WorkspaceSidebar({
  activeSessionId,
  sessions,
  status,
  collapsed,
  setCollapsed,
  onNewSession,
  onSelectSession,
}: WorkspaceSidebarProps) {
  return (
    <aside
      className={cn(
        "fixed left-0 top-16 z-40 flex h-[calc(100vh-64px)] flex-col border-r border-white/10 bg-black/30 backdrop-blur-2xl transition-all duration-300",
        collapsed ? "w-[72px]" : "w-[280px]"
      )}
    >
      {/* Toggle Button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3 top-6 z-50 flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-zinc-950 text-zinc-400 transition hover:bg-zinc-900 hover:text-white"
      >
        {collapsed ? (
          <ChevronRight className="size-4" />
        ) : (
          <ChevronLeft className="size-4" />
        )}
      </button>

      <div className="flex min-h-0 flex-1 flex-col p-3 overflow-hidden">
        
        {/* Header */}
        <div
          className={cn(
            "flex items-center",
            collapsed ? "justify-center" : "justify-between"
          )}
        >
          <div
            className={cn(
              "flex items-center",
              collapsed ? "justify-center" : "gap-3"
            )}
          >
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.03]">
              <div className="absolute inset-0 rounded-2xl bg-teal-300/10 blur-md" />

              <Sparkles className="relative size-5 text-teal-200" />
            </div>

            {!collapsed && (
              <div>
                <h2 className="text-sm font-semibold text-zinc-100">
                  Workspace
                </h2>

                <p className="text-[11px] text-zinc-500">
                  AI orchestration panel
                </p>
              </div>
            )}
          </div>

          {!collapsed && (
            <StatusPill
              status={status}
              className="hidden lg:inline-flex"
            />
          )}
        </div>

        {/* New Session */}
        <Button
          className={cn(
            "mt-6 border border-white/10 bg-white/[0.04] text-zinc-100 shadow-none transition hover:bg-white/[0.08]",
            collapsed
              ? "h-11 w-11 self-center rounded-2xl p-0"
              : "h-11 w-full rounded-2xl"
          )}
          onClick={onNewSession}
          type="button"
        >
          <Plus className="size-4 shrink-0" />

          {!collapsed && (
            <span className="ml-2">
              New Session
            </span>
          )}
        </Button>

        {/* Sessions */}
        <div className="mt-8 flex min-h-0 flex-1 flex-col overflow-hidden">
          
          {!collapsed && (
            <div className="mb-3 flex items-center justify-between px-1">
              <span className="text-[11px] uppercase tracking-wider text-zinc-500">
                Recent Sessions
              </span>

              <span className="font-mono text-[10px] text-zinc-600">
                {sessions.length.toString().padStart(2, "0")}
              </span>
            </div>
          )}

          <div
            className={cn(
              "dashboard-scrollbar flex flex-1 flex-col overflow-y-auto",
              collapsed ? "gap-3 items-center" : "gap-2 pr-1"
            )}
          >
            {sessions.length === 0 ? (
              collapsed ? (
                <div className="mt-2 h-2 w-2 rounded-full bg-zinc-700" />
              ) : (
                <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-4 text-sm text-zinc-500">
                  Sessions appear here after the first generation.
                </div>
              )
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  type="button"
                  className={cn(
                    "group transition-all duration-200",
                    collapsed
                      ? "flex h-11 w-11 items-center justify-center rounded-2xl border"
                      : "flex items-center gap-3 rounded-2xl border px-4 py-3 text-left",
                    activeSessionId === session.id
                      ? "border-teal-300/20 bg-teal-300/10 shadow-[0_10px_40px_rgba(45,212,191,0.08)]"
                      : "border-white/5 bg-white/[0.025] hover:border-white/10 hover:bg-white/[0.04]"
                  )}
                >
                  <div
                    className={cn(
                      "flex items-center justify-center rounded-xl",
                      collapsed
                        ? "h-9 w-9"
                        : "h-9 w-9 border border-white/10 bg-white/[0.03]",
                      activeSessionId === session.id &&
                        "text-teal-200"
                    )}
                  >
                    <MessageSquareDashed className="size-4" />
                  </div>

                  {!collapsed && (
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-zinc-200">
                        {session.title || "Untitled Session"}
                      </p>

                      <p className="mt-1 text-[11px] text-zinc-500">
                        Active workspace thread
                      </p>
                    </div>
                  )}
                </button>
              ))
            )}
          </div>
        </div>
 
        {/* Bottom Profile */}
        <div
          className={cn(
            "mt-5 border-t border-white/10 pt-4",
            collapsed && "flex justify-center"
          )}
        >
        
          <div
            className={cn(
              "group transition hover:bg-white/[0.04]",
              collapsed
                ? "flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.02]"
                : "flex items-center justify-between rounded-2xl border border-white/5 bg-white/[0.02] p-3"
            )}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-teal-300 to-cyan-400 text-sm font-semibold text-black">
                N
              </div>

              {!collapsed && (
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-zinc-200">
                    Nikhil Yagnik
                  </p>

                  <p className="truncate font-mono text-[10px] text-zinc-500">
                    graph.runtime.v1
                  </p>
                </div>
              )}
            </div>

            {!collapsed && (
              <Settings className="size-4 text-zinc-500 transition group-hover:text-zinc-300" />
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}