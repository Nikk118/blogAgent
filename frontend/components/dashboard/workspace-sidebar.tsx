"use client"

import { useEffect, useState } from "react"

import {
  ChevronLeft,
  ChevronRight,
  MessageSquareDashed,
  Plus,
  Settings,
  Sparkles,
  Trash2,
} from "lucide-react"

import {
  onAuthStateChanged,
  signOut,
  User,
} from "firebase/auth"

import { useRouter } from "next/navigation"

import { auth } from "@/lib/firebase"

import { StatusPill } from "@/components/dashboard/status-pill"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

import type {
  ExecutionStatus,
  WorkspaceSession,
} from "@/types/blog"

interface WorkspaceSidebarProps {
  activeSessionId: string | null
  sessions: WorkspaceSession[]
  status: ExecutionStatus
  collapsed: boolean
  isLoadingBlogs: boolean
  setCollapsed: React.Dispatch<
    React.SetStateAction<boolean>
  >
  onSelectSession: (id: string) => void
  onClearSession: () => void
}

export function WorkspaceSidebar({
  activeSessionId,
  sessions,
  status,
  collapsed,
  setCollapsed,
  onSelectSession,
  onClearSession,
  isLoadingBlogs,
}: WorkspaceSidebarProps) {

  const router = useRouter()

  const [showMenu, setShowMenu] = useState(false)

  const [user, setUser] = useState<User | null>(null)

  useEffect(() => {

    const unsubscribe = onAuthStateChanged(
      auth,
      (currentUser) => {
        setUser(currentUser)
      }
    )

    return () => unsubscribe()

  }, [])

  const handleLogout = async () => {

    try {

      await signOut(auth)

      router.push("/login")

    } catch (error) {

      console.error(error)
    }
  }

  return (
    <aside
      className={cn(
        "fixed left-0 top-16 z-40 flex h-[calc(100vh-64px)] flex-col border-r border-white/10 bg-black/30 backdrop-blur-2xl transition-all duration-300",
        collapsed ? "w-[72px]" : "w-[280px]"
      )}
    >

      {/* Collapse Button */}
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

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-3">

        {/* Header */}
        <div
          className={cn(
            "flex items-center",
            collapsed
              ? "justify-center"
              : "justify-between"
          )}
        >

          <div
            className={cn(
              "flex items-center",
              collapsed
                ? "justify-center"
                : "gap-3"
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
                  Persistent AI blog sessions
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
            "my-6 border border-white/10 bg-white/[0.04] text-zinc-100 shadow-none transition hover:bg-white/[0.08]",
            collapsed
              ? "h-11 w-11 self-center rounded-2xl p-0"
              : "h-11 w-full rounded-2xl"
          )}
          onClick={onClearSession}
          type="button"
        >
          <Plus className="size-4 shrink-0" />

          {!collapsed && (
            <span className="ml-2">
              New Blog
            </span>
          )}
        </Button>

        {/* Sessions */}
        <div
  className={cn(
    "dashboard-scrollbar flex flex-1 flex-col overflow-y-auto",
    collapsed
      ? "items-center gap-3"
      : "gap-2 pr-1"
  )}
>

  {/* =========================
      LOADING SKELETON
  ========================= */}

  {isLoadingBlogs ? (

    <div
      className={cn(
        "flex flex-col",
        collapsed
          ? "items-center gap-3"
          : "gap-3"
      )}
    >

      {[1, 2, 3].map((item) => (

        <div
          key={item}
          className={cn(
            "animate-pulse rounded-2xl border border-white/5 bg-white/[0.03]",
            collapsed
              ? "h-11 w-11"
              : "flex items-center gap-3 px-4 py-3"
          )}
        >

          {!collapsed && (

            <>
              <div className="h-9 w-9 rounded-xl bg-white/10" />

              <div className="flex flex-1 flex-col gap-2">

                <div className="h-3 w-32 rounded bg-white/10" />

                <div className="h-2 w-20 rounded bg-white/5" />
              </div>
            </>
          )}
        </div>
      ))}
    </div>

  ) : sessions.length === 0 ? (

    collapsed ? (

      <div className="mt-2 h-2 w-2 rounded-full bg-zinc-700" />

    ) : (

      <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-4 text-sm text-zinc-500">
        Generated blogs will appear here automatically.
      </div>
    )

  ) : (

    sessions.map((session) => (

      <button
        key={session.id}
        onClick={() =>
          onSelectSession(session.id)
        }
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
              {session.title ||
                "Untitled Blog"}
            </p>

            <p className="mt-1 text-[11px] text-zinc-500">
              Saved blog workspace
            </p>
          </div>
        )}
      </button>
    ))
  )}
</div>

        {/* Profile */}
        <div
          className={cn(
            "mt-5 border-t border-white/10 pt-4",
            collapsed && "flex justify-center"
          )}
        >

          <div
            className={cn(
              "group relative transition hover:bg-white/[0.04]",
              collapsed
                ? "flex h-11 w-11 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.02]"
                : "flex items-center justify-between rounded-2xl border border-white/5 bg-white/[0.02] p-3"
            )}
          >

            <div className="flex min-w-0 items-center gap-3">

              <div className="flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-full bg-gradient-to-br from-teal-300 to-cyan-400 text-sm font-semibold text-black">

                {user?.photoURL ? (

                  <img
                    src={user.photoURL}
                    alt="profile"
                    className="h-full w-full object-cover"
                  />

                ) : (

                  <span>
                    {user?.displayName?.[0]?.toUpperCase() ||
                      user?.email?.[0]?.toUpperCase() ||
                      "U"}
                  </span>
                )}
              </div>

              {!collapsed && (

                <div className="min-w-0">

                  <p className="truncate text-sm font-medium text-zinc-200">
                    {user?.displayName ||
                      user?.email ||
                      "Guest User"}
                  </p>

                  <p className="truncate font-mono text-[10px] text-zinc-500">
                    cloud.persistence.enabled
                  </p>
                </div>
              )}
            </div>

            {!collapsed && (

              <div className="relative">

                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    setShowMenu((prev) => !prev)
                  }}
                  className="rounded-xl p-2 text-zinc-500 transition hover:bg-white/[0.05] hover:text-zinc-300"
                >
                  <Settings className="size-4" />
                </button>

                {showMenu && (

                  <div
                    className="absolute right-0 bottom-14 z-[9999] w-44 rounded-2xl border border-white/10 bg-zinc-950 p-2 shadow-2xl"
                    onClick={(e) => e.stopPropagation()}
                  >

                    <button
                      className="flex w-full items-center gap-2 rounded-xl px-3 py-2 text-sm text-zinc-300 transition hover:bg-white/[0.05]"
                    >
                      <Trash2 className="size-4" />
                      Clear Workspace
                    </button>

                    <button
                      onClick={handleLogout}
                      className="mt-1 flex w-full items-center rounded-xl px-3 py-2 text-sm text-teal-300 transition hover:bg-red-500/10"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </aside>
  )
}