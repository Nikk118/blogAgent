"use client";

import { useEffect, useState } from "react";
import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store";

import {
  ChevronLeft,
  ChevronRight,
  MessageSquareDashed,
  Plus,
  Settings,
  Sparkles,
} from "lucide-react";

import { onAuthStateChanged, signOut, User } from "firebase/auth";
import { useRouter } from "next/navigation";
import { auth } from "@/lib/firebase";
import { StatusPill } from "@/components/dashboard/status-pill";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ExecutionStatus, WorkspaceSession } from "@/types/blog";

interface WorkspaceSidebarProps {
  activeSessionId: string | null;
  sessions: WorkspaceSession[];
  status: ExecutionStatus;
  collapsed: boolean;
  isLoadingBlogs: boolean;
  setCollapsed: React.Dispatch<React.SetStateAction<boolean>>;
  onSelectSession: (id: string) => void;
  onClearSession: () => void;
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
  const router = useRouter();
  const [showMenu, setShowMenu] = useState(false);
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  const handleLogout = async () => {
    try {
      useBlogWorkspaceStore.getState().resetStore();
      await signOut(auth);
      router.push("/login");
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 z-40 flex h-screen flex-col border-r-[3px] border-black bg-[#f5f0e8] transition-all duration-300 text-black",
        collapsed ? "w-[72px]" : "w-[280px]"
      )}
    >
      {/* Top hatch accent strip */}
      <div
        className="absolute top-0 left-0 right-0 h-[4px] z-10"
        style={{
          background: "repeating-linear-gradient(90deg,#c8f135 0,#c8f135 16px,#000 16px,#000 20px)",
        }}
      />

      {/* Collapse tab */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute -right-3.5 top-6 z-50 flex h-7 w-7 items-center justify-center border-[2px] border-black bg-[#fce135] text-black shadow-[2px_2px_0px_#000] transition hover:bg-[#ff2d78] hover:text-white hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0px_#000]"
      >
        {collapsed ? (
          <ChevronRight className="size-4 stroke-[3px]" />
        ) : (
          <ChevronLeft className="size-4 stroke-[3px]" />
        )}
      </button>

      <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-4 pt-6">

        {/* Header Block */}
        <div className={cn("flex items-center mt-1", collapsed ? "justify-center" : "justify-between")}>
          <div className={cn("flex items-center", collapsed ? "justify-center" : "gap-3")}>
            {/* Logo box — black with lime icon */}
            <div className="relative flex h-10 w-10 shrink-0 items-center justify-center border-[2px] border-black bg-black shadow-[3px_3px_0px_#000]">
              <Sparkles className="relative size-5 text-[#c8f135] stroke-[2.5px]" />
            </div>

            {!collapsed && (
              <span
                className="uppercase tracking-[0.1em] text-black leading-none"
                style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "20px" }}
              >
                Blog Agent
              </span>
            )}
          </div>

          {!collapsed && (
            <StatusPill status={status} className="hidden lg:inline-flex" />
          )}
        </div>

        {/* New Blog Button */}
        <Button
          className={cn(
            "my-6 border-[2px] border-black bg-[#ff2d78] font-black uppercase tracking-wider text-white shadow-[4px_4px_0px_#000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000] hover:bg-[#ff2d78] rounded-none",
            collapsed ? "h-11 w-11 self-center p-0 shadow-[2px_2px_0px_#000]" : "h-11 w-full"
          )}
          onClick={onClearSession}
          type="button"
        >
          <Plus className="size-5 shrink-0 stroke-[3px]" />
          {!collapsed && <span className="ml-2">New Blog</span>}
        </Button>

        {/* Sessions List */}
        <div className="mt-4 flex min-h-0 flex-1 flex-col">
          {!collapsed && (
            <div className="mb-3 flex items-center justify-between px-1">
              <span className="text-[11px] font-black uppercase tracking-widest text-black">
                Recent Blogs
              </span>
              <span className="border-[2px] border-black bg-black px-1.5 font-mono text-[10px] font-black uppercase text-[#c8f135]">
                {sessions.length.toString().padStart(2, "0")}
              </span>
            </div>
          )}

          <div
            className={cn(
              "dashboard-scrollbar min-h-0 flex-1 overflow-y-auto",
              collapsed ? "flex flex-col items-center gap-4" : "flex flex-col gap-2 pr-1"
            )}
          >
            {isLoadingBlogs ? (
              <div className={cn("flex flex-col", collapsed ? "items-center gap-4" : "gap-2")}>
                {[1, 2, 3].map((item) => (
                  <div
                    key={item}
                    className={cn(
                      "border-[2px] border-black/20 bg-black/5 animate-pulse",
                      collapsed ? "h-11 w-11" : "flex items-center gap-3 px-3 py-2.5"
                    )}
                  >
                    {!collapsed && (
                      <>
                        <div className="h-8 w-8 border border-black/20 bg-black/10" />
                        <div className="flex flex-1 flex-col gap-2">
                          <div className="h-3 w-28 bg-black/10" />
                          <div className="h-2 w-16 bg-black/5" />
                        </div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ) : sessions.length === 0 ? (
              collapsed ? (
                <div className="mt-2 h-2.5 w-2.5 border-[2px] border-black bg-black/20" />
              ) : (
                <div className="border-[2px] border-dashed border-black bg-black/5 p-4 font-mono text-xs font-bold text-gray-500">
                  Generated blogs will appear here automatically.
                </div>
              )
            ) : (
              sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => onSelectSession(session.id)}
                  type="button"
                  className={cn(
                    "group transition-all text-left border-[2px] border-black",
                    collapsed
                      ? "flex h-11 w-11 items-center justify-center"
                      : "flex items-center gap-3 px-3 py-2.5",
                    activeSessionId === session.id
                      ? "bg-[#fce135] shadow-[3px_3px_0px_#000]"
                      : "bg-transparent hover:bg-white hover:shadow-[2px_2px_0px_#000]"
                  )}
                >
                  <div
                    className={cn(
                      "flex items-center justify-center h-8 w-8 border-[2px] border-black shrink-0 transition-colors",
                      activeSessionId === session.id
                        ? "bg-black text-[#c8f135]"
                        : "bg-white text-black group-hover:bg-[#ff2d78] group-hover:text-white"
                    )}
                  >
                    <MessageSquareDashed className="size-4 stroke-[2px]" />
                  </div>

                  {!collapsed && (
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-xs font-black uppercase tracking-tight text-black">
                        {session.title || "Untitled Blog"}
                      </p>
                      <p className="mt-0.5 font-mono text-[10px] font-bold text-gray-500">
                        Saved workspace
                      </p>
                    </div>
                  )}
                </button>
              ))
            )}
          </div>
        </div>

        {/* Profile Block */}
        <div
          className={cn(
            "mt-5 border-t-[3px] border-black pt-4",
            collapsed && "flex justify-center"
          )}
        >
          <div
            className={cn(
              "border-[2px] border-black bg-white shadow-[3px_3px_0px_#000] transition-all",
              collapsed
                ? "flex h-11 w-11 items-center justify-center"
                : "flex items-center justify-between p-2.5"
            )}
          >
            <div className="flex min-w-0 items-center gap-2.5">
              {/* Avatar */}
              <div className="flex h-8 w-8 shrink-0 items-center justify-center border-[2px] border-black overflow-hidden bg-[#ff2d78] text-xs font-black text-white">
                {user?.photoURL ? (
                  <img src={user.photoURL} alt="profile" className="h-full w-full object-cover" />
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
                  <p className="truncate text-xs font-black text-black">
                    {user?.displayName || user?.email || "Guest User"}
                  </p>
                  <p className="truncate font-mono text-[9px] font-bold text-gray-500">
                    cloud.persistence.enabled
                  </p>
                </div>
              )}
            </div>

            {!collapsed && (
              <div className="relative">
                <button
                  onClick={(e) => { e.stopPropagation(); setShowMenu((prev) => !prev); }}
                  className="border-[2px] border-transparent p-1.5 text-black transition hover:border-black hover:bg-[#fce135]"
                >
                  <Settings className="size-4 stroke-[2.5px]" />
                </button>

                {showMenu && (
                  <div
                    className="absolute right-0 bottom-11 z-[9999] w-40 border-[2px] border-black bg-white p-1.5 shadow-[4px_4px_0px_#000]"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <button
                      onClick={handleLogout}
                      className="flex w-full items-center px-2.5 py-1.5 font-mono text-xs font-black uppercase tracking-wider text-[#ff2d78] transition hover:bg-black hover:text-white"
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
  );
}