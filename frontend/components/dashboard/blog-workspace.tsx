"use client"

import { useCallback, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { onAuthStateChanged } from "firebase/auth"
import { AlertTriangle } from "lucide-react"

import { auth } from "@/lib/firebase"
import { generateBlog, getBlogs } from "@/lib/api"
import { cn } from "@/lib/utils"
import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store"
import { Header } from "../layout/header"

import { WorkspaceHero } from "@/components/dashboard/workspace-hero"
import { WorkspaceInput } from "@/components/dashboard/workspace-input"
import { WorkspaceSidebar } from "@/components/dashboard/workspace-sidebar"
import { WorkspaceTabs } from "@/components/dashboard/workspace-tabs"
import {
  flattenEvidence,
  getImageSpecs,
  getMarkdownWordCount,
  getSectionCount,
} from "@/lib/blog-normalizers"

export function BlogWorkspace() {
  const {
    activeSessionId,
    advancePhase,
    completeGeneration,
    draftTopic,
    failGeneration,
    clearSession,
    phaseIndex,
    selectSession,
    sessions,
    setDraftTopic,
    startGeneration,
    status,
    setSessions,
  } = useBlogWorkspaceStore()

  const router = useRouter()

  // Layout Controls
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [isLoadingBlogs, setIsLoadingBlogs] = useState(true)
  
  const activeSession = sessions.find(
    (session) => session.id === activeSessionId
  )

  // Synchronize Auth & Historical Session Trees
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (!user) {
        setIsLoadingBlogs(false)
        return
      }

      try {
        const blogs = await getBlogs()
        const mappedSessions = blogs.map((blog: any) => ({
          id: String(blog.id),
          topic: blog.title,
          title: blog.title,
          status: "completed",
          result: {
            ...blog.content,
            generated_images: blog.generated_images,
          },
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
        }))

        setSessions(mappedSessions)
        selectSession(null as any)
      } catch (error: any) {
        console.error("Failed to load blogs", error)

        if (error?.message === "User not authenticated" || error?.response?.status === 401) {
          router.replace("/login")
          return
        }
      } finally {
        setIsLoadingBlogs(false)
      }
    })

    return unsubscribe
  }, [selectSession, setSessions, router])

  // Context Metrics Calculation
  const result = activeSession?.result
  const evidenceCount = flattenEvidence(result).length
  const imageCount = getImageSpecs(result).length
  const sectionCount = getSectionCount(result)
  const wordCount = getMarkdownWordCount(result?.final)
  const topic = activeSession?.topic || draftTopic

  // Phase Multi-step Polling Simulator
  useEffect(() => {
    if (status !== "running") return

    const interval = window.setInterval(() => {
      advancePhase()
    }, 1600)

    return () => window.clearInterval(interval)
  }, [advancePhase, status])

  // Operational Request Dispatcher
  const handleGenerate = useCallback(async () => {
    const cleanTopic = draftTopic.trim()
    if (!cleanTopic || status === "running") return

    const sessionId = startGeneration(cleanTopic)

    try {
      const data = await generateBlog(cleanTopic)
      
      // Refresh all blogs to get images from DB
      const blogs = await getBlogs()
      const mappedSessions = blogs.map((blog: any) => ({
        id: String(blog.id),
        topic: blog.title,
        title: blog.title,
        status: "completed",
        result: {
          ...blog.content,
          generated_images: blog.generated_images,
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }))
      setSessions(mappedSessions)

      completeGeneration(
        sessionId,
        data.result,
        data.topic || cleanTopic
      )
    } catch (error) {
      failGeneration(sessionId, getErrorMessage(error))
    }
  }, [completeGeneration, draftTopic, failGeneration, startGeneration, status, setSessions])

return (
  <div className="relative pt-16 min-h-screen overflow-hidden bg-[#f5f0e8] text-black selection:bg-[#fce135]">

    {/* Background Blueprint Grid */}
    <div className="pointer-events-none absolute inset-0 runtime-grid opacity-[0.06]" />

    {/* Main Structural Core */}
    <div className="relative z-10 flex min-h-screen">

      <WorkspaceSidebar
        activeSessionId={activeSessionId}
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        onClearSession={clearSession}
        onSelectSession={selectSession}
        sessions={sessions}
        status={status}
        isLoadingBlogs={isLoadingBlogs}
      />

      <main
        className={cn(
          "relative flex min-w-0 flex-1 flex-col pb-12 transition-all duration-300",
          sidebarCollapsed ? "md:ml-[72px]" : "md:ml-[280px]"
        )}
      >
        <Header collapsed={sidebarCollapsed} />

        <div className="dashboard-scrollbar min-h-0 flex-1 overflow-y-auto">
          <div
            className={cn(
              "flex w-full flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8 transition-all duration-300",
              sidebarCollapsed ? "mx-auto max-w-[1400px]" : "mx-auto max-w-[1720px]"
            )}
          >

            {/* Error strip */}
            {activeSession?.error ? (
              <div className="flex items-center gap-3 border-[3px] border-black bg-[#ff2d78] p-4 text-white shadow-[6px_6px_0px_#000]">
                <AlertTriangle className="size-5 shrink-0 stroke-[2.5px]" />
                <span className="font-mono text-xs font-black uppercase tracking-widest leading-none">
                  Error // {activeSession.error}
                </span>
              </div>
            ) : null}

            <WorkspaceHero
              activeSession={activeSession}
              draftTopic={draftTopic}
              evidenceCount={evidenceCount}
              imageCount={imageCount}
              result={result}
              sectionCount={sectionCount}
              status={status}
              wordCount={wordCount}
            />

            <WorkspaceTabs
              error={activeSession?.error}
              phaseIndex={phaseIndex}
              result={result}
              status={status}
              topic={topic}
            />
          </div>
        </div>

        {(!activeSessionId || status === "running") && (
          <WorkspaceInput
            draftTopic={draftTopic}
            status={status}
            sidebarCollapsed={sidebarCollapsed}
            isCentered={activeSessionId === null && status === "idle"}
            onDraftTopicChange={setDraftTopic}
            onGenerate={handleGenerate}
          />
        )}

      </main>
    </div>
  </div>
)
}

function getErrorMessage(error: unknown) {
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message?: unknown }).message)
  }
  return "Generation failed. Check the backend runtime and retry."
}