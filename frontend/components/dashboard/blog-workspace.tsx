"use client"

import { AlertTriangle } from "lucide-react"
import { useCallback, useEffect, useState } from "react"

import { WorkspaceHero } from "@/components/dashboard/workspace-hero"
import { WorkspaceInput } from "@/components/dashboard/workspace-input"
import { WorkspaceSidebar } from "@/components/dashboard/workspace-sidebar"
import { WorkspaceTabs } from "@/components/dashboard/workspace-tabs"

import {
  generateBlog,
  getBlogs,
} from "@/lib/api"
import {
  flattenEvidence,
  getImageSpecs,
  getMarkdownWordCount,
  getSectionCount,
} from "@/lib/blog-normalizers"

import { cn } from "@/lib/utils"

import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store"
import { Header } from "../layout/header"

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

  // Sidebar state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const activeSession = sessions.find(
    (session) => session.id === activeSessionId
  )
  useEffect(() => {

  async function loadBlogs() {

    try {

      const blogs = await getBlogs()

      const mappedSessions = blogs.map(
  (blog: any) => ({

    id: String(blog.id),

    topic: blog.title,

    title: blog.title,

    status: "completed",

    result: {
      ...blog.content,
      images: blog.images,
    },

    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  })
)

      setSessions(mappedSessions)

      // Auto select latest blog
      if (mappedSessions.length > 0) {

        selectSession(
          mappedSessions[0].id
        )
      }

    } catch (error) {

      console.error(
        "Failed to load blogs",
        error
      )
    }finally {

  setIsLoadingBlogs(false)
}
  }

  loadBlogs()

}, [selectSession, setSessions])

  const result = activeSession?.result

  const evidenceCount = flattenEvidence(result).length
  const imageCount = getImageSpecs(result).length
  const sectionCount = getSectionCount(result)
  const wordCount = getMarkdownWordCount(result?.final)

  const topic = activeSession?.topic || draftTopic

  useEffect(() => {
    if (status !== "running") {
      return
    }

    
    const interval = window.setInterval(() => {
      advancePhase()
    }, 1600)

    return () => window.clearInterval(interval)
  }, [advancePhase, status])
const [isLoadingBlogs, setIsLoadingBlogs] =
  useState(true)
  const handleGenerate = useCallback(async () => {
    const cleanTopic = draftTopic.trim()

    if (!cleanTopic || status === "running") {
      return
    }

    const sessionId = startGeneration(cleanTopic)

    try {
      const data = await generateBlog(cleanTopic)

      completeGeneration(
        sessionId,
        data.result,
        data.topic || cleanTopic
      )
    } catch (error) {
      failGeneration(sessionId, getErrorMessage(error))
    }
  }, [
    completeGeneration,
    draftTopic,
    failGeneration,
    startGeneration,
    status,
  ])


  
  return (
    <div className="relative min-h-screen overflow-hidden bg-[#030407] text-zinc-100">
      
      {/* Background Effects */}
      <div className="pointer-events-none absolute inset-0 runtime-grid opacity-[0.16]" />

      <div className="pointer-events-none absolute left-1/3 top-[-18rem] size-[44rem] rounded-full bg-teal-300/10 blur-3xl" />

      <div className="pointer-events-none absolute bottom-[-22rem] right-[-12rem] size-[42rem] rounded-full bg-rose-300/8 blur-3xl" />

      {/* Layout */}
      <div className="relative z-10 flex min-h-screen">
        
        {/* Sidebar */}
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

        {/* Main Content */}
        <main
  className={cn(
    "relative flex min-w-0 flex-1 flex-col pb-10 transition-all duration-300",
    sidebarCollapsed
      ? "md:ml-[72px]"
      : "md:ml-[280px]"
  )}
>
 <Header  collapsed={sidebarCollapsed}/>
          <div className="dashboard-scrollbar min-h-0 flex-1 overflow-y-auto">
            
           <div
  className={cn(
    "flex w-full flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8 transition-all duration-300",
    sidebarCollapsed
      ? "mx-auto max-w-[1400px]"
      : "mx-auto max-w-[1720px]"
  )}
>
              
              {/* Error */}
              {activeSession?.error ? (
                <div className="flex items-start gap-3 rounded-2xl border border-rose-300/20 bg-rose-300/10 p-4 text-sm text-rose-100">
                  <AlertTriangle className="mt-0.5 size-4 shrink-0" />

                  <span>{activeSession.error}</span>
                </div>
              ) : null}

              {/* Hero */}
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

              {/* Tabs */}
              <WorkspaceTabs
                error={activeSession?.error}
                phaseIndex={phaseIndex}
                result={result}
                status={status}
                topic={topic}
              />
            </div>
          </div>

          {/* Input */}
          {(!activeSessionId || status === "running") && (
         <WorkspaceInput
  draftTopic={draftTopic}
  status={status}
  sidebarCollapsed={sidebarCollapsed}
  isCentered={
    activeSessionId === null && status === "idle"
  }
  onDraftTopicChange={setDraftTopic}
  onGenerate={handleGenerate}
/>)}
         

        </main>
      </div>
    </div>
  )
}

function getErrorMessage(error: unknown) {
  if (
    error &&
    typeof error === "object" &&
    "message" in error
  ) {
    return String(
      (error as { message?: unknown }).message
    )
  }

  return "Generation failed. Check the backend runtime and retry."
}