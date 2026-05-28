"use client"

import Link from "next/link"
import { ArrowLeft, Download } from "lucide-react"
import { MarkdownRenderer } from "@/components/dashboard/markdown-renderer"
import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store"

export default function BlogPage() {

  const { sessions, activeSessionId } = useBlogWorkspaceStore()

  const activeSession = sessions.find(
    (session) => session.id === activeSessionId
  )

  const markdown = activeSession?.result?.final || ""

  function downloadMarkdown() {
    const blob = new Blob([markdown], { type: "text/markdown" })
    const link = document.createElement("a")
    link.href = URL.createObjectURL(blob)
    link.download = "blog-article.md"
    link.click()
    URL.revokeObjectURL(link.href)
  }

  return (
    <main className="min-h-screen bg-[#09090b] text-zinc-100">

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/70 backdrop-blur-xl">

        <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">

          {/* Back */}
          <Link
            href="/"
            className="flex items-center gap-2 text-sm text-zinc-400 transition hover:text-white"
          >
            <ArrowLeft className="size-4" />
            Back to Workspace
          </Link>

          {/* Download */}
          <button
            onClick={downloadMarkdown}
            className="
              flex items-center gap-2
              rounded-xl
              border border-white/10
              bg-white/[0.03]
              px-4 py-2
              text-sm text-zinc-300
              transition-all duration-200
              hover:bg-white/[0.06]
              hover:text-white
            "
          >
            <Download className="size-4" />
            Download
          </button>
        </div>
      </header>

      {/* Content */}
      <section className="px-6 py-16">

        <div className="mx-auto max-w-4xl">

          <article
            id="blog-content"
            className="
              rounded-3xl
              border border-white/10
              bg-[#09090b]
              p-6
              shadow-[0_30px_120px_rgba(0,0,0,0.34)]
              sm:p-10
            "
          >
            {markdown ? (
              <MarkdownRenderer markdown={markdown} />
            ) : (
              <div className="flex min-h-[400px] items-center justify-center text-zinc-500">
                No generated blog found.
              </div>
            )}
          </article>
        </div>
      </section>
    </main>
  )
}