"use client"

import Link from "next/link"
import {
  ArrowLeft,
  Download,
} from "lucide-react"

import JSZip from "jszip"
import { saveAs } from "file-saver"

import { MarkdownRenderer } from "@/components/dashboard/markdown-renderer"
import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store"

export default function BlogPage() {

  const {
    sessions,
    activeSessionId,
  } = useBlogWorkspaceStore()

  const activeSession = sessions.find(
    (session) =>
      session.id === activeSessionId
  )

  const imageMap = (
    activeSession?.result?.images ?? []
  ).reduce(
    (acc, img) => {
      acc[img.filename] = img.image_data
      return acc
    },
    {} as Record<string, string>
  )

  const markdown =
    activeSession?.result?.final || ""

  function downloadMarkdown() {

    const blob = new Blob(
      [markdown],
      {
        type: "text/markdown",
      }
    )

    const link =
      document.createElement("a")

    link.href =
      URL.createObjectURL(blob)

    link.download =
      "blog-article.md"

    link.click()

    URL.revokeObjectURL(
      link.href
    )
  }

  async function downloadImages() {

    const images =
      activeSession?.result?.images ?? []

    if (images.length === 0) {
      return
    }

    const zip = new JSZip()

    images.forEach((img: any) => {

      if (!img.image_data) return

      zip.file(
        img.filename,
        img.image_data,
        {
          base64: true,
        }
      )
    })

    const blob =
      await zip.generateAsync({
        type: "blob",
      })

    saveAs(
      blob,
      `${activeSession?.title || "blog"}-images.zip`
    )
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

          {/* Download Buttons */}
          <div className="flex items-center gap-3">

            <button
              onClick={downloadImages}
              className="
                flex items-center gap-2
                rounded-xl
                border border-cyan-500/20
                bg-cyan-500/10
                px-4 py-2
                text-sm text-cyan-300
                transition-all duration-200
                hover:bg-cyan-500/20
                hover:text-cyan-200
              "
            >
              <Download className="size-4" />
              Download Images
            </button>

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
              Download Markdown
            </button>

          </div>

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
              <MarkdownRenderer
                markdown={markdown}
                images={imageMap}
              />
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