"use client"

import Link from "next/link"
import {
  ArrowLeft,
  Download,
} from "lucide-react"

import JSZip from "jszip"
import { saveAs } from "file-saver"
import { Header } from "@/components/layout/header"
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
    activeSession?.result?.generated_images ?? []
  ).reduce(
    (acc, img) => {
    if (img.image_data) {
      acc[img.filename] = img.image_data
    }
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
      activeSession?.result?.generated_images ?? []

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
  <main className="min-h-screen bg-[#f5f0e8] text-black">

  {/* Header */}
    
  <header className="sticky top-0 z-50 border-b-[3px] border-black bg-[#f5f0e8]">
    <div className="mx-auto flex h-16 max-w-5xl items-center justify-between px-6">

      {/* Back */}
      <Link
        href="/"
        className="flex items-center gap-2 border-[2px] border-black bg-white px-3 py-1.5 font-mono text-xs font-black uppercase tracking-wider text-black shadow-[2px_2px_0px_#000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0px_#000] hover:bg-[#fce135]"
      >
        <ArrowLeft className="size-4 stroke-[2.5px]" />
        Back to Workspace
      </Link>

      {/* Download Buttons */}
      <div className="flex items-center gap-3">
        <button
          onClick={downloadImages}
          className="flex items-center gap-2 border-[2px] border-black bg-black px-4 py-2 font-mono text-xs font-black uppercase tracking-wider text-[#c8f135] shadow-[3px_3px_0px_#000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0px_#000]"
        >
          <Download className="size-4 stroke-[2.5px]" />
          Download Images
        </button>

        <button
          onClick={downloadMarkdown}
          className="flex items-center gap-2 border-[2px] border-black bg-[#fce135] px-4 py-2 font-mono text-xs font-black uppercase tracking-wider text-black shadow-[3px_3px_0px_#000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[2px_2px_0px_#000]"
        >
          <Download className="size-4 stroke-[2.5px]" />
          Download Markdown
        </button>
      </div>

    </div>
  </header>

  {/* Content */}
  <section className="px-6 py-12">
    <div className="mx-auto max-w-4xl">
      <article
        id="blog-content"
        className="border-[3px] border-black bg-white p-6 shadow-[8px_8px_0px_#000] sm:p-10"
      >
        {markdown ? (
          <MarkdownRenderer markdown={markdown} images={imageMap} />
        ) : (
          <div className="flex min-h-[400px] items-center justify-center font-mono text-sm font-black uppercase tracking-widest text-gray-400">
            No generated blog found.
          </div>
        )}
      </article>
    </div>
  </section>

</main>
  )
}