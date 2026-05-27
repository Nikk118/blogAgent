"use client"

import Link from "next/link"

import {
  ArrowLeft,
  Download,
  ExternalLink,
  Image as ImageIcon,
} from "lucide-react"

import html2canvas from "html2canvas"
import jsPDF from "jspdf"

import { MarkdownRenderer } from "@/components/dashboard/markdown-renderer"

import { useBlogWorkspaceStore } from "@/stores/blog-workspace-store"

export default function BlogPage() {
  const { sessions, activeSessionId } =
    useBlogWorkspaceStore()

  const activeSession = sessions.find(
    (session) => session.id === activeSessionId
  )

  const markdown =
    activeSession?.result?.final || ""

  async function downloadPDF() {
    const article =
      document.getElementById("blog-content")

    if (!article) return

    const canvas = await html2canvas(article, {
      scale: 2,
      backgroundColor: "#09090b",
    })

    const imgData =
      canvas.toDataURL("image/png")

    const pdf = new jsPDF({
      orientation: "portrait",
      unit: "px",
      format: "a4",
    })

    const pdfWidth =
      pdf.internal.pageSize.getWidth()

    const pdfHeight =
      (canvas.height * pdfWidth) /
      canvas.width

    pdf.addImage(
      imgData,
      "PNG",
      0,
      0,
      pdfWidth,
      pdfHeight
    )

    pdf.save("blog-article.pdf")
  }

  async function downloadImage() {
    const article =
      document.getElementById("blog-content")

    if (!article) return

    const canvas = await html2canvas(article, {
      scale: 2,
      backgroundColor: "#09090b",
    })

    const url =
      canvas.toDataURL("image/png")

    const link =
      document.createElement("a")

    link.href = url
    link.download = "blog-article.png"

    link.click()
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

          {/* Actions */}
          <div className="flex items-center gap-3">
            
            {/* Download PDF */}
            <button
              onClick={downloadPDF}
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
              Download PDF
            </button>

            {/* Download Image */}
            <button
              onClick={downloadImage}
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
              <ImageIcon className="size-4" />
              Download Image
            </button>
          </div>
        </div>
      </header>

      {/* Content */}
      <section className="px-6 py-16">
        
        <div className="mx-auto max-w-4xl">
          
          <article
            id="blog-content"
            className="rounded-3xl border border-white/10 bg-[#07090e]/82 p-6 shadow-[0_30px_120px_rgba(0,0,0,0.34)] sm:p-10"
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