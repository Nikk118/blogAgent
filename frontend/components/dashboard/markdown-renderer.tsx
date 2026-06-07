"use client"

import Link from "next/link"
import remarkGfm from "remark-gfm"
import { Check, Copy, ExternalLink, FileText } from "lucide-react"
import { useState } from "react"
import ReactMarkdown, { type Components } from "react-markdown"

import { EmptyState, PanelSkeleton } from "@/components/dashboard/empty-state"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ExecutionStatus } from "@/types/blog"

type ImageMap = Record<string, string>

export function MarkdownPreview({
  markdown,
  status,
  images = {},
}: {
  markdown?: string
  status: ExecutionStatus
  images?: ImageMap
}) {
  if (!markdown && status === "running") {
    return (
      <div className="border-[3px] border-black bg-white p-6 shadow-[6px_6px_0px_#000000] rounded-none">
        <PanelSkeleton />
      </div>
    )
  }

  if (!markdown) {
    return (
      <EmptyState
        description="The generated markdown artifact will render here with structured headings, callouts, lists, links, and styled code blocks."
        icon={FileText}
        title="No markdown artifact yet"
      />
    )
  }

  return (
    <article className="border-[3px] border-black bg-white p-5 shadow-[6px_6px_0px_#000000] sm:p-8 rounded-none">
      
      {/* Header Container - Split with a clean structural border */}
      <div className="mb-6 flex flex-col gap-4 border-b-[3px] border-black pb-5 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-mono text-lg font-black uppercase tracking-tight text-black">
            Markdown Preview
          </h2>
          <p className="mt-0.5 font-mono text-xs font-bold text-gray-500">
            Generated article artifact
          </p>
        </div>
        
        {/* Action Button - Heavy Neo-Brutalist Neo Pink Link */}
        <Link href="/blog" className="shrink-0">
          <Button
            className="w-full border-[2px] border-black bg-[#ff007f] px-4 py-2 text-xs font-black uppercase tracking-wider text-white shadow-[4px_4px_0px_#000000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000000] hover:bg-[#ff007f] sm:w-auto rounded-none"
          >
            <ExternalLink className="mr-2 size-4 stroke-[3px]" />
            View Full Blog
          </Button>
        </Link>
      </div>

      {/* Article Content Display Container */}
      <div className="text-black max-w-none">
        <MarkdownRenderer markdown={markdown} images={images} />
      </div>
    </article>
  )
}

export function MarkdownRenderer({
  markdown,
  images = {},
}: {
  markdown: string
  images?: ImageMap
}) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={makeMarkdownComponents(images)}
    >
      {markdown}
    </ReactMarkdown>
  )
}

function makeMarkdownComponents(images: ImageMap): Components {
  return {
    ...markdownComponents,

    img({ alt, src }) {
      const source = typeof src === "string" ? src.trim() : ""

      if (
        !source ||
        source.includes("IMAGE GENERATION FAILED") ||
        source.includes("RESOURCE_EXHAUSTED") ||
        source.includes("ERROR:") ||
        source.includes("Quota")
      ) {
        return null
      }

      const filename = source.replace("images/", "").split("/").pop() || ""
      const base64 = images[filename]
      const resolvedSrc = base64 ? `data:image/png;base64,${base64}` : source

      return (
        <span className="my-6 block overflow-hidden border-[3px] border-black bg-white shadow-[4px_4px_0px_#000000] rounded-none">
          <img
            src={resolvedSrc}
            alt={alt || "Generated image"}
            className="aspect-[16/9] w-full object-cover border-b-[2px] border-black grayscale-[15%] hover:grayscale-0 transition-all"
          />
          {alt ? (
            <span className="block bg-[#fafafa] px-4 py-2.5 font-mono text-xs font-bold text-gray-700">
              {alt}
            </span>
          ) : null}
        </span>
      )
    },
  }
}

const markdownComponents: Components = {
  a({ children, href }) {
    return (
      <a
        className="font-black text-black underline decoration-[#ff007f] decoration-[3px] underline-offset-4 transition-all hover:bg-[#ff007f]/10"
        href={href}
        rel="noreferrer"
        target="_blank"
      >
        {children}
      </a>
    )
  },

  blockquote({ children }) {
    return (
      <blockquote className="my-6 border-l-[4px] border-black bg-[#fce166]/10 px-5 py-4 font-medium text-gray-800 rounded-none italic">
        {children}
      </blockquote>
    )
  },

  code({ children, className }) {
    const match = /language-(\w+)/.exec(className || "")
    const code = String(children).replace(/\n$/, "")

    if (match) {
      return <CodeBlock code={code} language={match[1]} />
    }

    return (
      <code className="border border-black bg-gray-100 px-1.5 py-0.5 font-mono text-[0.9em] font-black text-black rounded-none">
        {children}
      </code>
    )
  },

  h1({ children }) {
    return (
      <h1 className="mb-6 mt-4 font-mono text-3xl font-black uppercase tracking-tight text-black border-b-[2px] border-black pb-2">
        {children}
      </h1>
    )
  },

  h2({ children }) {
    return (
      <h2 className="mb-4 mt-8 border-b-[2px] border-black/20 pb-1 font-mono text-xl font-black uppercase tracking-tight text-black">
        {children}
      </h2>
    )
  },

  h3({ children }) {
    return (
      <h3 className="mb-3 mt-6 font-mono text-base font-black uppercase text-black">
        {children}
      </h3>
    )
  },

  hr() {
    return <hr className="my-8 border-t-[2px] border-black" />
  },

  li({ children }) {
    return <li className="pl-1 font-medium text-black">{children}</li>
  },

  ol({ children }) {
    return (
      <ol className="my-4 list-decimal space-y-2 pl-6 font-mono text-sm font-bold text-black marker:font-black">
        {children}
      </ol>
    )
  },

  p({ children }) {
    return (
      <p className="my-4 text-sm font-medium leading-7 text-gray-800">
        {children}
      </p>
    )
  },

  pre({ children }) {
    return <>{children}</>
  },

  strong({ children }) {
    return (
      <strong className="font-black text-black bg-[#fce166]/30 px-0.5">{children}</strong>
    )
  },

  table({ children }) {
    return (
      <div className="dashboard-scrollbar my-6 overflow-x-auto border-[2px] border-black shadow-[4px_4px_0px_#000000] rounded-none">
        <table className="w-full border-collapse text-left text-sm">{children}</table>
      </div>
    )
  },

  tbody({ children }) {
    return <tbody className="divide-y border-black divide-black/10 bg-white">{children}</tbody>
  },

  td({ children }) {
    return <td className="px-4 py-3 font-medium text-gray-800">{children}</td>
  },

  th({ children }) {
    return (
      <th className="bg-black border-b border-black px-4 py-2.5 font-mono text-xs font-black uppercase tracking-wider text-white">
        {children}
      </th>
    )
  },

  ul({ children }) {
    return (
      <ul className="my-4 list-disc space-y-2 pl-6 text-black marker:text-black">
        {children}
      </ul>
    )
  },
}

function CodeBlock({
  code,
  language,
}: {
  code: string
  language: string
}) {
  const [copied, setCopied] = useState(false)

  async function copyCode() {
    if (typeof navigator === "undefined") return
    await navigator.clipboard.writeText(code)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1200)
  }

  return (
    <div className="my-6 overflow-hidden border-[3px] border-black bg-white shadow-[5px_5px_0px_#000000] rounded-none">
      <div className="flex items-center justify-between border-b-[2px] border-black bg-[#fafafa] px-4 py-2">
        <span className="font-mono text-xs font-black uppercase tracking-wider text-black">
          {language}
        </span>
        <Button
          className={cn(
            "h-7 rounded-none border-[2px] border-black bg-white px-3 font-mono text-[11px] font-black uppercase text-black shadow-[2px_2px_0px_#000000] hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0px_#000000] hover:bg-white",
            copied && "bg-[#fce166] hover:bg-[#fce166]"
          )}
          onClick={copyCode}
          size="sm"
          type="button"
        >
          {copied ? <Check className="size-3.5 stroke-[3px]" /> : <Copy className="size-3.5 stroke-[3px]" />}
          {copied ? "Copied" : "Copy"}
        </Button>
      </div>
      <pre className="dashboard-scrollbar overflow-x-auto bg-[#fafafa] p-4 text-xs font-medium leading-6 text-black">
        <code className="font-mono">{code}</code>
      </pre>
    </div>
  )
}