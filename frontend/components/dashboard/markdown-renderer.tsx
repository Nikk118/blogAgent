"use client"

import Link from "next/link"

import {
  Check,
  Copy,
  ExternalLink,
  FileText,
} from "lucide-react"

import { useState } from "react"

import ReactMarkdown, {
  type Components,
} from "react-markdown"

import {
  EmptyState,
  PanelSkeleton,
} from "@/components/dashboard/empty-state"

import { Button } from "@/components/ui/button"

import { cn } from "@/lib/utils"

import type { ExecutionStatus } from "@/types/blog"

export function MarkdownPreview({
  markdown,
  status,
}: {
  markdown?: string
  status: ExecutionStatus
}) {
  if (!markdown && status === "running") {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-6">
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
    <article className="rounded-2xl border border-white/10 bg-[#07090e]/82 p-5 shadow-[0_30px_120px_rgba(0,0,0,0.34)] sm:p-8">
      
      {/* Top Bar */}
      <div className="mb-6 flex items-center justify-between">
        
        <div>
          <h2 className="text-sm font-medium text-zinc-200">
            Markdown Preview
          </h2>

          <p className="text-xs text-zinc-500">
            Generated article artifact
          </p>
        </div>

        <Link href="/blog">
          <Button
            variant="outline"
            className="border-white/10 bg-white/[0.03] text-zinc-300 hover:bg-white/[0.06] hover:text-white"
          >
            <ExternalLink className="mr-2 size-4" />
            View Full Blog
          </Button>
        </Link>
      </div>

      <MarkdownRenderer markdown={markdown} />
    </article>
  )
}

export function MarkdownRenderer({
  markdown,
}: {
  markdown: string
}) {
  return (
    <ReactMarkdown components={markdownComponents}>
      {markdown}
    </ReactMarkdown>
  )
}

const markdownComponents: Components = {
  a({ children, href }) {
    return (
      <a
        className="font-medium text-teal-200 underline decoration-teal-200/30 underline-offset-4 transition hover:text-teal-100 hover:decoration-teal-100"
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
      <blockquote className="my-6 rounded-2xl border border-teal-300/15 bg-teal-300/[0.055] px-5 py-4 text-zinc-300">
        {children}
      </blockquote>
    )
  },

  code({ children, className }) {
    const match = /language-(\w+)/.exec(
      className || ""
    )

    const code = String(children).replace(
      /\n$/,
      ""
    )

    if (match) {
      return (
        <CodeBlock
          code={code}
          language={match[1]}
        />
      )
    }

    return (
      <code className="rounded-md border border-white/10 bg-white/[0.06] px-1.5 py-0.5 font-mono text-[0.9em] text-teal-100">
        {children}
      </code>
    )
  },

  h1({ children }) {
    return (
      <h1 className="mb-6 mt-0 text-balance text-4xl font-semibold leading-tight text-zinc-50">
        {children}
      </h1>
    )
  },

  h2({ children }) {
    return (
      <h2 className="mb-4 mt-10 border-t border-white/10 pt-7 text-2xl font-semibold tracking-tight text-zinc-50">
        {children}
      </h2>
    )
  },

  h3({ children }) {
    return (
      <h3 className="mb-3 mt-7 text-xl font-semibold text-zinc-100">
        {children}
      </h3>
    )
  },

  hr() {
    return <hr className="my-9 border-white/10" />
  },

  img({ alt, src }) {
    const source =
      typeof src === "string"
        ? src.trim()
        : ""

    // Skip broken image payloads
    if (
      !source ||
      source.includes("IMAGE GENERATION FAILED") ||
      source.includes("RESOURCE_EXHAUSTED") ||
      source.includes("ERROR:") ||
      source.includes("Quota")
    ) {
      return null
    }

    return (
      <figure className="my-8 overflow-hidden rounded-2xl border border-white/10 bg-white/[0.035]">
        
        <img
          src={source}
          alt={alt || "Generated image"}
          className="aspect-[16/9] w-full object-cover"
        />

        {alt ? (
          <figcaption className="border-t border-white/10 px-4 py-3 text-sm text-zinc-500">
            {alt}
          </figcaption>
        ) : null}
      </figure>
    )
  },

  li({ children }) {
    return (
      <li className="pl-1 text-zinc-300">
        {children}
      </li>
    )
  },

  ol({ children }) {
    return (
      <ol className="my-5 list-decimal space-y-2 pl-6 marker:text-teal-200/80">
        {children}
      </ol>
    )
  },

  p({ children }) {
    return (
      <p className="my-4 text-[15px] leading-7 text-zinc-300">
        {children}
      </p>
    )
  },

  pre({ children }) {
    return <>{children}</>
  },

  strong({ children }) {
    return (
      <strong className="font-semibold text-zinc-50">
        {children}
      </strong>
    )
  },

  table({ children }) {
    return (
      <div className="dashboard-scrollbar my-7 overflow-x-auto rounded-2xl border border-white/10">
        <table className="w-full border-collapse text-sm">
          {children}
        </table>
      </div>
    )
  },

  tbody({ children }) {
    return (
      <tbody className="divide-y divide-white/10">
        {children}
      </tbody>
    )
  },

  td({ children }) {
    return (
      <td className="px-4 py-3 text-zinc-400">
        {children}
      </td>
    )
  },

  th({ children }) {
    return (
      <th className="bg-white/[0.04] px-4 py-3 text-left text-xs font-medium uppercase text-zinc-300">
        {children}
      </th>
    )
  },

  ul({ children }) {
    return (
      <ul className="my-5 list-disc space-y-2 pl-6 marker:text-teal-200/80">
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
  const [copied, setCopied] =
    useState(false)

  async function copyCode() {
    if (typeof navigator === "undefined") {
      return
    }

    await navigator.clipboard.writeText(code)

    setCopied(true)

    window.setTimeout(
      () => setCopied(false),
      1200
    )
  }

  return (
    <div className="my-7 overflow-hidden rounded-2xl border border-white/10 bg-[#050609] shadow-[0_24px_80px_rgba(0,0,0,0.35)]">
      
      <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.035] px-4 py-2.5">
        
        <span className="font-mono text-[11px] uppercase text-zinc-500">
          {language}
        </span>

        <Button
          className={cn(
            "h-7 rounded-lg border border-white/10 bg-white/[0.04] px-2 text-[11px] text-zinc-400 hover:bg-white/[0.08] hover:text-zinc-100",
            copied && "text-emerald-200"
          )}
          onClick={copyCode}
          size="sm"
          type="button"
          variant="ghost"
        >
          {copied ? (
            <Check className="size-3.5" />
          ) : (
            <Copy className="size-3.5" />
          )}

          {copied ? "Copied" : "Copy"}
        </Button>
      </div>

      <pre className="dashboard-scrollbar overflow-x-auto p-4 text-[13px] leading-6">
        <code className="font-mono text-zinc-300">
          {code}
        </code>
      </pre>
    </div>
  )
}