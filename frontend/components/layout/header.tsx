"use client"

import { Sparkles } from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"

interface HeaderProps {
  collapsed: boolean
}

export function Header({
  collapsed,
}: HeaderProps) {
  return (
    <header
      className={cn(
        "fixed top-0 right-0 z-30 h-16 border-b border-white/10 bg-[#0a0a0a] transition-all duration-300",
        collapsed
          ? "left-[72px]"
          : "left-[280px]"
      )}
    >
      <div className="flex h-full items-center justify-between px-6">

        {/* Left */}
        <div className="flex items-center gap-3">

         

          <div>
            <Link
            href="/">
            <h1 className="text-sm font-semibold tracking-wide text-white">
              Blog Agent OS
            </h1>
            </Link>

            
          </div>
        </div>

        {/* Right */}
        <Link
          href="/roadmap"
          className="
            rounded-2xl
            border border-white/10
            bg-white/[0.03]
            px-4 py-2
            text-sm text-zinc-300
            transition
            hover:bg-white/[0.06]
            hover:text-white
          "
        >
          Roadmap
        </Link>

      </div>
    </header>
  )
}