"use client"

import { Sparkles } from "lucide-react"
import Link from "next/link"
import { cn } from "@/lib/utils"

interface HeaderProps {
  collapsed: boolean
}

export function Header({ collapsed }: HeaderProps) {
  return (
    <header
      className={cn(
        "fixed top-0 right-0 z-30 h-16 border-b-[3px] border-black bg-[#f5f0e8] transition-all duration-300",
        collapsed ? "left-[72px]" : "left-[280px]"
      )}
    >
      {/* Hatch stripe accent - same as hero */}
      <div
        className="absolute top-0 left-0 right-0 h-[4px]"
        style={{
          background: "repeating-linear-gradient(90deg,#c8f135 0,#c8f135 16px,#000 16px,#000 20px)",
        }}
      />

      <div className="flex h-full items-center justify-between px-6">

        {/* Left - Brand */}
        <div className="flex items-center gap-3">
          {/* Black icon box */}
          <div className="flex size-8 items-center justify-center border-[2px] border-black bg-black">
            <Sparkles className="size-4 text-[#c8f135] stroke-[2.5px]" />
          </div>

          <Link href="/">
            <h1
              className="text-black uppercase tracking-[0.12em] hover:text-[#ff2d78] transition-colors"
              style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "22px" }}
            >
              Blog Agent OS
            </h1>
          </Link>

         
          
        </div>

        {/* Right - Roadmap CTA */}
        <Link
          href="/roadmap"
          className="
            border-[2px] border-black
            bg-[#fce135]
            px-4 py-1.5
            font-mono text-[10px] font-black uppercase tracking-widest text-black
            shadow-[3px_3px_0px_#000]
            transition-all
            hover:translate-x-[1px] hover:translate-y-[1px]
            hover:bg-[#ff2d78] hover:text-white
            hover:shadow-[2px_2px_0px_#000]
          "
        >
          Roadmap
        </Link>

      </div>
    </header>
  )
}