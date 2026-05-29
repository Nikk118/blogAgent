
"use client"

import { Sparkles } from "lucide-react"
import { useEffect } from "react"

export function LoadingScreen() {
  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-[#030407]">

      {/* Background Glow */}
      <div className="absolute h-[500px] w-[500px] rounded-full bg-teal-300/10 blur-3xl" />

      <div className="relative flex flex-col items-center">

        {/* Logo */}
        <div className="flex h-20 w-20 items-center justify-center rounded-3xl border border-white/10 bg-teal-300 text-black shadow-[0_0_80px_rgba(45,212,191,0.25)]">

          <Sparkles className="size-10" />
        </div>

        {/* Title */}
        <h1 className="mt-6 text-3xl font-bold text-white">
          Blog Agent OS
        </h1>

        <p className="mt-2 text-zinc-500">
          Initializing workspace...
        </p>

        {/* Loader */}
        <div className="mt-8 h-1 w-64 overflow-hidden rounded-full bg-white/5">

          <div className="h-full w-1/3 animate-[loading_1.2s_ease-in-out_infinite] rounded-full bg-teal-300" />

        </div>
      </div>
    </div>
  )
}