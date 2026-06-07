
"use client"

import { Sparkles } from "lucide-react"
import { useEffect } from "react"

export function LoadingScreen() {
return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-white text-black">
      {/* Structural canvas layout blueprint grid pattern */}
      <div className="pointer-events-none absolute inset-0 runtime-grid opacity-[0.06]" />

      {/* Main Focal Loading Card Deck */}
      <div className="relative flex flex-col items-center border-[3px] border-black bg-white p-8 px-12 shadow-[8px_8px_0px_#000000] rounded-none">

        {/* Logo Block - Pure high-contrast vector square */}
        <div className="flex h-20 w-20 items-center justify-center border-[3px] border-black bg-[#ff007f] text-white shadow-[4px_4px_0px_#000000] rounded-none">
          <Sparkles className="size-10 stroke-[2.5px]" />
        </div>

        {/* Brand Header Label */}
        <h1 className="mt-6 font-mono text-2xl font-black uppercase tracking-widest text-black">
          Blog Agent OS
        </h1>

        {/* Sub-status terminal context message */}
        <p className="mt-1 font-mono text-xs font-black uppercase tracking-wider text-gray-600">
          Initializing workspace...
        </p>

        {/* Loading Bar Track - Transformed into an industrial mechanical channel */}
        <div className="mt-8 h-6 w-64 overflow-hidden border-[3px] border-black bg-gray-100 p-1 rounded-none shadow-[4px_4px_0px_#000000]">
          {/* Retained your global 'loading' animation key, wrapped inside flat geometric boundaries */}
          <div className="h-full w-1/3 animate-[loading_1.2s_ease-in-out_infinite] border-r-[2px] border-black bg-[#fce166] rounded-none" />
        </div>
        
      </div>
    </div>
  )
}