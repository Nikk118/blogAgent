import { Sparkles } from "lucide-react"
import Link from "next/link"

export function Header() {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b border-white/10 bg-black/30 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">
        
        {/* Logo */}
        <div className="flex items-center gap-3">
          <Link href="/" className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-300 text-black shadow-lg shadow-teal-500/20">
            <Sparkles className="h-5 w-5" />
          </Link>

          <div>
            <h1 className="text-sm font-semibold tracking-wide text-white">
              Blog Agent OS
            </h1>

            <p className="text-xs text-zinc-500">
              Autonomous AI Workspace
            </p>
          </div>
        </div>

        {/* Right */}
        <div className="hidden md:flex items-center gap-3">
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
      </div>
    </header>
  )
}