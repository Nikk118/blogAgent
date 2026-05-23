export function Footer() {
  return (
    <footer className="relative z-50 border-t border-white/10 bg-[#030407]/95 backdrop-blur-2xl">
      <div className="mx-auto flex min-h-[120px] max-w-7xl flex-col items-center justify-center gap-4 px-6 py-8 text-sm text-zinc-500 md:flex-row md:justify-between">
        
        <p>
          © 2026 Blog Agent OS. Built with AI workflows.
        </p>

        <div className="flex items-center gap-6">
          <span className="cursor-pointer transition-colors hover:text-zinc-300">
            Privacy
          </span>

          <span className="cursor-pointer transition-colors hover:text-zinc-300">
            Docs
          </span>

          <span className="cursor-pointer transition-colors hover:text-zinc-300">
            GitHub
          </span>
        </div>
      </div>
    </footer>
  )
}