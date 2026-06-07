export function Footer() {
  return (
    // Replaced bg-white with a cream background tone
    <footer className="relative z-50 border-t-[3px] border-black bg-[#f5f2e9] text-black">
      <div className="mx-auto flex min-h-[100px] max-w-7xl flex-col items-center justify-center gap-6 px-6 py-8 md:flex-row md:justify-between">
        
        {/* Copyright Brand Stamp */}
        <p className="font-mono text-xs font-black uppercase tracking-widest text-gray-800 text-center md:text-left">
          © 2026 Blog Agent OS. Built with AI workflows.
        </p>

        {/* Action Link Clusters - Using bg-[#f5f2e9] to blend with footer */}
        <div className="flex flex-wrap items-center justify-center gap-3">
          <span 
            className="cursor-pointer border-[2px] border-black bg-[#f5f2e9] px-3 py-1 font-mono text-xs font-black uppercase tracking-wider shadow-[2px_2px_0px_#000000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:bg-[#fce166] hover:shadow-[1px_1px_0px_#000000]"
          >
            Privacy
          </span>

          <span 
            className="cursor-pointer border-[2px] border-black bg-[#f5f2e9] px-3 py-1 font-mono text-xs font-black uppercase tracking-wider shadow-[2px_2px_0px_#000000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:bg-[#ff007f] hover:text-white hover:shadow-[1px_1px_0px_#000000]"
          >
            Docs
          </span>

          <span 
            className="cursor-pointer border-[2px] border-black bg-[#f5f2e9] px-3 py-1 font-mono text-xs font-black uppercase tracking-wider shadow-[2px_2px_0px_#000000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:bg-[#fce166] hover:shadow-[1px_1px_0px_#000000]"
          >
            GitHub
          </span>
        </div>
        
      </div>
    </footer>
  )
}