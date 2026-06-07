import { Image as ImageIcon, ImagePlus, ChevronDown } from "lucide-react"
import { useState } from "react"
import { EmptyState, PanelSkeleton } from "@/components/dashboard/empty-state"
import { getImageSpecs } from "@/lib/blog-normalizers"
import { cn } from "@/lib/utils"
import type { BlogResult, ExecutionStatus } from "@/types/blog"

function PromptExpander({ prompt }: { prompt: string }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="border-[2px] border-black bg-white shadow-[4px_4px_0px_#000000] transition-all rounded-none">
      <button
        onClick={() => setOpen(!open)}
        className={cn(
          "flex w-full items-center justify-between px-3 py-2.5 text-left transition-colors font-mono text-xs font-black uppercase tracking-wider rounded-none",
          open 
            ? "bg-[#ff007f] text-white border-b-[2px] border-black" 
            : "bg-white text-black hover:bg-[#ff007f]/10"
        )}
        aria-expanded={open}
      >
        <span className="flex items-center gap-2">
          <span className={cn(
            "size-2 border border-black", 
            open ? "bg-white" : "bg-black"
          )} />
          <span className="tracking-widest">
            Image prompt
          </span>
        </span>
        <ChevronDown
          className={cn(
            "size-4 transition-transform duration-200 stroke-[3px]",
            open ? "rotate-180 text-white" : "text-black"
          )}
        />
      </button>

      <div
        className={cn(
          "grid transition-all duration-200 ease-in-out",
          open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
        )}
      >
        <div className="overflow-hidden">
          <p className="bg-[#fafafa] border-t border-black/10 px-4 py-3 font-mono text-xs font-medium leading-5 text-gray-800 break-words">
            {prompt}
          </p>
        </div>
      </div>
    </div>
  )
}

export function ImagesView({
  result,
  status,
}: {
  result?: BlogResult
  status: ExecutionStatus
}) {
  const specs = getImageSpecs(result)
  const dbImages = (result as any)?.generated_images ?? []

  const images = specs.map((spec: any) => {
    const dbImage = dbImages.find((img: any) => img.filename === spec.filename)
    return {
      ...spec,
      image_data: dbImage?.image_data,
    }
  })

  if (images.length === 0 && status === "running") {
    return <PanelSkeleton />
  }

  if (images.length === 0) {
    return (
      <EmptyState
        description="Image specs and markdown placeholders appear here when the reducer requests technical visuals."
        icon={ImageIcon}
        title="No image plan generated"
      />
    )
  }

  return (
  <div className="grid gap-6 lg:grid-cols-2 text-black">
  {images.map((image, index) => (
    <article
      className="group overflow-hidden border-[3px] border-black bg-[#f5f0e8] shadow-[6px_6px_0px_#000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000]"
      key={`${image.filename}-${index}`}
    >
      {/* Image frame */}
      <div className="relative aspect-[16/9] overflow-hidden border-b-[3px] border-black bg-black/5">
        {image.image_data ? (
          <img
            src={`data:image/png;base64,${image.image_data}`}
            alt={image.alt}
            className="h-full w-full object-cover transition-all grayscale-[15%] group-hover:grayscale-0"
            loading="lazy"
          />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-4 select-none">
            <div className="pointer-events-none absolute inset-0 runtime-grid opacity-[0.06]" />
            <div className="relative z-10 flex flex-col items-center gap-2">
              <div className="flex size-12 items-center justify-center border-[2px] border-black bg-[#fce135] text-black shadow-[3px_3px_0px_#000]">
                <ImagePlus className="size-5 stroke-[2.5px]" />
              </div>
              <span className="mt-2 font-mono text-xs font-black uppercase tracking-widest text-black">
                Visual asset {index + 1}
              </span>
            </div>
          </div>
        )}

        {/* Resolution tag */}
        <div className="absolute left-3 top-3 z-20 border-[2px] border-black bg-black px-2 py-0.5 font-mono text-[10px] font-black tracking-wider text-[#c8f135] shadow-[2px_2px_0px_#fce135]">
          {image.size || "1024x1024"}
        </div>
      </div>

      {/* Metadata */}
      <div className="space-y-4 p-5 bg-[#f5f0e8]">
        <div className="min-w-0">
          <h3 className="line-clamp-2 font-mono text-sm font-black uppercase tracking-tight text-black">
            {image.alt}
          </h3>
          <p className="mt-1 truncate font-mono text-[11px] font-bold text-gray-400">
            {image.filename}
          </p>
        </div>

        {image.caption && (
          <p className="border-l-[3px] border-black pl-3 py-0.5 text-xs font-semibold leading-5 text-gray-700">
            {image.caption}
          </p>
        )}

        {image.prompt && (
          <div className="pt-1">
            <PromptExpander prompt={image.prompt} />
          </div>
        )}
      </div>
    </article>
  ))}
</div>
  )
}