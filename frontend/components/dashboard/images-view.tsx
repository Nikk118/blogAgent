import { Image as ImageIcon, ImagePlus, Sparkles } from "lucide-react"

import { EmptyState, PanelSkeleton } from "@/components/dashboard/empty-state"
import { getImageSpecs } from "@/lib/blog-normalizers"
import type { BlogResult, ExecutionStatus } from "@/types/blog"

export function ImagesView({
  result,
  status,
}: {
  result?: BlogResult
  status: ExecutionStatus
}) {
 const specs = getImageSpecs(result)

const dbImages = (result as any)?.images ?? []

const images = specs.map((spec: any) => {

  const dbImage = dbImages.find(
    (img: any) =>
      img.filename === spec.filename
  )

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
    <div className="grid gap-4 lg:grid-cols-2">
      {images.map((image, index) => (
        <article
          className="group overflow-hidden rounded-2xl border border-white/10 bg-[#1b2a41]/[0.035] transition duration-300 hover:-translate-y-1 hover:border-fuchsia-200 hover:bg-[#1b2a41]/[0.06]"
          key={`${image.filename}-${index}`}
        >
          <div className="relative aspect-[16/9] overflow-hidden border-b border-white/10">

  {image.image_data ? (
    <img
      src={`data:image/png;base64,${image.image_data}`}
      alt={image.alt}
      className="h-full w-full object-cover"
      loading="lazy"
    />
  ) : (
    <>
      <div className="absolute inset-0 runtime-grid opacity-20" />

      <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(45,212,191,0.24),transparent_34%),radial-gradient(circle_at_80%_0%,rgba(217,70,239,0.2),transparent_32%),linear-gradient(135deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02))]" />

      <div className="absolute bottom-4 left-4 flex items-center gap-2 text-[#ffffff]">
        <span className="flex size-10 items-center justify-center rounded-2xl border border-white/10 bg-[#1b2a41]/90 backdrop-blur">
          <ImagePlus className="size-4" />
        </span>

        <span className="text-sm font-medium">
          Visual asset {index + 1}
        </span>
      </div>
    </>
  )}

  <div className="absolute left-4 top-4 rounded-full border border-white/10 bg-[#1b2a41]/90 px-3 py-1 font-mono text-[11px] text-[#e4e4e4]/90 backdrop-blur">
    {image.size || "1024x1024"}
  </div>

  <span className="absolute inset-x-0 top-0 h-24 bg-gradient-to-b from-white/10 to-transparent opacity-0 group-hover:opacity-100 animate-scanline" />
</div>
          <div className="space-y-3 p-4">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <h3 className="line-clamp-2 text-base font-semibold text-[#ffffff]">
                  {image.alt}
                </h3>
                <p className="mt-1 truncate font-mono text-[11px] text-[#e4e4e4]/60">
                  {image.filename}
                </p>
              </div>
              
            </div>
            {image.caption ? (
              <p className="text-sm leading-6 text-[#e4e4e4]/60">{image.caption}</p>
            ) : null}
            {image.prompt ? (
              <div className="rounded-xl border border-white/10 bg-[#1b2a41]/60 p-3">
                <p className="mb-2 flex items-center gap-2 text-[11px] font-medium uppercase text-[#e4e4e4]/60">
                  <Sparkles className="size-3.5" />
                  Prompt
                </p>
                <p className="line-clamp-4 text-xs leading-5 text-[#e4e4e4]/80">
                  {image.prompt}
                </p>
              </div>
            ) : null}
          </div>
        </article>
      ))}
    </div>
  )
}
