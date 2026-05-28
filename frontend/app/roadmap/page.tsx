import Link from "next/link"

import {
  ArrowLeft,
  CheckCircle2,
  CircleDashed,
  Clock3,
  XCircle,
} from "lucide-react"

const completed = [
  "Firebase Authentication",
  "Google Login",
  "Email/Password Login",
  "Protected Routes",
  "Workspace UI",
  "Sidebar Sessions",
  "Blog Export (PDF/Image)",
]

const inProgress = [
  "Persistent Blog Sessions",
  "Improved Export Formatting",
  "Session History",
]

const blocked = [
  {
    title: "AI Image Generation",
    reason: "Google API key not configured",
  },
]

const planned = [
  "PostgreSQL Database",
  "Cloud Sync",
  "Blog Search",
  "Multi-device Persistence",
  "Stripe Billing",
  "Public Blog Sharing",
]

function Section({
  title,
  icon,
  items,
}: {
  title: string
  icon: React.ReactNode
  items: React.ReactNode
}) {
  return (
    <section className="rounded-3xl border border-white/10 bg-white/[0.03] p-6 backdrop-blur-xl">
      <div className="mb-6 flex items-center gap-3">
        {icon}

        <h2 className="text-xl font-semibold text-white">
          {title}
        </h2>
      </div>

      {items}
    </section>
  )
}

export default function RoadmapPage() {
  return (
    <main className="min-h-screen bg-[#09090b] text-zinc-100">

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/10 bg-black/70 backdrop-blur-xl">

        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6">

          <div>
            <h1 className="text-lg font-semibold text-white">
              Blog Agent Roadmap
            </h1>

            <p className="text-sm text-zinc-500">
              Development progress and system status
            </p>
          </div>

          <Link
            href="/"
            className="
              flex items-center gap-2
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
            <ArrowLeft className="size-4" />

            Back to Workspace
          </Link>
        </div>
      </header>

      {/* Content */}
      <section className="mx-auto grid max-w-7xl gap-6 px-6 py-10 lg:grid-cols-2">

        {/* Completed */}
        <Section
          title="Completed"
          icon={
            <CheckCircle2 className="size-6 text-emerald-400" />
          }
          items={
            <div className="space-y-3">
              {completed.map((item) => (
                <div
                  key={item}
                  className="flex items-center gap-3 rounded-2xl border border-emerald-500/10 bg-emerald-500/5 px-4 py-3"
                >
                  <CheckCircle2 className="size-4 text-emerald-400" />

                  <span className="text-sm text-zinc-200">
                    {item}
                  </span>
                </div>
              ))}
            </div>
          }
        />

        {/* In Progress */}
        <Section
          title="In Progress"
          icon={
            <Clock3 className="size-6 text-yellow-400" />
          }
          items={
            <div className="space-y-3">
              {inProgress.map((item) => (
                <div
                  key={item}
                  className="flex items-center gap-3 rounded-2xl border border-yellow-500/10 bg-yellow-500/5 px-4 py-3"
                >
                  <Clock3 className="size-4 text-yellow-400" />

                  <span className="text-sm text-zinc-200">
                    {item}
                  </span>
                </div>
              ))}
            </div>
          }
        />

        {/* Blocked */}
        <Section
          title="Blocked"
          icon={
            <XCircle className="size-6 text-red-400" />
          }
          items={
            <div className="space-y-3">
              {blocked.map((item) => (
                <div
                  key={item.title}
                  className="rounded-2xl border border-red-500/10 bg-red-500/5 p-4"
                >
                  <div className="flex items-center gap-3">
                    <XCircle className="size-4 text-red-400" />

                    <span className="text-sm font-medium text-zinc-100">
                      {item.title}
                    </span>
                  </div>

                  <p className="mt-2 text-sm text-zinc-400">
                    {item.reason}
                  </p>
                </div>
              ))}
            </div>
          }
        />

        {/* Planned */}
        <Section
          title="Planned"
          icon={
            <CircleDashed className="size-6 text-cyan-400" />
          }
          items={
            <div className="space-y-3">
              {planned.map((item) => (
                <div
                  key={item}
                  className="flex items-center gap-3 rounded-2xl border border-cyan-500/10 bg-cyan-500/5 px-4 py-3"
                >
                  <CircleDashed className="size-4 text-cyan-400" />

                  <span className="text-sm text-zinc-200">
                    {item}
                  </span>
                </div>
              ))}
            </div>
          }
        />
      </section>
    </main>
  )
}