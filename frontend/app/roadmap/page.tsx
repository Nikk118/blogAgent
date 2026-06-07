import Link from "next/link"
import {
  ArrowLeft,
  CheckCircle2,
  CircleDashed,
  Clock3,
  Sparkles,
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
  "Persistent Blog Sessions",
  "Improved Export Formatting",
  "Session History",
  "PostgreSQL Database",
  "Cloud Sync",
  "AI Image Generation",
  "Multi-Agent LangGraph Pipeline",
  "RAG / FAISS Evidence System",
  "LangSmith Tracing",
  "Neo-Brutalist UI Redesign",
]

const inProgress = [
  "Blog Search",
  "Multi-device Persistence",
  "Public Blog Sharing",
]

const planned = [
  "Stripe Billing & SaaS Tiers",
  "Usage Limits per Plan",
  "Team Workspaces",
  "Custom Domain Blog Publishing",
  "API Access for Developers",
  "Webhook Integrations",
  "White-label Export",
]

const blocked: { title: string; reason: string }[] = []

export default function RoadmapPage() {
  return (
    <main className="min-h-screen bg-[#f5f0e8] text-black">

      {/* Header */}
      <header className="sticky top-0 z-50 border-b-[3px] border-black bg-[#f5f0e8]">
        <div
          className="absolute top-0 left-0 right-0 h-[4px]"
          style={{
            background: "repeating-linear-gradient(90deg,#c8f135 0,#c8f135 16px,#000 16px,#000 20px)",
          }}
        />
        <div className="flex h-16 items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex size-8 items-center justify-center border-[2px] border-black bg-black">
              <Sparkles className="size-4 text-[#c8f135] stroke-[2.5px]" />
            </div>
            <span
              className="uppercase tracking-[0.1em] text-black leading-none"
              style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "20px" }}
            >
              Blog Agent OS
            </span>
            <div className="hidden sm:flex items-center ml-2 pl-4 border-l-[2px] border-black">
              <span className="inline-flex items-center border-[2px] border-black bg-[#fce135] px-2 py-0.5 font-mono text-[9px] font-black uppercase tracking-wider text-black shadow-[2px_2px_0px_#000]">
                Roadmap
              </span>
            </div>
          </div>

          <Link
            href="/"
            className="flex items-center gap-2 border-[2px] border-black bg-white px-3 py-1.5 font-mono text-xs font-black uppercase tracking-wider text-black shadow-[2px_2px_0px_#000] transition-all hover:translate-x-[1px] hover:translate-y-[1px] hover:shadow-[1px_1px_0px_#000] hover:bg-[#fce135]"
          >
            <ArrowLeft className="size-4 stroke-[2.5px]" />
            Back
          </Link>
        </div>
      </header>

      {/* Page title */}
      <div className="border-b-[3px] border-black bg-black px-6 py-5">
        <div className="mx-auto max-w-7xl flex items-end justify-between">
          <div>
            <h1
              className="uppercase leading-none text-[#c8f135]"
              style={{ fontFamily: "var(--font-bebas), sans-serif", fontSize: "42px" }}
            >
              Development Roadmap
            </h1>
            <p className="font-mono text-[11px] font-black uppercase tracking-widest text-gray-500 mt-1">
              System status · Feature pipeline · SaaS milestones
            </p>
          </div>
          <span className="hidden sm:flex items-center gap-2 border-[2px] border-[#c8f135] px-3 py-1.5 font-mono text-[10px] font-black uppercase tracking-wider text-[#c8f135]">
            <span className="w-2 h-2 bg-[#00e676] border-[2px] border-[#c8f135]" />
            Active Development
          </span>
        </div>
      </div>

      {/* Grid */}
      <section className="mx-auto grid max-w-7xl gap-6 px-6 py-8 lg:grid-cols-2">

        {/* Completed */}
        <RoadmapSection
          title="Completed"
          count={completed.length}
          icon={<CheckCircle2 className="size-5 stroke-[2.5px]" />}
          headerBg="bg-[#c8f135]"
          headerText="text-black"
        >
          {completed.map((item) => (
            <div
              key={item}
              className="flex items-center gap-3 border-b-[1px] border-black/10 py-2.5 last:border-0"
            >
              <CheckCircle2 className="size-4 shrink-0 stroke-[2.5px] text-black" />
              <span className="font-mono text-xs font-bold text-black">{item}</span>
            </div>
          ))}
        </RoadmapSection>

        {/* In Progress */}
        <RoadmapSection
          title="In Progress"
          count={inProgress.length}
          icon={<Clock3 className="size-5 stroke-[2.5px]" />}
          headerBg="bg-[#fce135]"
          headerText="text-black"
        >
          {inProgress.map((item) => (
            <div
              key={item}
              className="flex items-center gap-3 border-b-[1px] border-black/10 py-2.5 last:border-0"
            >
              <Clock3 className="size-4 shrink-0 stroke-[2.5px] text-black" />
              <span className="font-mono text-xs font-bold text-black">{item}</span>
            </div>
          ))}
        </RoadmapSection>

        {/* Planned / SaaS */}
        <RoadmapSection
          title="Planned — SaaS"
          count={planned.length}
          icon={<CircleDashed className="size-5 stroke-[2.5px]" />}
          headerBg="bg-[#ff2d78]"
          headerText="text-white"
        >
          {planned.map((item) => (
            <div
              key={item}
              className="flex items-center gap-3 border-b-[1px] border-black/10 py-2.5 last:border-0"
            >
              <CircleDashed className="size-4 shrink-0 stroke-[2.5px] text-black" />
              <span className="font-mono text-xs font-bold text-black">{item}</span>
            </div>
          ))}
        </RoadmapSection>

        {/* Blocked — empty state */}
        <RoadmapSection
          title="Blocked"
          count={0}
          icon={<XCircle className="size-5 stroke-[2.5px]" />}
          headerBg="bg-black"
          headerText="text-white"
        >
          <div className="flex flex-col items-center justify-center py-10 gap-2">
            <span className="font-mono text-xs font-black uppercase tracking-widest text-gray-400">
              Nothing blocked
            </span>
            <span className="w-8 h-[2px] bg-black/20" />
          </div>
        </RoadmapSection>

      </section>
    </main>
  )
}

function RoadmapSection({
  title,
  count,
  icon,
  headerBg,
  headerText,
  children,
}: {
  title: string
  count: number
  icon: React.ReactNode
  headerBg: string
  headerText: string
  children: React.ReactNode
}) {
  return (
    <section className="border-[3px] border-black bg-[#f5f0e8] shadow-[6px_6px_0px_#000]">
      {/* Section header */}
      <div className={`flex items-center justify-between border-b-[3px] border-black px-4 py-3 ${headerBg}`}>
        <div className={`flex items-center gap-2 font-mono text-xs font-black uppercase tracking-widest ${headerText}`}>
          {icon}
          {title}
        </div>
        <span className={`border-[2px] border-black bg-white px-2 py-0.5 font-mono text-[10px] font-black text-black shadow-[2px_2px_0px_#000]`}>
          {String(count).padStart(2, "0")}
        </span>
      </div>

      {/* Items */}
      <div className="px-5 py-2">
        {children}
      </div>
    </section>
  )
}