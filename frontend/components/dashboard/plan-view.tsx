import {
  CheckCircle2,
  Code2,
  Compass,
  FileText,
  ListChecks,
  Quote,
  Search,
} from "lucide-react"

import { EmptyState, PanelSkeleton } from "@/components/dashboard/empty-state"
import { cn } from "@/lib/utils"
import type { BlogPlan, ExecutionStatus } from "@/types/blog"

export function PlanView({
  plan,
  status,
}: {
  plan?: BlogPlan | null
  status: ExecutionStatus
}) {
  if (!plan && status === "running") return <PanelSkeleton />

  if (!plan) {
    return (
      <EmptyState
        description="Run the graph to see the editorial structure, task fanout, constraints, and section goals."
        icon={ListChecks}
        title="No plan compiled yet"
      />
    )
  }

  return (
    <div className="space-y-6 text-black">

      {/* Stats Row */}
      <div className="grid gap-4 md:grid-cols-3">
        <PlanStat label="Audience" value={plan.audience} />
        <PlanStat label="Tone"     value={plan.tone} />
        <PlanStat label="Kind"     value={plan.blog_kind.replace("_", " ")} />
      </div>

      {/* Constraints Block */}
      {plan.constraints.length > 0 && (
        <div className="border-[3px] border-black bg-[#f5f0e8] p-5 shadow-[4px_4px_0px_#000]">
          <div className="mb-4 flex items-center gap-2 font-mono text-xs font-black uppercase tracking-widest text-black">
            <Compass className="size-4 stroke-[3px] text-[#ff2d78]" />
            Constraints
          </div>
          <div className="flex flex-wrap gap-2.5">
            {plan.constraints.map((constraint) => (
              <span
                key={constraint}
                className="border-[2px] border-black bg-[#fce135] px-3 py-1 font-mono text-xs font-black uppercase tracking-wider text-black shadow-[2px_2px_0px_#000]"
              >
                {constraint}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Task Cards */}
      <div className="space-y-5">
        {plan.tasks.map((task, index) => (
          <article
            key={task.id}
            className="group border-[3px] border-black bg-[#f5f0e8] p-5 shadow-[6px_6px_0px_#000] transition-all hover:translate-x-[2px] hover:translate-y-[2px] hover:shadow-[2px_2px_0px_#000]"
          >
            {/* Header */}
            <div className="flex flex-col gap-4 border-b-[2px] border-black pb-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex min-w-0 gap-3.5">
                {/* Index badge */}
                <span className="flex size-9 shrink-0 items-center justify-center border-[2px] border-black bg-[#ff2d78] font-mono text-xs font-black text-white shadow-[2px_2px_0px_#000]">
                  {String(index + 1).padStart(2, "0")}
                </span>

                <div className="min-w-0">
                  <h3 className="text-base font-black uppercase tracking-tight text-black sm:text-lg">
                    {task.title}
                  </h3>
                  <p className="mt-1 text-sm font-bold leading-6 text-gray-700">
                    {task.goal}
                  </p>
                </div>
              </div>

              {/* Word count stamp */}
              <span className="self-start border-[2px] border-black bg-black px-3 py-1 font-mono text-[10px] font-black uppercase tracking-wider text-[#c8f135] shadow-[2px_2px_0px_#000]">
                {task.target_words} words
              </span>
            </div>

            {/* Content Grid */}
            <div className="mt-4 grid gap-5 lg:grid-cols-[minmax(0,1fr)_220px]">
              {/* Bullets */}
              <ul className="space-y-2.5">
                {task.bullets.map((bullet) => (
                  <li
                    key={bullet}
                    className="flex items-start gap-2.5 text-sm font-bold leading-6 text-black"
                  >
                    <CheckCircle2 className="mt-1 size-4 shrink-0 stroke-[3px] text-[#ff2d78]" />
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>

              {/* Signal flags */}
              <div className="grid grid-cols-3 gap-2 lg:grid-cols-1 lg:self-start">
                <TaskSignal active={task.requires_research} icon={Search} label="Research"  />
                <TaskSignal active={task.require_citations} icon={Quote}  label="Citations" />
                <TaskSignal active={task.require_code}      icon={Code2}  label="Code"      />
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}

function PlanStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-[3px] border-black bg-[#f5f0e8] p-4 shadow-[4px_4px_0px_#000]">
      <p className="font-mono text-[10px] font-black uppercase tracking-widest text-gray-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm font-black capitalize text-black">
        {value}
      </p>
    </div>
  )
}

function TaskSignal({
  active,
  icon: Icon,
  label,
}: {
  active: boolean
  icon: typeof FileText
  label: string
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 border-[2px] border-black px-3 py-1.5 font-mono text-[11px] font-black uppercase tracking-wider transition-all",
        active
          ? "bg-[#fce135] text-black shadow-[2px_2px_0px_#000]"
          : "bg-black/5 text-gray-400 opacity-50 pointer-events-none"
      )}
    >
      <Icon className={cn("size-3.5", active ? "stroke-[2.5px]" : "stroke-[2px]")} />
      <span>{label}</span>
    </div>
  )
}