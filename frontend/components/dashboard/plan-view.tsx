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
  if (!plan && status === "running") {
    return <PanelSkeleton />
  }

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
    <div className="space-y-5">
      <div className="grid gap-3 md:grid-cols-3">
        <PlanStat label="Audience" value={plan.audience} />
        <PlanStat label="Tone" value={plan.tone} />
        <PlanStat label="Kind" value={plan.blog_kind.replace("_", " ")} />
      </div>

      {plan.constraints.length > 0 ? (
        <div className="rounded-2xl border border-white/10 bg-[#1b2a41]/[0.035] p-4">
          <div className="mb-3 flex items-center gap-2 text-sm font-medium text-[#ffffff]">
            <Compass className="size-4 text-[#f5a31a]" />
            Constraints
          </div>
          <div className="flex flex-wrap gap-2">
            {plan.constraints.map((constraint) => (
              <span
                className="rounded-full border border-white/10 bg-[#1b2a41]/60 px-3 py-1 text-xs text-[#e4e4e4]/80"
                key={constraint}
              >
                {constraint}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      <div className="space-y-3">
        {plan.tasks.map((task, index) => (
          <article
            className="group rounded-2xl border border-white/10 bg-[#1b2a41]/[0.035] p-4 transition duration-300 hover:-translate-y-0.5 hover:border-white/20 hover:bg-[#1b2a41]/[0.055]"
            key={task.id}
          >
            <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
              <div className="flex min-w-0 gap-3">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-xl border border-teal-200 bg-[#f5a31a]/10 font-mono text-xs text-teal-800">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div className="min-w-0">
                  <h3 className="text-base font-semibold text-[#ffffff]">
                    {task.title}
                  </h3>
                  <p className="mt-1 text-sm leading-6 text-[#e4e4e4]/60">
                    {task.goal}
                  </p>
                </div>
              </div>
              <span className="rounded-full border border-white/10 bg-[#1b2a41]/60 px-3 py-1 font-mono text-[11px] text-[#e4e4e4]/80">
                {task.target_words} words
              </span>
            </div>

            <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_220px]">
              <ul className="space-y-2">
                {task.bullets.map((bullet) => (
                  <li
                    className="flex gap-2 text-sm leading-6 text-[#e4e4e4]/80"
                    key={bullet}
                  >
                    <CheckCircle2 className="mt-1 size-3.5 shrink-0 text-emerald-700/80" />
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>

              <div className="grid grid-cols-3 gap-2 lg:grid-cols-1">
                <TaskSignal
                  active={task.requires_research}
                  icon={Search}
                  label="Research"
                />
                <TaskSignal
                  active={task.require_citations}
                  icon={Quote}
                  label="Citations"
                />
                <TaskSignal
                  active={task.require_code}
                  icon={Code2}
                  label="Code"
                />
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
    <div className="rounded-2xl border border-white/10 bg-[#1b2a41]/[0.035] p-4">
      <p className="text-[11px] font-medium uppercase text-[#e4e4e4]/60">{label}</p>
      <p className="mt-2 text-sm font-medium capitalize text-[#ffffff]">
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
        "flex items-center gap-2 rounded-xl border px-2.5 py-2 text-xs",
        active
          ? "border-teal-200 bg-[#f5a31a]/10 text-teal-800"
          : "border-white/10 bg-[#1b2a41]/60 text-[#e4e4e4]/60"
      )}
    >
      <Icon className="size-3.5" />
      <span>{label}</span>
    </div>
  )
}
