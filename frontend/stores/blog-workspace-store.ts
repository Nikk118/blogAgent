import { create } from "zustand"

import { getResultTitle } from "@/lib/blog-normalizers"

import type {
  BlogResult,
  ExecutionStatus,
  WorkspaceSession,
} from "@/types/blog"

interface BlogWorkspaceState {

  activeSessionId: string | null

  draftTopic: string

  phaseIndex: number

  sessions: WorkspaceSession[]

  status: ExecutionStatus

  setDraftTopic: (
    topic: string
  ) => void

  setSessions: (
    sessions: WorkspaceSession[]
  ) => void

  startGeneration: (
    topic: string
  ) => string

  completeGeneration: (
    id: string,
    result: BlogResult,
    topic: string
  ) => void

  failGeneration: (
    id: string,
    message: string
  ) => void

  selectSession: (
    id: string
  ) => void

  clearSession: () => void

  advancePhase: () => void
}

export const useBlogWorkspaceStore =
  create<BlogWorkspaceState>(
    (set, get) => ({

      activeSessionId: null,

      draftTopic: "",

      phaseIndex: 0,

      sessions: [],

      status: "idle",

      setDraftTopic: (topic) =>
        set({
          draftTopic: topic,
        }),

      // =========================
      // HYDRATE FROM DATABASE
      // =========================

      setSessions: (sessions) =>
        set({
          sessions,
        }),

      // =========================
      // START GENERATION
      // =========================

      startGeneration: (topic) => {

        const id = createSessionId()

        const now =
          new Date().toISOString()

        const session: WorkspaceSession = {

          id,

          title: topic,

          topic,

          createdAt: now,

          updatedAt: now,

          status: "running",
        }

        set((state) => ({

          activeSessionId: id,

          draftTopic: topic,

          phaseIndex: 0,

          sessions: [
            session,
            ...state.sessions,
          ],

          status: "running",
        }))

        return id
      },

      // =========================
      // COMPLETE GENERATION
      // =========================

      completeGeneration: (
        id,
        result,
        topic
      ) => {

        const now =
          new Date().toISOString()

        const title =
          getResultTitle(
            result,
            topic
          )

        set((state) => ({

          phaseIndex: 6,

          sessions: state.sessions.map(
            (session) =>

              session.id === id

                ? {
                    ...session,

                    result,

                    status: "completed",

                    title,

                    topic,

                    updatedAt: now,
                  }

                : session
          ),

          status:
            state.activeSessionId === id
              ? "completed"
              : state.status,
        }))
      },

      // =========================
      // FAIL GENERATION
      // =========================

      failGeneration: (
        id,
        message
      ) => {

        const now =
          new Date().toISOString()

        set((state) => ({

          sessions: state.sessions.map(
            (session) =>

              session.id === id

                ? {
                    ...session,

                    error: message,

                    status: "failed",

                    updatedAt: now,
                  }

                : session
          ),

          status:
            state.activeSessionId === id
              ? "failed"
              : state.status,
        }))
      },

      // =========================
      // SELECT SESSION
      // =========================

      selectSession: (id) => {

        const session =
          get().sessions.find(
            (item) => item.id === id
          )

        if (!session) {
          return
        }

        set({

          activeSessionId: id,

          draftTopic: session.topic,

          phaseIndex:
            session.status === "completed"
              ? 6
              : 0,

          status: session.status,
        })
      },

      // =========================
      // CLEAR ACTIVE SESSION
      // =========================

      clearSession: () =>
        set({

          activeSessionId: null,

          draftTopic: "",

          phaseIndex: 0,

          status: "idle",
        }),

      // =========================
      // ADVANCE PHASE
      // =========================

      advancePhase: () =>
        set((state) => ({

          phaseIndex:
            state.status === "running"

              ? Math.min(
                  state.phaseIndex + 1,
                  5
                )

              : state.phaseIndex,
        })),
    })
  )

function createSessionId() {

  if (
    typeof crypto !== "undefined" &&
    "randomUUID" in crypto
  ) {
    return crypto.randomUUID()
  }

  return `session-${Date.now()}-${Math.random()
    .toString(16)
    .slice(2)}`
}