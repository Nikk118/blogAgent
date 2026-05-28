"use client"

import { useEffect, useState } from "react"

import { useRouter } from "next/navigation"

import {
  onAuthStateChanged,
  User,
} from "firebase/auth"

import { auth } from "@/lib/firebase"

import { BlogWorkspace } from "@/components/dashboard/blog-workspace"

export default function HomePage() {

  const router = useRouter()

  const [user, setUser] = useState<User | null>(null)

  const [loading, setLoading] = useState(true)

  useEffect(() => {

    const unsubscribe = onAuthStateChanged(
      auth,
      (currentUser) => {

        if (!currentUser) {
          router.push("/login")
        } else {
          setUser(currentUser)
        }

        setLoading(false)
      }
    )

    return () => unsubscribe()

  }, [router])

  if (loading) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-black text-white">
        Loading...
      </main>
    )
  }

  if (!user) {
    return null
  }

  return <BlogWorkspace />
}