import axios from "axios"

import { auth } from "@/lib/firebase"

import type { BlogResponse } from "@/types/blog"

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
})

export async function generateBlog(
  topic: string
): Promise<BlogResponse> {

  const token =
    await auth.currentUser?.getIdToken()

  const response = await API.post(
    "/blog/generate",
    {
      topic,
    },
    {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  )

  return response.data
}