import axios from "axios"

import type { BlogResponse } from "@/types/blog"

const API = axios.create({
  baseURL: "http://127.0.0.1:8000",
})

export async function generateBlog(topic: string): Promise<BlogResponse> {
  const response = await API.post(
    "/blog/generate",
    {
      topic,
    }
  )

  return response.data
}
